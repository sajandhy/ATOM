"""DeepSeek-V4 DSpark draft wrapper for SGLang external model loading.

SGLang's DSpark worker (``DSparkWorkerV2``) loads the draft model under the
architecture name ``DeepseekV4ForCausalLMDSpark`` (SGLang rewrites the HF config
``architectures[0]`` to this name when it detects ``dspark_block_size`` in the
draft checkpoint config and ``--speculative-algorithm DSPARK`` is set).

For V4-Flash-0731 the draft is **inline**: the three DSpark backbone layers live
under the same checkpoint as the target (``mtp.{0,1,2}.*``).  SGLang loads the
draft as a separate worker with ``speculative_draft_model_path`` pointing to the
same checkpoint, then calls ``set_embed_and_head`` to share the target's embedding
and LM-head weights.

## Interface contract

``DraftBlockProposer`` and ``TargetHiddenKvInjector`` expect:

* ``markov_head``         — adapts ATOM's ``DSparkMarkovHead`` to SGLang's
                            ``VanillaMarkov`` protocol (``apply_step_logits``,
                            ``sample_block``).
* ``gamma``               — int, ``dspark_block_size`` (5 for V4-Flash-0731).
* ``confidence_head``     — wrapped ``DSparkConfidenceHead`` or ``None``.
* ``attach_shared_modules``  — bind target embed_tokens + lm_head.
* ``forward_embed``       — embed input_ids with the shared embedding.
* ``compute_base_logits`` — normed hidden [B*T, dim] → base_logits [B*T, vocab].
* ``write_target_hidden_kv(main_hidden, swa_loc, positions, pool)``
                          — inject target hidden into draft SWA window.
                            This signature is the ``_inject_mla`` path, triggered
                            because ``ATOMDeepSeekV4ProxyKVPool`` now exposes
                            ``set_swa_key_buffer_radix_fused_norm_rope`` as a
                            routing sentinel.
* ``forward(...)``        — SGLang model-runner entry: returns a
                            ``LogitsProcessorOutput`` whose ``.hidden_states`` is
                            ``[B * gamma, hidden_size]`` for the Markov sampler.

## KV injection

ATOM's ``DSparkLayer.write_context_kv`` needs the ATOM forward context (for
``cu_seqlens_q`` / ``state_slot_out``), which is not set when
``TargetHiddenKvInjector`` calls us.  Instead we bypass ``write_context_kv`` and
write to the SWA plane directly: for each token we compute the projected KV
row, derive the ring slot from ``swa_loc`` (``swa_loc = slot * ring_slots + pos
% ring_slots``), and scatter-write it.  This is equivalent to what ``swa_write``
does but operates on individual token locations instead of seqlen-tagged spans.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, Optional, Tuple

import torch
from sglang.srt.distributed import get_pp_group
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.server_args import get_global_server_args
from torch import nn

from atom.config import SpeculativeConfig
from atom.plugin.config import generate_atom_config_for_plugin_mode
from atom.plugin.sglang.runtime import (
    SGLangForwardBatchMetadata,
    SGLangPluginRuntime,
    plugin_runtime_scope,
)

logger = logging.getLogger("atom.plugin.sglang.models")


# ---------------------------------------------------------------------------
# Markov head adapter
# ---------------------------------------------------------------------------

class _SGLangMarkovHeadAdapter(nn.Module):
    """Wrap ATOM's ``DSparkMarkovHead`` to satisfy SGLang's ``VanillaMarkov`` protocol.

    SGLang's ``DraftBlockProposer`` drives the Markov correction loop with:
        - ``markov_head.apply_step_logits(logits, token_ids, hidden_states)``
        - ``markov_head.sample_block(base_logits, first_prev_tokens,
                                     hidden_states, sampler)``

    ATOM's ``DSparkMarkovHead`` uses ``nn.Embedding`` for both W1 and W2 while
    SGLang's ``VanillaMarkov`` has W2 as ``nn.Linear``.  This adapter bridges
    the two without moving parameters.
    """

    markov_head_type = "vanilla"

    def __init__(self, atom_markov_head: nn.Module) -> None:
        super().__init__()
        self._head = atom_markov_head

    def get_prev_embeddings(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self._head.markov_w1(token_ids.long())

    def compute_step_bias(
        self,
        token_ids: torch.Tensor,
        hidden_states: Optional[torch.Tensor],
    ) -> torch.Tensor:
        del hidden_states
        embed = self._head.markov_w1(token_ids.long())  # [*, r]
        # Cache the fp32 markov_w2 transpose once. The naive per-call
        # `markov_w2.weight.float().t()` allocates a [vocab, r] fp32 temporary
        # (~131 MB for V4-Flash) on EVERY step; under CUDA-graph capture that
        # temporary is baked into each bs bucket's graph pool (×gamma×buckets)
        # and OOMs. The first call runs during eager warmup, so the cache lands
        # in the normal allocator and every captured graph just reads it.
        w2t = getattr(self, "_w2_f32_t", None)
        if w2t is None:
            w2t = self._head.markov_w2.weight.float().t().contiguous()
            self._w2_f32_t = w2t
        return torch.matmul(embed.float(), w2t)

    def apply_step_logits(
        self,
        logits: torch.Tensor,
        *,
        token_ids: torch.Tensor,
        hidden_states: Optional[torch.Tensor],
    ) -> torch.Tensor:
        return logits + self.compute_step_bias(token_ids, hidden_states)

    def apply_block_logits(
        self,
        base_logits: torch.Tensor,
        *,
        token_ids: torch.Tensor,
        hidden_states: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if base_logits.size(-2) == 0:
            return base_logits
        return base_logits + self.compute_step_bias(token_ids, hidden_states)

    def sample_block(
        self,
        base_logits: torch.Tensor,
        *,
        first_prev_tokens: torch.Tensor,
        hidden_states: Optional[torch.Tensor],
        sampler,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, proposal_len = base_logits.shape[:2]
        if proposal_len == 0:
            empty = torch.empty(
                batch_size, 0, dtype=torch.long, device=base_logits.device
            )
            return empty, base_logits

        sampled_tokens = []
        corrected_logits = []
        prev_tokens = first_prev_tokens.long()
        for step_idx in range(proposal_len):
            step_logits = self.apply_step_logits(
                base_logits[:, step_idx, :],
                token_ids=prev_tokens,
                hidden_states=None,
            )
            next_tokens = sampler(step_logits, step_idx)
            sampled_tokens.append(next_tokens)
            corrected_logits.append(step_logits.unsqueeze(1))
            prev_tokens = next_tokens
        return (
            torch.stack(sampled_tokens, dim=1),
            torch.cat(corrected_logits, dim=1),
        )


# ---------------------------------------------------------------------------
# Confidence head adapter
# ---------------------------------------------------------------------------

class _SGLangConfidenceHeadAdapter(nn.Module):
    """Adapt ATOM's ``DSparkConfidenceHead`` to SGLang's interface."""

    def __init__(self, atom_confidence_head: nn.Module) -> None:
        super().__init__()
        self._head = atom_confidence_head
        self.register_buffer(
            "sts_temperatures",
            torch.ones((), dtype=torch.float32),
            persistent=False,
        )
        self._last_confidence_raw: Optional[torch.Tensor] = None
        # SGLang's DSparkVerifyPlanner checks confidence_head.with_markov.
        # ATOM's DSparkConfidenceHead always uses the Markov embedding.
        self.with_markov = True

    def forward(
        self,
        hidden_states: torch.Tensor,
        markov_embed_stack: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if markov_embed_stack is None:
            raise ValueError("DSpark confidence head requires markov_embed_stack")
        return self._head(hidden_states, markov_embed_stack)

    def apply_sts(self, confidence_raw: torch.Tensor) -> torch.Tensor:
        self._last_confidence_raw = confidence_raw
        return torch.sigmoid(confidence_raw.float() / self.sts_temperatures)


# ---------------------------------------------------------------------------
# SWA plane write helper (bypasses write_context_kv for out-of-context inject)
# ---------------------------------------------------------------------------

def _write_main_kv_to_swa_plane(
    layer,
    main_hidden: torch.Tensor,
    swa_loc: torch.Tensor,
    positions: torch.Tensor,
) -> None:
    """Write target hidden states into one DSpark layer's SWA ring plane.

    Called from ``write_target_hidden_kv`` outside any ATOM forward context.

    ``swa_loc``: int64 SWA pool token locations; -1 = skip.

    We always use the bf16 path for the direct scatter (fake-quantize through
    fp8 E4M3, store as bf16) because the fp8 2buff packing layout is designed
    for the contiguous ring writer (``swa_write``) which handles block alignment
    internally.  Using fake_quant=True is correct for QAT-trained DSpark: the
    draft was trained with quantization noise, and the bf16 window still
    carries the properly round-tripped values.
    """
    n_tokens = int(main_hidden.shape[0])
    if n_tokens == 0:
        return

    a = layer.attn
    plane_rows = int(a.swa_kv.view(-1, a.swa_kv.shape[-1]).shape[0])
    # Guard: locs must be in [0, plane_rows). An out-of-range index_put_ faults
    # the GPU (manifests as a hang under HIP), so drop OOB rows defensively.
    in_range = (swa_loc >= 0) & (swa_loc < plane_rows)
    oob = (swa_loc >= plane_rows)
    if bool(oob.any().item()):
        logger.warning(
            "[DSINJ-OOB] dropped %d/%d rows past plane_rows=%d (max_loc=%d) — "
            "draft SWA plane undersized; draft READ would fault",
            int(oob.sum().item()), int(swa_loc.numel()), plane_rows,
            int(swa_loc.max().item()),
        )
    valid_mask = in_range
    if not bool(valid_mask.any().item()):
        return

    valid_hidden = main_hidden[valid_mask]
    valid_positions = positions[valid_mask]
    valid_locs = swa_loc[valid_mask]

    # Always project with fake_quant=True (bf16 output regardless of kv dtype).
    # This avoids the 2buff packing mismatch in the direct-scatter path.
    main_kv = layer._compute_main_kv(
        valid_hidden, valid_positions, fake_quant=True
    )  # [N_valid, head_dim]  bf16

    # If _compute_main_kv returned more rows than valid_locs (head expansion),
    # collapse: [T*h, D] -> [T, h, D] -> mean over h.
    n_locs = int(valid_locs.shape[0])
    if main_kv.shape[0] != n_locs and n_locs > 0:
        ratio = main_kv.shape[0] // n_locs
        if ratio > 1 and main_kv.shape[0] == n_locs * ratio:
            main_kv = main_kv.view(n_locs, ratio, main_kv.shape[-1]).mean(1)
        else:
            main_kv = main_kv[:n_locs]

    # swa_kv is the flat bf16 SWA ring view [plane_rows, head_dim].
    # Scatter the per-token KV into the valid locations.
    plane = a.swa_kv.view(-1, a.swa_kv.shape[-1])
    if main_kv.shape[-1] != plane.shape[-1]:
        # Truncate or pad to match allocated plane head_dim.
        if main_kv.shape[-1] > plane.shape[-1]:
            main_kv = main_kv[:, :plane.shape[-1]]
        else:
            main_kv = torch.nn.functional.pad(main_kv, (0, plane.shape[-1] - main_kv.shape[-1]))
    plane.index_put_((valid_locs,), main_kv.to(plane.dtype), accumulate=False)


# ---------------------------------------------------------------------------
# Dummy lm_head passthrough
# ---------------------------------------------------------------------------

class _LogitsHeadPassthrough(nn.Module):
    """Dummy lm_head that passes hidden states through unchanged.

    SGLang's ``LogitsProcessor._compute_lm_head`` first checks for
    ``set_lora`` + ``apply_lora`` (LoRA branch) and calls ``forward`` if
    both are present.  We implement both as no-ops to take that branch,
    which routes to ``self(hidden_states)`` = our pass-through forward.
    """

    quant_method = None
    org_vocab_size = None  # filled in by wrapper __init__

    def set_lora(self, *args, **kwargs) -> None:
        pass

    def apply_lora(self, *args, **kwargs) -> None:
        pass

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states


# ---------------------------------------------------------------------------
# Main DSpark draft wrapper
# ---------------------------------------------------------------------------

class DeepseekV4ForCausalLMDSpark(nn.Module):
    """SGLang-compatible DSpark draft wrapper backed by ATOM's ``DeepseekV4DSpark``.

    Registered as ``DeepseekV4ForCausalLMDSpark`` in the ATOM external model
    package, overriding SGLang's native class of the same name.

    SGLang's ``DraftBlockProposer._run_forward`` calls ``forward_batch_generation``
    through the draft runner.  The runner calls ``forward()``, which runs ATOM's
    parallel DSpark backbone and returns hidden states via
    ``logits_output.hidden_states``.  The proposer then calls
    ``compute_base_logits`` → ``markov_head.sample_block``.
    """

    sglang_skip_quant_config = True

    @staticmethod
    def _resolve_local_model_path(model_path: str) -> str:
        """Resolve a HF Hub model ID to its local snapshot path."""
        import os

        if os.path.isdir(model_path):
            return model_path
        try:
            import glob

            hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
            cache_dir = os.path.join(hf_home, "hub")
            safe_name = "models--" + model_path.replace("/", "--")
            snaps = glob.glob(os.path.join(cache_dir, safe_name, "snapshots", "*"))
            if snaps:
                return snaps[0]
        except Exception:
            pass
        return model_path

    def _resolve_dspark_stage_count(self, draft_model_path: str) -> None:
        """Ensure hf_config.dspark_num_layers is set before DeepseekV4DSpark init.

        ``_count_dspark_stages`` reads ``model.safetensors.index.json`` from the
        local path.  When ``draft_model_path`` is a HF Hub ID, it resolves the
        local snapshot and counts the stages, falling back to 3 (V4-Flash-0731).
        """
        hf_config = self.atom_config.hf_config
        if getattr(hf_config, "dspark_num_layers", None):
            return  # already set

        local_path = self._resolve_local_model_path(draft_model_path)
        from atom.models.deepseek_v4_dspark import _count_dspark_stages

        count = _count_dspark_stages(local_path, default=3)
        hf_config.dspark_num_layers = count
        logger.info("DSpark stage count resolved: %d (from %s)", count, local_path)

    def __init__(
        self,
        config,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        del prefix, quant_config
        super().__init__()

        logger.info("Initializing ATOM DSpark draft for %s", self.__class__.__name__)

        self.pp_group = get_pp_group()
        self.config = config

        with plugin_runtime_scope(framework="sglang"):
            self.atom_config = generate_atom_config_for_plugin_mode(config)

        server_args = get_global_server_args()
        draft_model_path = (
            server_args.speculative_draft_model_path or server_args.model_path
        )
        self.atom_config.model = draft_model_path

        # hf_config_override renames architectures → ATOM's DSpark naming.
        SpeculativeConfig.hf_config_override(
            self.atom_config.hf_config, model_path=draft_model_path
        )

        # _count_dspark_stages needs a local filesystem path to read
        # model.safetensors.index.json.  Resolve the HF Hub local snapshot
        # path and set dspark_num_layers=3 as the fallback so construction
        # succeeds even when the path is a Hub ID (not a local dir).
        self._resolve_dspark_stage_count(draft_model_path)

        with plugin_runtime_scope(framework="sglang", atom_config=self.atom_config):
            from atom.models.deepseek_v4_dspark import DeepseekV4DSpark
            from atom.plugin.register import (
                init_aiter_dist,
                register_ops_to_sglang,
                set_attn_cls,
            )

            register_ops_to_sglang(atom_config=self.atom_config)
            set_attn_cls()
            init_aiter_dist(config=self.atom_config)

            self.model = DeepseekV4DSpark(config=self.atom_config)
            self.model.atom_config = self.atom_config

        # gamma: number of draft tokens per block.
        self.gamma = int(getattr(config, "dspark_block_size", 5))

        # Expose ATOM's heads through SGLang's protocol.
        last_stage = self.model.model.mtp[-1]
        self.markov_head = _SGLangMarkovHeadAdapter(last_stage.markov_head)
        self.confidence_head = _SGLangConfidenceHeadAdapter(last_stage.confidence_head)

        self.logits_processor = LogitsProcessor(config, skip_all_gather=True)
        self._logits_head = _LogitsHeadPassthrough()
        self._logits_head.org_vocab_size = int(config.vocab_size)

    # ------------------------------------------------------------------
    # Shared weights (SGLang calls these after loading target weights)
    # ------------------------------------------------------------------

    def attach_shared_modules(
        self, *, embed_tokens: nn.Module, lm_head: nn.Module
    ) -> None:
        self.model.model.embed = embed_tokens
        self.model.model.head = lm_head
        # DSpark CUDA-graph draft-sampler capture reads
        # `draft_model.lm_head.org_vocab_size` (dspark_draft_sampler.py:63/145).
        # The shared head must expose it; ParallelLMHead usually does, but set a
        # fallback so graph capture never trips the __getattr__ guard.
        if not hasattr(lm_head, "org_vocab_size"):
            try:
                lm_head.org_vocab_size = int(self.config.vocab_size)
            except Exception:  # noqa: BLE001 - best-effort attribute set
                pass

    @property
    def lm_head(self):
        """The shared LM head, exposed under SGLang's draft-model attribute name.

        DSpark's CUDA-graph path (dspark_worker_v2 / dspark_draft_sampler) reads
        `draft_model.lm_head.{weight,org_vocab_size}`. ATOM's DSpark backbone
        keeps it at `model.model.head`; expose it here so graph capture finds it.
        """
        return self.model.model.head

    def get_embed_and_head(self):
        return self.model.model.embed.weight, self.model.model.head.weight

    def set_embed_and_head(self, embed, head):
        self.model.model.embed.weight = embed
        self.model.model.head.weight = head

    def set_embed(self, embed):
        self.model.model.embed.weight = embed

    # ------------------------------------------------------------------
    # SGLang DSpark draft protocol
    # ------------------------------------------------------------------

    def forward_embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Embed input_ids with the shared target embedding."""
        return self.model.model.embed(input_ids.long())

    def compute_base_logits(
        self, hidden: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Project normed backbone hidden [B*T, dim] → base_logits [B*T, vocab]."""
        head = self.model.model.head
        weight = head.weight
        if hidden.dtype != weight.dtype:
            hidden = hidden.to(weight.dtype)
        from sglang.srt.distributed.communication_op import (
            tensor_model_parallel_all_gather,
        )

        local_logits = torch.matmul(hidden, weight.T)
        base_logits = tensor_model_parallel_all_gather(local_logits, dim=-1)
        vocab_size = int(getattr(head, "org_vocab_size", base_logits.shape[-1]))
        return base_logits[..., :vocab_size], None

    def _allocate_dspark_swa_windows(self, num_slots: int, device: torch.device) -> None:
        """Allocate and bind standalone bf16 SWA windows for each DSpark draft layer.

        In SGLang plugin mode, ATOM's ``allocate_kv_cache`` does not run for the
        draft model, so the DSpark backbone's per-layer SWA ring planes
        (``a.swa_plane``, ``a.swa_window``) are never bound.  This method
        creates them from scratch using the DSpark layer's own ``window_size``
        and ``gamma`` parameters.
        """
        from dataclasses import dataclass

        @dataclass(frozen=True)
        class _WindowParams:
            ring_start: int
            slot_rows: int
            ring_slots: int
            ring_stride: int
            run_rows: int

        ring_slots = self.model.window_size + self.gamma  # window + max_spec_steps
        # swa_kv is the bf16 flat view that _write_main_kv_to_swa_plane indexes.
        # Layout: [num_slots * ring_slots, head_dim] contiguous.
        head_dim = int(self.config.hidden_size // self.config.num_attention_heads) * int(
            getattr(self.config, "num_key_value_heads", 1)
        )
        # DSpark MLA uses the compressed KV head dim.
        # For V4: qk_nope_head_dim=128, qk_rope_head_dim=64, head_dim=576 compressed → 192
        # The layer's _compute_main_kv returns [T, head_dim] where head_dim=qk_nope+rope.
        # Read it from the layer's actual attn module linears.
        for layer in self.model.model.mtp:
            a = layer.attn

            # Read head_dim from the attention module directly — for DSpark MLA,
            # _compute_main_kv returns [T, a.head_dim] where a.head_dim is
            # the full kv_lora_rank (512 for V4-Flash), not qk_nope + qk_rope.
            layer_head_dim = getattr(a, "head_dim", None)
            if layer_head_dim is None:
                layer_head_dim = int(getattr(a, "kv_lora_rank", 512))

            swa_plane = torch.zeros(
                num_slots * ring_slots,
                layer_head_dim,
                dtype=torch.bfloat16,
                device=device,
            )
            a.swa_plane = swa_plane
            a.swa_plane_rope = None
            a.kv_fp8 = False
            # CRITICAL: dspark_attention dispatches its KV READ on
            # `a.unified_kv_rope is not None` (deepseek_v4_dspark.py:802), NOT on
            # `kv_fp8`. We store bf16 in swa_plane, so the read MUST take the bf16
            # path — force unified_kv_rope=None or the draft reinterprets our bf16
            # window as fp8 2buff and reads garbage (backbone predicts noise).
            a.unified_kv_rope = None
            # swa_kv is the flat ring view used by direct scatter writes.
            a.swa_kv = swa_plane
            # Bind WindowParams: simple contiguous ring, slot size = ring_slots rows.
            a.swa_window = _WindowParams(
                ring_start=0,
                slot_rows=ring_slots,
                ring_slots=ring_slots,
                ring_stride=1,
                run_rows=1,
            )
            a.swa_cache_size = ring_slots

        self._dspark_swa_allocated = True
        logger.info(
            "DSpark SWA windows allocated: %d slots x %d ring_slots x head_dim per layer",
            num_slots,
            ring_slots,
        )

    def write_target_hidden_kv(
        self,
        main_hidden: torch.Tensor,
        swa_loc: torch.Tensor,
        positions: torch.Tensor,
        pool: Any,
        state_slot: torch.Tensor | None = None,
        final_pos: torch.Tensor | None = None,
    ) -> None:
        """Inject target hidden states into the DSpark rolling KV window.

        Called by ``TargetHiddenKvInjector`` via the ``_inject_mla`` path
        (triggered because ``ATOMDeepSeekV4ProxyKVPool`` exposes
        ``set_swa_key_buffer_radix_fused_norm_rope`` as a routing sentinel).

        ``main_hidden``: either
          - ``[T, hidden_size]`` — a single layer's hidden states (all stages
            get the same hidden), or
          - ``[T, hidden_size * num_target_layers]`` — concatenated per-layer
            hidden states produced by the target's ``set_dspark_layers_to_capture``
            forward hook; sliced and distributed to each backbone stage.
        ``swa_loc``: int32 SWA pool token locations in the TARGET proxy pool's
            arena geometry (negative = skip). Only used as a fallback when
            ``state_slot`` is not supplied.
        ``positions``: [T] absolute token positions.
        ``state_slot``: [T] per-token draft req_pool_index (the ring slot the
            draft reads from). When provided, the destination row is computed in
            the DRAFT's own ring geometry — ``slot * draft_ring + pos % draft_ring``
            — instead of using ``swa_loc`` directly. This is required for correct
            injection: ``swa_loc`` lives in the target proxy pool's SWA arena
            (ring size ``pool.swa_cache_size``), which differs from the draft's
            standalone plane (ring size ``window_size + gamma``).
        ``final_pos``: [T] last absolute position of each token's request. Used to
            drop tokens older than the draft window so they do not race for a ring
            slot (``pos % draft_ring``) during a long prefill chunk.
        """
        if main_hidden is None or main_hidden.numel() == 0:
            return

        # Lazy SWA window allocation on first inject.
        # We do NOT call bind_deepseek_v4_proxy_cache_views here — the DSpark
        # draft layers need their own private rolling SWA window, not the target
        # proxy pool views. Allocate standalone bf16 planes on first call.
        #
        # Size num_slots to cover every possible state_slot (draft req_pool_index).
        # The proxy pool's num_slots == max_num_reqs, which can exceed max_num_seqs;
        # under-sizing here makes dest_row = slot*ring + ... run past the plane and
        # fault the GPU (a silent hang). Prefer the pool's num_slots.
        if not getattr(self, "_dspark_swa_allocated", False):
            # state_slot == draft req_pool_index, which ranges over the req-to-token
            # pool (max_num_reqs), NOT just num_slots or max_num_seqs. Under-sizing
            # makes the DRAFT READ gather past the plane and fault the GPU. Cover the
            # widest of every candidate bound.
            num_slots = max(
                int(getattr(pool, "max_num_reqs", 0) or 0),
                int(getattr(pool, "num_slots", 0) or 0),
                int(getattr(self.atom_config, "max_num_seqs", 256)),
            )
            device = main_hidden.device
            self._allocate_dspark_swa_windows(num_slots=num_slots, device=device)

        positions_64 = positions.to(dtype=torch.int64)

        if state_slot is not None:
            # Compute the destination row in the draft's OWN ring geometry.
            # The draft reads its plane at slot * draft_ring + pos % draft_ring
            # (see _allocate_dspark_swa_windows / DSparkLayer swa_window), so the
            # write must use the same layout — NOT the target-pool swa_loc.
            draft_ring = int(self.model.window_size + self.gamma)
            slot_64 = state_slot.to(dtype=torch.int64)
            dest_row = slot_64 * draft_ring + (positions_64 % draft_ring)
            # Drop tokens older than the draft window: within one long prefill
            # chunk, tokens sharing a ring slot (pos % draft_ring) would race, so
            # keep only the last window_size positions per request.
            win = int(self.model.window_size)
            if final_pos is not None:
                keep = positions_64 > (final_pos.to(dtype=torch.int64) - win)
                dest_row = torch.where(keep, dest_row, dest_row.new_full((), -1))
        else:
            # Fallback: use the target-pool swa_loc directly (only correct when the
            # draft plane geometry matches the target arena, e.g. unified_kv).
            dest_row = swa_loc.to(dtype=torch.int64)

        stages = list(self.model.model.mtp)

        # Project concatenated target hidden states through stage-0 main_proj +
        # main_norm before passing to wqkv_a. _compute_main_kv expects [T, dim]
        # (post-projection), not the raw concatenated [T, dim * n_target_layers].
        # project_context() owns main_proj/main_norm and runs once; the result
        # is shared across all stages (each stage has its own wqkv_a / kv cache).
        projected = self.model.project_context(main_hidden)  # [T, dim]

        for layer in stages:
            _write_main_kv_to_swa_plane(layer, projected, dest_row, positions_64)

    # ------------------------------------------------------------------
    # SGLang model-runner forward
    # ------------------------------------------------------------------

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        **kwargs,
    ):
        """Run the DSpark parallel backbone and return hidden states.

        ``DraftBlockProposer._run_forward`` calls this through the draft runner.
        The forward batch has ``forward_mode == TARGET_VERIFY`` and
        ``spec_algorithm == DSPARK``.

        Returns ``LogitsProcessorOutput`` whose ``.hidden_states`` is
        ``[B * gamma, hidden_size]`` — the post-norm backbone hidden state.
        ``DraftBlockProposer`` calls ``compute_base_logits`` then
        ``markov_head.sample_block`` on it.
        """
        del input_embeds, kwargs

        with plugin_runtime_scope(framework="sglang", atom_config=self.atom_config):
            with SGLangPluginRuntime(
                atom_config=self.atom_config,
                forward_batch=forward_batch,
                positions=positions,
                input_ids=input_ids,
            ) as runtime:
                # Do NOT bind proxy pool views for DSpark draft — the draft
                # uses its own standalone SWA windows (see _allocate_dspark_swa_windows),
                # not the target's proxy pool planes.
                metadata = SGLangForwardBatchMetadata.build(runtime.forward_batch)
                with SGLangForwardBatchMetadata.bind(metadata):
                    bs = int(forward_batch.batch_size)
                    # anchor_ids: [B] — the bonus/last-verified token per request,
                    # which is the first entry in the [B * gamma] flattened block.
                    # anchor_positions: [B] — each request's anchor position =
                    # positions[0], positions[gamma], positions[2*gamma], ...
                    all_input_ids = runtime.input_ids   # [B * gamma]
                    all_positions = runtime.positions    # [B * gamma]
                    anchor_ids = all_input_ids[::self.gamma][:bs]      # [B]
                    anchor_positions = all_positions[::self.gamma][:bs] # [B]

                    # Ensure the standalone SWA windows exist BEFORE the backbone
                    # runs. Normally allocated lazily on the first inject, but under
                    # CUDA-graph capture the draft forward runs with no preceding
                    # inject, so dspark_attention would read a.swa_window=None.
                    if not getattr(self, "_dspark_swa_allocated", False):
                        from atom.plugin.sglang.deepseek_v4_bridge import (
                            maybe_get_proxy_pool_from_sglang_backend,
                        )

                        _pool, _ = maybe_get_proxy_pool_from_sglang_backend()
                        _ns = max(
                            int(getattr(_pool, "max_num_reqs", 0) or 0),
                            int(getattr(_pool, "num_slots", 0) or 0),
                            int(getattr(self.atom_config, "max_num_seqs", 256)),
                        )
                        self._allocate_dspark_swa_windows(
                            num_slots=_ns, device=input_ids.device
                        )

                    # ATOM DSpark backbone: [B] anchor → [B*gamma, dim] normed hidden.
                    # _DSparkInner.forward returns (normed [B*T, dim], hc [B,T,dim]).
                    normed_hidden, _hc_hidden = self.model.model(
                        anchor_ids, anchor_positions, self.gamma
                    )

        if self.pp_group.is_last_rank:
            from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode

            # DraftBlockProposer._run_forward sets capture_hidden_mode=NULL, but
            # needs logits_output.hidden_states to contain the backbone normed
            # hidden. Force FULL capture so the logits processor stores them.
            orig_capture_mode = getattr(forward_batch, "capture_hidden_mode", None)
            try:
                forward_batch.capture_hidden_mode = CaptureHiddenMode.FULL
                return self.logits_processor(
                    input_ids,
                    normed_hidden,
                    self._logits_head,
                    forward_batch,
                    hidden_states_before_norm=normed_hidden,
                )
            finally:
                if orig_capture_mode is not None:
                    forward_batch.capture_hidden_mode = orig_capture_mode
        return normed_hidden

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        del weights
        from atom.model_loader.loader import load_model

        server_args = get_global_server_args()
        draft_model_path = (
            server_args.speculative_draft_model_path or server_args.model_path
        )
        self.atom_config.model = draft_model_path
        with plugin_runtime_scope(framework="sglang", atom_config=self.atom_config):
            return load_model(
                model=self.model,
                model_name_or_path=draft_model_path,
                hf_config=self.atom_config.hf_config,
                load_dummy=self.atom_config.load_dummy,
                spec_decode=True,
            )


EntryClass = [DeepseekV4ForCausalLMDSpark]
