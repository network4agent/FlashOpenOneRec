"""
FlashPrefill patch for vLLM 0.13.x with Qwen3 (dense) support.

Based on the FlashPrefill vLLM 0.12 patch. vLLM 0.13's FlashAttentionImpl.forward
has the same signature as 0.12, so the patch is largely identical.

Usage:
    Before importing vLLM / before the vLLM engine starts, do:
        from flash_prefill.patches import patch_loader_vllm_0_13

    Or call apply_patch_to_worker() explicitly.

IMPORTANT:
    - enable_chunked_prefill must be False
    - enable_prefix_caching must be False
"""
import os
import torch
import types

from flash_prefill.ops import flash_prefill_varlen_func, flash_prefill

import vllm.model_executor.model_loader as vllm_loader
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionMetadata,
    cascade_attention,
    FlashAttentionBackend,
)
from vllm.attention.backends.abstract import AttentionLayer, AttentionType
from vllm.attention.utils.fa_utils import reshape_and_cache_flash, flash_attn_varlen_func


ATTENTION_CONFIG = {
    "flashprefill_llama": {
        "block_size": 128,
        "attention_sink": 2,
        "window_size": 4,
        "alpha": 0.18,
        "last_n_full_block": 2,
    },
    "flashprefill_qwen2": {
        "block_size": 128,
        "attention_sink": 2,
        "window_size": 4,
        "alpha": 0.08,
        "last_n_full_block": 2,
    },
    "flashprefill_qwen3": {
        "block_size": 128,
        "attention_sink": 2,
        "window_size": 4,
        "alpha": 0.10,
        "last_n_full_block": 2,
    },
    "flashprefill_qwen3moe": {
        "block_size": 128,
        "attention_sink": 2,
        "window_size": 4,
        "alpha": 0.12,
        "last_n_full_block": 2,
    },
}

# ---- Choose config here: change to match your model ----
cfg = ATTENTION_CONFIG["flashprefill_qwen3"]


def apply_patch_to_worker(worker=None):
    import vllm.v1.attention.backends.flash_attn as fa_backend

    rank = str(getattr(worker, "rank", os.environ.get("RANK", "0")))
    cache_dir = f"/tmp/triton_cache_rank_{rank}"
    os.makedirs(cache_dir, exist_ok=True)
    os.environ["TRITON_CACHE_DIR"] = cache_dir

    print(f"[FlashPrefill] Triton cache dir: {cache_dir}")

    def flash_prefill_forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with FlashPrefill for prefill, standard FA for decode."""
        assert output is not None, "Output tensor must be provided."

        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "fused output quantization is not yet supported for FlashAttentionImpl"
            )

        if attn_metadata is None:
            return output.fill_(0)

        attn_type = self.attn_type

        num_actual_tokens = attn_metadata.num_actual_tokens

        if attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            return self._forward_encoder_attention(
                query[:num_actual_tokens],
                key[:num_actual_tokens],
                value[:num_actual_tokens],
                output[:num_actual_tokens],
                attn_metadata,
                layer,
            )

        key_cache, value_cache = kv_cache.unbind(0)

        if (
            self.kv_sharing_target_layer_name is None
            and key is not None
            and value is not None
        ):
            reshape_and_cache_flash(
                key,
                value,
                key_cache,
                value_cache,
                attn_metadata.slot_mapping,
                self.kv_cache_dtype,
                layer._k_scale,
                layer._v_scale,
            )

        if self.kv_cache_dtype.startswith("fp8"):
            dtype = FlashAttentionBackend.get_fp8_dtype_for_flashattn(
                self.kv_cache_dtype
            )
            key_cache = key_cache.view(dtype)
            value_cache = value_cache.view(dtype)

        if not attn_metadata.use_cascade:
            cu_seqlens_q = attn_metadata.query_start_loc
            seqused_k = attn_metadata.seq_lens
            max_seqlen_q = attn_metadata.max_query_len
            max_seqlen_k = attn_metadata.max_seq_len
            block_table = attn_metadata.block_table
            scheduler_metadata = attn_metadata.scheduler_metadata

            descale_shape = (cu_seqlens_q.shape[0] - 1, self.num_kv_heads)

            is_prefill = attn_metadata.max_query_len > 1

            if self.dcp_world_size > 1:
                self._forward_with_dcp(
                    query[:num_actual_tokens],
                    key[:num_actual_tokens],
                    value[:num_actual_tokens],
                    key_cache,
                    value_cache,
                    output[:num_actual_tokens],
                    attn_metadata,
                    q_descale=layer._q_scale.expand(descale_shape),
                    k_descale=layer._k_scale.expand(descale_shape),
                    v_descale=layer._v_scale.expand(descale_shape),
                )
                return output
            else:
                if is_prefill:
                    # === FlashPrefill: block-sparse attention for prefill ===
                    flash_prefill_varlen_func(
                        query[:num_actual_tokens],
                        key[:num_actual_tokens],
                        value[:num_actual_tokens],
                        output[:num_actual_tokens],
                        cu_seqlens_q,
                        max_seqlen_q,
                        cfg["alpha"],
                        cfg["block_size"],
                        cfg["attention_sink"],
                        cfg["window_size"],
                        cfg["last_n_full_block"],
                    )
                    return output
                else:
                    # === Standard FlashAttention for decode ===
                    flash_attn_varlen_func(
                        q=query[:num_actual_tokens],
                        k=key_cache,
                        v=value_cache,
                        out=output[:num_actual_tokens],
                        cu_seqlens_q=cu_seqlens_q,
                        max_seqlen_q=max_seqlen_q,
                        seqused_k=seqused_k,
                        max_seqlen_k=max_seqlen_k,
                        softmax_scale=self.scale,
                        causal=attn_metadata.causal,
                        alibi_slopes=self.alibi_slopes,
                        window_size=self.sliding_window,
                        block_table=block_table,
                        softcap=self.logits_soft_cap,
                        scheduler_metadata=scheduler_metadata,
                        fa_version=self.vllm_flash_attn_version,
                        q_descale=layer._q_scale.expand(descale_shape),
                        k_descale=layer._k_scale.expand(descale_shape),
                        v_descale=layer._v_scale.expand(descale_shape),
                        num_splits=attn_metadata.max_num_splits,
                        s_aux=self.sinks,
                    )
                    return output

        cascade_attention(
            output[:num_actual_tokens],
            query[:num_actual_tokens],
            key_cache,
            value_cache,
            cu_query_lens=attn_metadata.query_start_loc,
            max_query_len=attn_metadata.max_query_len,
            cu_prefix_query_lens=attn_metadata.cu_prefix_query_lens,
            prefix_kv_lens=attn_metadata.prefix_kv_lens,
            suffix_kv_lens=attn_metadata.suffix_kv_lens,
            max_kv_len=attn_metadata.max_seq_len,
            softmax_scale=self.scale,
            alibi_slopes=self.alibi_slopes,
            sliding_window=self.sliding_window,
            logits_soft_cap=self.logits_soft_cap,
            block_table=attn_metadata.block_table,
            common_prefix_len=attn_metadata.common_prefix_len,
            max_num_splits=attn_metadata.max_num_splits,
            fa_version=self.vllm_flash_attn_version,
            prefix_scheduler_metadata=attn_metadata.prefix_scheduler_metadata,
            suffix_scheduler_metadata=attn_metadata.scheduler_metadata,
            q_descale=layer._q_scale,
            k_descale=layer._k_scale,
            v_descale=layer._v_scale,
            s_aux=self.sinks,
        )
        return output

    fa_backend.FlashAttentionImpl.forward = flash_prefill_forward
    print(f"[FlashPrefill] Successfully patched Rank {rank} for vLLM 0.13")


apply_patch_to_worker()
