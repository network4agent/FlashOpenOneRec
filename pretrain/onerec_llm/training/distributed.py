"""Distributed training utilities for FSDP/TP model sharding and checkpoint loading."""

from typing import Any, Dict, Optional

import torch
import torch.distributed as dist
from torch import nn
from onerec_llm.utils.ds_utils import format_dict_or_list
from onerec_llm.utils.distributed import get_world_size_and_rank

from torch.distributed._composable.fsdp import CPUOffloadPolicy, fully_shard, MixedPrecisionPolicy
from torch.distributed._tensor import distribute_tensor
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
)

def apply_tp(
    model: nn.Module,
    tp_mesh: DeviceMesh,
    model_class: str = 'Qwen3ForCausalLM',
) -> None:
    """Apply Tensor Parallelism to transformer decoder layers.
    
    Parallelizes attention projections and MLP layers following the Megatron-style
    column/row split: column-parallel on the first linear (q/k/v/gate/up) and
    row-parallel on the second (o/down).
    
    Args:
        model: Model to parallelize.
        tp_mesh: Device mesh for the TP dimension.
        model_class: Model class name.
    """
    tp_plan = {
        "self_attn.q_proj": ColwiseParallel(),
        "self_attn.k_proj": ColwiseParallel(),
        "self_attn.v_proj": ColwiseParallel(),
        "self_attn.o_proj": RowwiseParallel(),
        "mlp.gate_proj": ColwiseParallel(),
        "mlp.up_proj": ColwiseParallel(),
        "mlp.down_proj": RowwiseParallel(),
    }

    if model_class == 'Qwen3ForCausalLM':
        for layer in model.model.layers:
            parallelize_module(layer, tp_mesh, tp_plan)
    else:
        raise ValueError(f"Unsupported model_class for TP: {model_class}")


def shard_model(
    model: nn.Module,
    *,
    cpu_offload: bool,
    reshard_after_forward: bool = True,
    dp_mesh: Optional[DeviceMesh] = None,
    fp32_weight: bool = True,
    model_class: str = 'Qwen3ForCausalLM',
    fp32_reduce: bool = True
) -> None:
    """Shard a model with FSDP using the PyTorch Distributed fully_shard API.
    
    Args:
        model: Model to shard with FSDP.
        cpu_offload: If True, FSDP will offload parameters to CPU.
        reshard_after_forward: Whether to reshard after forward pass.
        dp_mesh: Device mesh for FSDP sharding under multiple parallelism.
        fp32_weight: If True, use fp32 for weights with bfloat16 params.
        model_class: Model class name. Currently only supports 'Qwen3ForCausalLM'.
        fp32_reduce: If True, use fp32 for gradient reduction.
    """
    fsdp_kwargs = {"reshard_after_forward": reshard_after_forward, "mesh": dp_mesh}
    
    if fp32_weight:
        fsdp_kwargs["mp_policy"] = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32 if fp32_reduce else torch.bfloat16
        )
    if cpu_offload:
        fsdp_kwargs["offload_policy"] = CPUOffloadPolicy()

    if model_class == 'Qwen3ForCausalLM':
        layers = list(model.model.layers)
    else:
        raise ValueError(f"Unsupported model_class: {model_class}")
    
    for layer in layers:
        fully_shard(layer, **fsdp_kwargs)
    
    fully_shard(model, **fsdp_kwargs)

    # Set up forward prefetch for layers
    prev = None
    for layer in reversed(layers):
        if prev is not None:
            layer.set_modules_to_forward_prefetch([prev])
        prev = layer
    model.set_modules_to_forward_prefetch([prev])


def load_from_full_model_state_dict(
    model: "FSDPModule",
    full_sd: Dict[str, Any],
    allow_random_init_params: Optional[str] = None,
    use_tie_weights: bool = False
) -> None:
    """Load full state dict into an FSDP-sharded model.
    
    Args:
        model: FSDP-sharded model to load into.
        full_sd: Full (unsharded) state dictionary.
        allow_random_init_params: Comma-separated parameter names to randomly initialize
            if not found in full_sd. Default: None.
        use_tie_weights: If True, tie lm_head.weight to model.embed_tokens.weight.
    """
    if isinstance(allow_random_init_params, str):
        allow_random_init_params = allow_random_init_params.split(',')
    
    meta_sharded_sd = model.state_dict()
    sharded_sd = {}
    
    if dist.get_rank() == 0:
        if use_tie_weights:
            full_sd['lm_head.weight'] = full_sd['model.embed_tokens.weight']

        extra_meta_sharded_sd = set(meta_sharded_sd.keys()) - set(full_sd.keys())
        extra_full_ds = set(full_sd.keys()) - set(meta_sharded_sd.keys())
        
        extra_meta_sharded_sd = {
            k: (v.shape, v.device, v.dtype) 
            for k, v in meta_sharded_sd.items() 
            if k in extra_meta_sharded_sd
        }
        extra_full_ds = {
            k: (v.shape, v.device, v.dtype) 
            for k, v in full_sd.items() 
            if k in extra_full_ds
        }

        device0 = full_sd[list(full_sd)[0]]
        for k in extra_meta_sharded_sd:
            if allow_random_init_params is not None and k in allow_random_init_params:
                full_sd[k] = torch.rand(extra_meta_sharded_sd[k][0]) * 0.1
                if full_sd[k].ndim >= 2:
                    nn.init.kaiming_normal_(full_sd[k], a=0, mode='fan_in', nonlinearity='relu')
                else:
                    nn.init.zeros_(full_sd[k])
                full_sd[k] = full_sd[k].to(device0)

        assert len(meta_sharded_sd) == len(full_sd), (
            f"Sharded State Dict doesn't equal to Full State Dict, "
            f"{len(meta_sharded_sd)} vs {len(full_sd)}\n"
            f"extra_meta_sharded_sd={format_dict_or_list(extra_meta_sharded_sd)}, "
            f"extra_full_ds={format_dict_or_list(extra_full_ds)}"
        )
        assert sorted(list(meta_sharded_sd.keys())) == sorted(list(full_sd.keys())), \
            "Keys of Sharded State Dict doesn't equal to Full State Dict"

    for param_name, sharded_meta_param in meta_sharded_sd.items():
        if dist.get_rank() == 0:
            full_tensor = full_sd[param_name].detach().cuda().type(sharded_meta_param.dtype)
        else:
            full_tensor = torch.empty(
                sharded_meta_param.size(),
                device="cuda",
                dtype=sharded_meta_param.dtype,
            )
        
        mesh = sharded_meta_param.device_mesh
        # Use WORLD group for broadcast so all ranks (across both TP and DP dims) receive the full tensor
        dist.broadcast(full_tensor, src=0)
        dist.barrier()
        
        sharded_tensor = distribute_tensor(
            full_tensor, mesh, sharded_meta_param.placements
        )
        sharded_sd[param_name] = nn.Parameter(sharded_tensor)

    model.load_state_dict(sharded_sd, assign=True)
