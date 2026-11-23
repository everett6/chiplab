from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class OperatorSpec(BaseModel):
    name: str
    op_type: str
    flops: float
    bytes_read: float
    bytes_write: float
    preferred_chiplet_type: str | None = None
    parallelism: float = 1.0


BYTES_PER_PARAM = {"FP32": 4.0, "TF32": 4.0, "FP16": 2.0, "BF16": 2.0, "FP8": 1.0,
                   "INT8": 1.0, "FP6": 0.75, "INT4": 0.5, "FP4": 0.5}


def llm_param_counts(workload: WorkloadSpec) -> tuple[float, float]:
    """Return (total, active-per-token) parameter counts for an LLM workload."""
    d, layers = workload.hidden_dim, workload.num_layers
    moe = "moe" in workload.type.lower() and workload.num_experts > 1
    experts = workload.num_experts if moe else 1
    active = min(workload.active_experts, experts) if moe else 1
    attn, ffn = 4.0 * d * d, 8.0 * d * d  # QKVO + two 4d FFN matrices per expert
    total = layers * (attn + ffn * experts)
    act = layers * (attn + ffn * active)
    if workload.model_params_b:
        scale = workload.model_params_b * 1e9 / total
        total, act = total * scale, act * scale
    return total, act


def is_llm_decode(workload: WorkloadSpec) -> bool:
    t = workload.type.lower()
    return "llm" in t and "prefill" not in t


def _llm_operator(workload: WorkloadSpec) -> OperatorSpec:
    """One step of LLM inference: weight + KV-cache traffic and matmul/attention FLOPs.

    Decode: one new token for each of batch_size sequences with sequence_length context.
    Prefill: batch_size prompts of sequence_length tokens.
    """
    b, s, d, layers = (workload.batch_size, workload.sequence_length, workload.hidden_dim,
                       workload.num_layers)
    bpp = BYTES_PER_PARAM.get(workload.precision.upper(), 1.0)
    total, active = llm_param_counts(workload)
    kv_bytes_per_token = 2.0 * layers * d * bpp * workload.kv_cache_fraction
    if is_llm_decode(workload):
        flops = b * (2.0 * active + 4.0 * layers * d * s)
        # MoE: the batch touches more experts than one token does, capped at all of them.
        frac = min(1.0, b * workload.active_experts / max(workload.num_experts, 1))
        touched = active + (total - active) * frac
        bytes_read = touched * bpp + b * s * kv_bytes_per_token
        bytes_write = b * kv_bytes_per_token
        name = "llm_decode_step"
    else:
        flops = b * s * (2.0 * active + 2.0 * layers * d * s)
        bytes_read = total * bpp
        bytes_write = b * s * kv_bytes_per_token
        name = "llm_prefill"
    return OperatorSpec(name=name, op_type="matrix_multiply", flops=flops,
                        bytes_read=bytes_read, bytes_write=bytes_write,
                        preferred_chiplet_type="tensor_engine", parallelism=float(b))


def expand_workload_operators(workload: WorkloadSpec) -> List[OperatorSpec]:
    """Synthesize representative operators for the given workload."""
    operators: List[OperatorSpec] = []

    for op in workload.operators:
        operators.append(op)

    if not workload.operators and "llm" in workload.type.lower():
        operators.append(_llm_operator(workload))
    elif not workload.operators:
        t = workload.type.lower()

        if "transformer" in t or "attention" in t or "decode" in t:
            operators.append(
                OperatorSpec(
                    name="attention_qkv",
                    op_type="matrix_multiply",
                    flops=2.0 * workload.hidden_dim * workload.sequence_length * workload.batch_size,
                    bytes_read=3 * workload.hidden_dim * workload.batch_size,
                    bytes_write=workload.hidden_dim * workload.batch_size,
                    preferred_chiplet_type="tensor_engine",
                    parallelism=float(workload.batch_size),
                )
            )
            operators.append(
                OperatorSpec(
                    name="attention_output",
                    op_type="matrix_multiply",
                    flops=2.0 * workload.hidden_dim * workload.sequence_length * workload.batch_size,
                    bytes_read=workload.hidden_dim * workload.batch_size,
                    bytes_write=workload.hidden_dim * workload.batch_size,
                    preferred_chiplet_type="tensor_engine",
                    parallelism=float(workload.batch_size),
                )
            )

        if "moe" in t:
            operators.append(
                OperatorSpec(
                    name="moe_gating",
                    op_type="dot_product",
                    flops=workload.hidden_dim * workload.num_experts * workload.batch_size,
                    bytes_read=workload.hidden_dim * workload.num_experts * workload.batch_size,
                    bytes_write=workload.batch_size,
                    preferred_chiplet_type="systolic_array",
                    parallelism=float(workload.num_experts),
                )
            )
            operators.append(
                OperatorSpec(
                    name="moe_expert_compute",
                    op_type="matrix_multiply",
                    # workload.flops/bytes are totals for the step: all active experts
                    # together (dividing by active_experts used to drop work).
                    flops=workload.flops,
                    bytes_read=workload.bytes_read,
                    bytes_write=workload.bytes_write,
                    preferred_chiplet_type="systolic_array",
                    parallelism=float(workload.active_experts),
                )
            )

        if "ssm" in t:
            operators.append(
                OperatorSpec(
                    name="ssm_step",
                    op_type="convolution",
                    flops=workload.hidden_dim * workload.sequence_length * workload.batch_size,
                    bytes_read=workload.hidden_dim * workload.batch_size,
                    bytes_write=workload.hidden_dim * workload.batch_size,
                    preferred_chiplet_type="vector_processor",
                    parallelism=float(workload.sequence_length),
                )
            )

        if "cnn" in t:
            operators.append(
                OperatorSpec(
                    name="conv2d",
                    op_type="convolution",
                    flops=workload.flops,
                    bytes_read=workload.bytes_read,
                    bytes_write=workload.bytes_write,
                    preferred_chiplet_type="systolic_array",
                    parallelism=workload.parallelism,
                )
            )

    return operators


class WorkloadSpec(BaseModel):
    name: str = "default_workload"
    type: str = "moe_decode"
    batch_size: int = 1
    sequence_length: int = 128
    hidden_dim: int = 4096
    num_layers: int = 32
    num_experts: int = 8
    active_experts: int = 2
    precision: str = "INT8"
    operators: List[OperatorSpec] = Field(default_factory=list)
    flops: float = 0.0
    bytes_read: float = 0.0
    bytes_write: float = 0.0
    # LLM workloads (type "llm_decode", "llm_prefill", "moe_llm_decode", ...): total model
    # parameters in billions. If omitted it is derived from hidden_dim and num_layers.
    model_params_b: float | None = None
    # KV-cache size relative to full multi-head attention: kv_heads / attention_heads
    # (grouped-query attention), e.g. 0.125 for a 64-head model with 8 KV heads.
    kv_cache_fraction: float = 1.0
