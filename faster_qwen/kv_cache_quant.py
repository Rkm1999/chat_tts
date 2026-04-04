"""
kv_cache_quant.py - Int8 KV cache quantization for faster_qwen3_tts StaticCache.

Replaces bfloat16 KV tensors with int8 + float32 per-token scales,
saving ~48% KV VRAM (1.03 bytes effective vs 2 bytes bfloat16).

Must be called after torch.compile, before warmup/CUDA graph capture.
"""
import torch


class QuantizedStaticLayer:
    """
    Drop-in replacement for transformers StaticLayer that stores K/V in int8 +
    float32 per-token absmax scale.  Mirrors StaticLayer's lazy-initialization
    pattern so it can be installed before warmup/CUDA graph capture.

    Quantization: per-token, per-head, absmax symmetric.
    All four tensors are pre-allocated at lazy_initialization time with fixed
    shapes so CUDA graph replay sees the same static addresses.
    """

    is_compileable = True
    is_sliding = False

    def __init__(self, max_cache_len: int):
        self.max_cache_len = max_cache_len
        self.is_initialized = False

    def lazy_initialization(self, key_states: torch.Tensor):
        self.max_batch_size, self.num_heads, _, self.head_dim = key_states.shape
        self.dtype = key_states.dtype
        self.device = key_states.device

        shape       = (self.max_batch_size, self.num_heads, self.max_cache_len, self.head_dim)
        scale_shape = (self.max_batch_size, self.num_heads, self.max_cache_len, 1)

        self.keys        = torch.zeros(shape,       dtype=torch.int8,    device=self.device)
        self.values      = torch.zeros(shape,       dtype=torch.int8,    device=self.device)
        self.keys_scale  = torch.ones(scale_shape,  dtype=torch.float32, device=self.device)
        self.values_scale= torch.ones(scale_shape,  dtype=torch.float32, device=self.device)

        # Mark as static so CUDA graph replay reuses the same pointers.
        for t in (self.keys, self.values, self.keys_scale, self.values_scale):
            torch._dynamo.mark_static_address(t)

        self.is_initialized = True

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: dict | None = None,
    ):
        if not self.is_initialized:
            self.lazy_initialization(key_states)

        cache_position = cache_kwargs.get("cache_position") if cache_kwargs is not None else None
        if cache_position is None:
            cache_position = torch.arange(key_states.shape[-2], device=self.device)

        k_fp = key_states.float()
        v_fp = value_states.float()
        k_scale = k_fp.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 127.0
        v_scale = v_fp.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 127.0
        k_q = (k_fp / k_scale).round().clamp(-128, 127).to(torch.int8)
        v_q = (v_fp / v_scale).round().clamp(-128, 127).to(torch.int8)

        self.keys.index_copy_(2, cache_position, k_q)
        self.values.index_copy_(2, cache_position, v_q)
        self.keys_scale.index_copy_(2, cache_position, k_scale)
        self.values_scale.index_copy_(2, cache_position, v_scale)

        # Full-buffer dequantize → bfloat16 (fixed shape, CUDA-graph safe).
        # Using a dynamic slice would break CUDA graph static shapes; unwritten
        # positions are zeros and are masked out by the causal attention mask.
        k_out = (self.keys.float() * self.keys_scale).to(torch.bfloat16)
        v_out = (self.values.float() * self.values_scale).to(torch.bfloat16)
        return k_out, v_out

    def reset(self):
        if self.is_initialized:
            self.keys.zero_()
            self.values.zero_()
            self.keys_scale.fill_(1.0)
            self.values_scale.fill_(1.0)

    def get_seq_length(self) -> int:
        return int((self.keys[0, 0].any(dim=-1)).sum()) if self.is_initialized else 0

    def get_max_cache_shape(self) -> int:
        return self.max_cache_len

    def get_mask_sizes(self, cache_position):
        return self.max_cache_len, 0


def replace_static_caches(model) -> int:
    """
    Replace StaticLayer objects in both CUDA-graph caches with QuantizedStaticLayer.
    Must be called after torch.compile, before warmup/CUDA graph capture.

    Returns the number of layers replaced.
    Raises RuntimeError if no compatible cache structure is found.
    """
    replaced = 0
    for cache_attr in ("predictor_graph", "talker_graph"):
        graph = getattr(model, cache_attr, None)
        if graph is None:
            continue
        cache = getattr(graph, "static_cache", None)
        if cache is None:
            continue

        if hasattr(cache, "layers"):
            new_layers = []
            for layer in cache.layers:
                max_len = getattr(layer, "max_cache_len", None)
                if max_len is None and hasattr(layer, "keys"):
                    # Already initialized — read from tensor shape
                    max_len = layer.keys.shape[2]
                if max_len is None:
                    raise RuntimeError(
                        f"Cannot determine max_cache_len from {type(layer).__name__} "
                        f"(attributes: {list(vars(layer).keys())})"
                    )
                new_layers.append(QuantizedStaticLayer(max_len))
            cache.layers = new_layers
            replaced += len(new_layers)
        else:
            raise RuntimeError(
                f"StaticCache at '{cache_attr}.static_cache' has no 'layers' attribute "
                f"(attributes: {list(vars(cache).keys())}). "
                "Cannot replace KV cache — check transformers version."
            )

    if replaced == 0:
        raise RuntimeError(
            "No patchable static cache found. "
            "Expected 'predictor_graph.static_cache' and/or 'talker_graph.static_cache'."
        )
    return replaced
