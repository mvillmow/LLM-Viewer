"""
Generic fallback config for HuggingFace transformer models without a dedicated config file.
This works for most standard causal language models (Llama-style, GPT-style, OPT-style).
"""


def get_num_attention_heads(model_params):
    # Try common attribute names
    return getattr(model_params, "num_attention_heads", None) \
        or getattr(model_params, "n_head", None) \
        or getattr(model_params, "num_heads", None) \
        or getattr(model_params, "n_heads", None) \
        or getattr(model_params, "config", {}).get("num_attention_heads", 12)


def get_hidden_size(model_params):
    return getattr(model_params, "hidden_size", None) \
        or getattr(model_params, "n_embd", None) \
        or getattr(model_params, "d_model", None) \
        or getattr(model_params, "config", {}).get("hidden_size", 768)


def get_num_key_value_heads(model_params):
    # Try common attribute names for GQA support
    # Default to num_attention_heads if no explicit GQA attributes found
    num_kv = getattr(model_params, "num_key_value_heads", None)
    if num_kv is not None:
        return num_kv
    # For non-GQA models, num_key_value_heads equals num_attention_heads
    return get_num_attention_heads(model_params)


def get_norm_layers(model_params):
    return ["attn_norm", "mlp_norm"]


def get_num_hidden_layers(model_params):
    return getattr(model_params, "num_hidden_layers", None) \
        or getattr(model_params, "n_layer", None) \
        or getattr(model_params, "num_layers", None) \
        or getattr(model_params, "n_layers", None) \
        or getattr(model_params, "config", {}).get("num_hidden_layers", 12)


def get_intermediate_size(model_params):
    # Try common MLP intermediate sizes
    intermediate = getattr(model_params, "intermediate_size", None)
    if intermediate is not None:
        return intermediate
    
    intermediate = getattr(model_params, "mlp_hidden_size", None)
    if intermediate is not None:
        return intermediate
        
    intermediate = getattr(model_params, "ffn_dim", None)
    if intermediate is not None:
        return intermediate
    
    # Default to 4x hidden_size (common for SwiGLU/standard MLPs)
    hidden_size = get_hidden_size(model_params)
    return hidden_size * 4 if hidden_size else 11008


def get_vocab_size(model_params):
    return getattr(model_params, "vocab_size", None) \
        or getattr(model_params, "config", {}).get("vocab_size", 32000)


def post_process(model_params, args):
    hiddensize = get_hidden_size(model_params)
    vocab_size = get_vocab_size(model_params)
    layers = []
    for stage in ["prefill", "decode"]:
        layers.append({
            'name': 'lm_head',
            'stage': stage,
            'OPs': args['batchsize'] * hiddensize * vocab_size * 1,
            'load_weight': hiddensize * vocab_size * args['w_byte'],
            'load_act': hiddensize * args['a_byte'],
            'store_act': vocab_size * args['a_byte'],
        })
    return layers


def get_linear_layers(model_params, tp_size: int):
    hidden_size = get_hidden_size(model_params)
    intermediate_size = get_intermediate_size(model_params)
    key_value_heads = get_num_key_value_heads(model_params)
    attention_heads = get_num_attention_heads(model_params)

    if tp_size > 1:
        assert hidden_size % tp_size == 0, f"hidden_size {hidden_size} not divisible by tp_size {tp_size}"
        assert intermediate_size % tp_size == 0, f"intermediate_size {intermediate_size} not divisible by tp_size {tp_size}"
        if key_value_heads:
            assert key_value_heads % tp_size == 0, f"key_value_heads {key_value_heads} not divisible by tp_size {tp_size}"

    # Standard transformer linear layers
    return {
        "q_proj": [hidden_size, hidden_size // tp_size],
        "k_proj": [hidden_size, hidden_size * key_value_heads // attention_heads // tp_size] if attention_heads and key_value_heads else [hidden_size, hidden_size // tp_size],
        "v_proj": [hidden_size, hidden_size * key_value_heads // attention_heads // tp_size] if attention_heads and key_value_heads else [hidden_size, hidden_size // tp_size],
        "out_proj": [hidden_size // tp_size, hidden_size],
        "gate_proj": [hidden_size, intermediate_size // tp_size],
        "up_proj": [hidden_size, intermediate_size // tp_size],
        "down_proj": [intermediate_size // tp_size, hidden_size],
    }


# Standard transformer layer graph
transformer_layer_graph = {
    "input": [],
    "attn_norm": ["input"],
    "q_proj": ["attn_norm"],
    "k_proj": ["attn_norm"],
    "v_proj": ["attn_norm"],
    "qk_matmul": ["q_proj", "k_proj"],
    "softmax": ["qk_matmul"],
    "sv_matmul": ["softmax", "v_proj"],
    "out_proj": ["sv_matmul"],
    "attn_add": ["input", "out_proj"],
    "mlp_norm": ["attn_add"],
    "gate_proj": ["mlp_norm"],
    "up_proj": ["mlp_norm"],
    "mlp_act": ["up_proj", "gate_proj"],
    "down_proj": ["mlp_act"],
    "mlp_add": ["attn_add", "down_proj"],
    "output": ["mlp_add"]
}


# Flash attention transformer layer graph
flashattention_transformer_layer_graph = {
    "input": [],
    "attn_norm": ["input"],
    "q_proj": ["attn_norm"],
    "k_proj": ["attn_norm"],
    "v_proj": ["attn_norm"],
    "fused_attention": ["q_proj", "k_proj", "v_proj"],
    "out_proj": ["fused_attention"],
    "attn_add": ["input", "out_proj"],
    "mlp_norm": ["attn_add"],
    "gate_proj": ["mlp_norm"],
    "up_proj": ["mlp_norm"],
    "mlp_act": ["up_proj", "gate_proj"],
    "down_proj": ["mlp_act"],
    "mlp_add": ["attn_add", "down_proj"],
    "output": ["mlp_add"]
}