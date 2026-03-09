"""
Generic fallback config for HuggingFace transformer models without a dedicated config file.
This works for most standard causal language models (Llama-style, GPT-style, OPT-style, MoE).
All architecture details are dynamically derived from the HuggingFace AutoConfig object.
"""


# ============================================================================
# Elementwise Layer Specs for Dynamic Analyzer
# ============================================================================

def get_elementwise_layers(model_params):
    """
    Return elementwise (non-linear) layer specs for the analyzer.
    This allows the analyzer to dynamically compute OPs/memory for each layer type
    without hardcoding specific layer names.
    
    Returns a dict: {layer_name: {
        "type": "norm" | "add" | "activation" | "moe_router",
        "formula": {...}  # optional parameters for the analyzer
    }}
    """
    is_parallel = detect_parallel_attention(model_params)
    is_moe = detect_mlp_type(model_params) == 'moe'
    norm_layers = get_norm_layers(model_params)
    
    layers = {}
    
    # Norm layers - each norm layer in the config
    for norm_name in norm_layers:
        layers[norm_name] = {"type": "norm"}
    
    # Residual adds (attn_add, mlp_add)
    layers["attn_add"] = {"type": "add"}
    layers["mlp_add"] = {"type": "add"}
    
    # MLP activations
    layers["mlp_act"] = {"type": "activation"}
    
    # MoE router (if MoE)
    if is_moe:
        layers["moe_router"] = {"type": "moe_router"}
    
    return layers


# These formulas are used by the analyzer to compute OPs for each layer type
# They are parameterized by batchsize, seqlen, hidden_size
ELEMENTWISE_LAYER_FORMULAS = {
    "norm": {
        "OPs_per_element": 7,  # sum, sub, pow, sum, div, mul, add
    },
    "add": {
        "OPs_per_element": 1,  # element-wise add
    },
    "activation": {
        "OPs_per_element": 2,  # typically activation + optional upsample
    },
    "moe_router": {
        "OPs_per_element": 1,  # routing computation
    },
}

# ============================================================================
# Architecture Detection Functions
# ============================================================================

def detect_mlp_type(model_params):
    """
    Detect MLP type based on model architecture.
    Returns 'llama' (SwiGLU: gate_proj + up_proj), 'standard' (up_proj only), or 'moe' (Mixture of Experts).
    """
    # Check for MoE first
    if detect_moe(model_params):
        return 'moe'
    
    # Check hidden_act
    hidden_act = getattr(model_params, 'hidden_act', None)
    if hidden_act:
        if callable(hidden_act):
            hidden_act = getattr(hidden_act, '__name__', '')
        if isinstance(hidden_act, str):
            hidden_act_lower = hidden_act.lower()
            if hidden_act_lower in ('silu', 'silu_gelu', 'swiglu', 'fast_swiglu', 'silu_gelu_new'):
                return 'llama'
            return 'standard'
    
    # Check model_type
    model_type = getattr(model_params, 'model_type', '').lower()
    swiglu_models = {'llama', 'mistral', 'qwen', 'mixtral', 'cohere', 'baichuan', 'yi'}
    if model_type in swiglu_models:
        return 'llama'
    
    return 'standard'


def detect_parallel_attention(model_params):
    """Detect if model uses parallel attention/MLP (GPT-J style)."""
    model_type = getattr(model_params, 'model_type', '').lower()
    parallel_models = {'gptj', 'gpt_neox', 'gpt-neox'}
    
    if model_type in parallel_models:
        return True
    if getattr(model_params, 'use_parallel_residual', False):
        return True
    
    return False


def detect_moe(model_params):
    """Detect if model uses Mixture of Experts (MoE)."""
    # Check for explicit MoE attributes
    num_experts = getattr(model_params, 'num_experts', None)
    if num_experts is None:
        num_experts = getattr(model_params, 'num_local_experts', None)
    
    if num_experts is not None and num_experts > 1:
        return True
    
    # Check expert_capacity
    if getattr(model_params, 'expert_capacity', None) is not None:
        return True
    
    # Check model_type for known MoE architectures
    model_type = getattr(model_params, 'model_type', '').lower()
    moe_models = {'mixtral', 'moe', 'switch_transformer', 'qwen_moe', 'stablelm'}
    if model_type in moe_models:
        return True
    
    return False


def get_num_experts(model_params):
    """Get number of experts in MoE layer."""
    num_experts = getattr(model_params, 'num_experts', None)
    if num_experts is None:
        num_experts = getattr(model_params, 'num_local_experts', None)
    
    if num_experts is None or num_experts <= 1:
        return 1
    return num_experts


def get_num_experts_per_tok(model_params):
    """Get number of experts each token is routed to."""
    num_experts_per_tok = getattr(model_params, 'num_experts_per_tok', None)
    if num_experts_per_tok is None:
        num_experts_per_tok = getattr(model_params, 'top_k', None)
    
    if num_experts_per_tok is None:
        if detect_moe(model_params):
            return 2
        return 1
    return num_experts_per_tok


def detect_num_norms(model_params):
    """Detect how many norm layers per transformer block."""
    if detect_parallel_attention(model_params):
        return 1
    return 2


def detect_gqa(model_params):
    """Detect if model uses Grouped Query Attention."""
    num_attention_heads = get_num_attention_heads(model_params)
    num_key_value_heads = get_num_key_value_heads(model_params)
    
    if num_attention_heads and num_key_value_heads:
        return num_key_value_heads < num_attention_heads
    return False


# ============================================================================
# Attribute Extraction Functions
# ============================================================================

def get_num_attention_heads(model_params):
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
    num_kv = getattr(model_params, "num_key_value_heads", None)
    if num_kv is not None:
        return num_kv
    
    num_kv = getattr(model_params, "multi_query_group_num", None)
    if num_kv is not None:
        return num_kv
    
    if getattr(model_params, "multi_query_attention", False):
        return 1
    
    return get_num_attention_heads(model_params)


def get_num_hidden_layers(model_params):
    return getattr(model_params, "num_hidden_layers", None) \
        or getattr(model_params, "n_layer", None) \
        or getattr(model_params, "num_layers", None) \
        or getattr(model_params, "n_layers", None) \
        or getattr(model_params, "config", {}).get("num_hidden_layers", 12)


def get_intermediate_size(model_params):
    intermediate = getattr(model_params, "intermediate_size", None)
    if intermediate is not None:
        return intermediate
    
    intermediate = getattr(model_params, "ffn_hidden_size", None)
    if intermediate is not None:
        return intermediate
    
    intermediate = getattr(model_params, "n_inner", None)
    if intermediate is not None:
        return intermediate
    
    intermediate = getattr(model_params, "ffn_dim", None)
    if intermediate is not None:
        return intermediate
    
    # MoE expert size
    intermediate = getattr(model_params, "expert_intermediate_size", None)
    if intermediate is not None:
        return intermediate
    
    hidden_size = get_hidden_size(model_params)
    return hidden_size * 4 if hidden_size else 11008


def get_vocab_size(model_params):
    return getattr(model_params, "vocab_size", None) \
        or getattr(model_params, "padded_vocab_size", None) \
        or getattr(model_params, "config", {}).get("vocab_size", 32000)


def get_norm_layers(model_params):
    if detect_parallel_attention(model_params):
        return ["attn_norm"]
    return ["attn_norm", "mlp_norm"]


# ============================================================================
# Linear Layers
# ============================================================================

def get_linear_layers(model_params, tp_size: int):
    """Get linear layer dimensions based on model architecture."""
    hidden_size = get_hidden_size(model_params)
    intermediate_size = get_intermediate_size(model_params)
    key_value_heads = get_num_key_value_heads(model_params)
    attention_heads = get_num_attention_heads(model_params)
    
    mlp_type = detect_mlp_type(model_params)

    if tp_size > 1:
        assert hidden_size % tp_size == 0
        assert intermediate_size % tp_size == 0
        if key_value_heads:
            assert key_value_heads % tp_size == 0

    # Base attention layers (always present)
    layers = {
        "q_proj": [hidden_size, hidden_size // tp_size],
        "k_proj": [hidden_size, hidden_size * key_value_heads // attention_heads // tp_size] if attention_heads and key_value_heads else [hidden_size, hidden_size // tp_size],
        "v_proj": [hidden_size, hidden_size * key_value_heads // attention_heads // tp_size] if attention_heads and key_value_heads else [hidden_size, hidden_size // tp_size],
        "out_proj": [hidden_size // tp_size, hidden_size],
    }
    
    # Add MLP layers based on architecture type
    if mlp_type == 'moe':
        # MoE: gate_proj + up_proj for router
        layers["gate_proj"] = [hidden_size, intermediate_size // tp_size]
        layers["up_proj"] = [hidden_size, intermediate_size // tp_size]
    elif mlp_type == 'llama':
        # SwiGLU
        layers["gate_proj"] = [hidden_size, intermediate_size // tp_size]
        layers["up_proj"] = [hidden_size, intermediate_size // tp_size]
    else:
        # Standard MLP
        layers["up_proj"] = [hidden_size, intermediate_size // tp_size]
    
    layers["down_proj"] = [intermediate_size // tp_size, hidden_size]
    
    return layers


def get_moe_info(model_params):
    """Get MoE metadata if applicable."""
    if detect_moe(model_params):
        return {
            "num_experts": get_num_experts(model_params),
            "num_experts_per_tok": get_num_experts_per_tok(model_params)
        }
    return None


# ============================================================================
# Post-process
# ============================================================================

def post_process(model_params, args):
    hidden_size = get_hidden_size(model_params)
    vocab_size = get_vocab_size(model_params)
    layers = []
    for stage in ["prefill", "decode"]:
        layers.append({
            'name': 'lm_head',
            'stage': stage,
            'OPs': args['batchsize'] * hidden_size * vocab_size * 1,
            'load_weight': hidden_size * vocab_size * args['w_byte'],
            'load_act': hidden_size * args['a_byte'],
            'store_act': vocab_size * args['a_byte'],
        })
    return layers


# ============================================================================
# Unified Layer Graph Builder
# ============================================================================

def build_layer_graph(model_params, use_flashattention=False):
    """Build transformer layer graph based on model architecture."""
    mlp_type = detect_mlp_type(model_params)
    is_moe = mlp_type == 'moe'
    is_parallel = detect_parallel_attention(model_params)
    norm_layers = get_norm_layers(model_params)
    linear_layers = get_linear_layers(model_params, tp_size=1)
    
    graph = {"input": []}
    
    # First norm layer
    if len(norm_layers) >= 1:
        graph[norm_layers[0]] = ["input"]
        prev_norm = norm_layers[0]
    else:
        prev_norm = "input"
    
    # Q, K, V projections
    attention_proj_inputs = []
    if "q_proj" in linear_layers:
        graph["q_proj"] = [prev_norm]
        attention_proj_inputs.append("q_proj")
    if "k_proj" in linear_layers:
        graph["k_proj"] = [prev_norm]
        attention_proj_inputs.append("k_proj")
    if "v_proj" in linear_layers:
        graph["v_proj"] = [prev_norm]
        attention_proj_inputs.append("v_proj")
    
    # Attention computation
    if use_flashattention and attention_proj_inputs:
        graph["fused_attention"] = attention_proj_inputs
        attn_out = "fused_attention"
    elif "q_proj" in linear_layers and "k_proj" in linear_layers:
        graph["qk_matmul"] = ["q_proj", "k_proj"]
        graph["softmax"] = ["qk_matmul"]
        if "v_proj" in linear_layers:
            graph["sv_matmul"] = ["softmax", "v_proj"]
        else:
            graph["sv_matmul"] = ["softmax"]
        attn_out = "sv_matmul"
    else:
        attn_out = attention_proj_inputs[0] if attention_proj_inputs else "input"
    
    # Output projection
    if "out_proj" in linear_layers:
        graph["out_proj"] = [attn_out]
        attn_out = "out_proj"
    
    # Attention residual add
    graph["attn_add"] = ["input", attn_out]
    
    # Determine MLP input and build MLP path
    if is_moe:
        # MoE: gate_proj + up_proj -> router -> experts
        mlp_input = "attn_add"
        if "gate_proj" in linear_layers:
            graph["gate_proj"] = [mlp_input]
        if "up_proj" in linear_layers:
            graph["up_proj"] = [mlp_input]
        
        # Router (simplified visualization)
        if "gate_proj" in linear_layers and "up_proj" in linear_layers:
            graph["moe_router"] = ["gate_proj", "up_proj"]
        
        # Expert computation (simplified)
        if "up_proj" in linear_layers:
            graph["mlp_act"] = ["up_proj"]
        if "down_proj" in linear_layers:
            graph["down_proj"] = ["mlp_act"]
        
        mlp_out = "down_proj" if "down_proj" in linear_layers else "mlp_act"
        
    elif is_parallel:
        # Parallel: MLP takes input directly from input
        mlp_input = "input"
        if len(norm_layers) >= 2:
            graph[norm_layers[1]] = ["attn_add"]
        
        if mlp_type == 'llama':
            if "gate_proj" in linear_layers:
                graph["gate_proj"] = [mlp_input]
            if "up_proj" in linear_layers:
                graph["up_proj"] = [mlp_input]
            if "gate_proj" in linear_layers and "up_proj" in linear_layers:
                graph["mlp_act"] = ["gate_proj", "up_proj"]
        else:
            if "up_proj" in linear_layers:
                graph["up_proj"] = [mlp_input]
                graph["mlp_act"] = ["up_proj"]
        
        # Add down_proj to graph
        if "down_proj" in linear_layers:
            graph["down_proj"] = ["mlp_act"]
        
        mlp_out = "down_proj" if "down_proj" in linear_layers else "mlp_act"
        
    else:
        # Serial: MLP gets input from attn_add
        mlp_input = "attn_add"
        if len(norm_layers) >= 2:
            graph[norm_layers[1]] = ["attn_add"]
            mlp_input = norm_layers[1]
        
        if mlp_type == 'llama':
            if "gate_proj" in linear_layers:
                graph["gate_proj"] = [mlp_input]
            if "up_proj" in linear_layers:
                graph["up_proj"] = [mlp_input]
            if "gate_proj" in linear_layers and "up_proj" in linear_layers:
                graph["mlp_act"] = ["gate_proj", "up_proj"]
        else:
            if "up_proj" in linear_layers:
                graph["up_proj"] = [mlp_input]
                graph["mlp_act"] = ["up_proj"]
        
        # Add down_proj to graph
        if "down_proj" in linear_layers:
            graph["down_proj"] = ["mlp_act"]
        
        mlp_out = "down_proj" if "down_proj" in linear_layers else "mlp_act"
    
    # MLP residual add and output
    graph["mlp_add"] = ["attn_add", mlp_out]
    graph["output"] = ["mlp_add"]
    
    return graph


def get_transformer_layer_graph(model_params):
    return build_layer_graph(model_params, use_flashattention=False)


def get_flashattention_layer_graph(model_params):
    return build_layer_graph(model_params, use_flashattention=True)


# Backward compatibility
transformer_layer_graph = {}
flashattention_transformer_layer_graph = {}
