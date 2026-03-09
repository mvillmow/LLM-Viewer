from hardwares.hardware_params import hardware_params

# Suggested models for autocomplete - these are pre-configured models that work well with the system
# Users can still specify ANY HuggingFace model ID, these are just suggestions
suggested_model_ids_sources = {
    "LLM360/K2-Think-V2": { "source": "huggingface" },
    "meta-llama/Llama-2-7b-hf": {"source": "huggingface"},
    "meta-llama/Llama-2-13b-hf": {"source": "huggingface"},
    "meta-llama/Llama-2-70b-hf": {"source": "huggingface"},
    "EleutherAI/gpt-j-6B":{"source": "huggingface"},
    "THUDM/chatglm3-6b": {"source": "huggingface"},
    "facebook/opt-125m": {"source": "huggingface"},
    "facebook/opt-1.3b": {"source": "huggingface"},
    "facebook/opt-2.7b": {"source": "huggingface"},
    "facebook/opt-6.7b": {"source": "huggingface"},
    "facebook/opt-30b": {"source": "huggingface"},
    "facebook/opt-66b": {"source": "huggingface"},
    # "DiT-XL/2": {"source": "DiT"},
    # "DiT-XL/4": {"source": "DiT"},
}
suggested_model_ids = [_ for _ in suggested_model_ids_sources.keys()]
avaliable_hardwares = [_ for _ in hardware_params.keys()]


def get_model_source(model_id):
    """Get the source for a model ID. Defaults to huggingface for unknown models."""
    if model_id in suggested_model_ids_sources:
        return suggested_model_ids_sources[model_id]["source"]
    return "huggingface"  # Default source for any model ID
