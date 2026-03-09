from hardwares.hardware_params import hardware_params

# Suggested models for autocomplete - these are pre-configured models that work well with the system
# Users can still specify ANY HuggingFace model ID, these are just suggestions
suggested_model_ids = [
    "LLM360/K2-Think-V2",
    "meta-llama/Llama-2-7b-hf",
    "meta-llama/Llama-2-13b-hf",
    "meta-llama/Llama-2-70b-hf",
    "EleutherAI/gpt-j-6B",
    "THUDM/chatglm3-6b",
    "facebook/opt-125m",
    "facebook/opt-1.3b",
    "facebook/opt-2.7b",
    "facebook/opt-6.7b",
    "facebook/opt-30b",
    "facebook/opt-66b",
]

avaliable_hardwares = [_ for _ in hardware_params.keys()]
