from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
import importlib
import os
from hardwares.hardware_params import hardware_params
from model_analyzer import ModelAnalyzer
from utils import str_number
import numpy as np
import re
from collections import deque

config_cache = {}


def get_analyer(model_id, hardware, config_path) -> ModelAnalyzer:
    config = f"{model_id}_{hardware}_{config_path}"
    if config not in config_cache:
        config_cache[config] = ModelAnalyzer(
            model_id,
            hardware,
            config_path,
        )
    return config_cache[config]


# def get_model_config(model_id,config_path):
#     if model_id not in config_cache:
#         model_config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
#         config = importlib.import_module(config_path.replace("/", ".").replace(".py", ""))
#         config_cache[model_id] = model_config,config
#     return config_cache[model_id]


def get_quant_bit(dtype):
    if dtype == "FP16":
        return 16
    elif dtype == "INT8":
        return 8
    elif dtype == "INT4":
        return 4
    elif "bit" in dtype:
        bitwidth = int(re.findall(r"\d+", dtype)[0])
        return bitwidth
    else:
        raise ValueError(f"Unsupported dtype:{dtype}")


def get_model_graph(model_id, hardware, config_path, inference_config):

    # Roofline model
    w_bit = get_quant_bit(inference_config["w_quant"])
    a_bit = get_quant_bit(inference_config["a_quant"])
    kv_bit = get_quant_bit(inference_config["kv_quant"])
    seq_length = int(inference_config["seq_length"])
    batch_size = int(inference_config["batch_size"])
    use_flashattention = bool(inference_config["use_flashattention"])
    gen_length = int(inference_config["gen_length"])
    tp_size = int(inference_config["tp_size"])

    analyzer = get_analyer(model_id, hardware, config_path)
    result = analyzer.analyze(
        seqlen=seq_length,
        batchsize=batch_size,
        w_bit=w_bit,
        a_bit=a_bit,
        kv_bit=kv_bit,
        use_flashattention=use_flashattention,
        tp_size=tp_size
    )
    bandwidth, max_OPS, onchip_buffer = analyzer.get_hardware_info()
    GQA = analyzer.get_model_info()["GQA"]
    hardware_info = {
        "bandwidth": bandwidth,
        "max_OPS": max_OPS,
        "onchip_buffer": onchip_buffer,
    }

    nodes = []
    edges = []

    def write_to_node(name, OPs, memory_access, info, input_names=[]):
        node = {
            "label": name,
            "id": name,
            "description": f"OPs:{str_number(OPs)}, Access:{str_number(memory_access)}",
            "info": info,
        }
        if GQA and name in ["qk_matmul", "sv_matmul"]:
            node["label"] += "(GQA)"
        nodes.append(node)
        for input_name in input_names:
            edge = {"source": input_name, "target": name}
            edges.append(edge)

    if use_flashattention:
        # Use dynamic graph generation based on model architecture
        if hasattr(analyzer.config, 'get_flashattention_layer_graph'):
            layer_graph = analyzer.config.get_flashattention_layer_graph(analyzer.model_params)
        else:
            # Fallback for configs without dynamic graph support
            layer_graph = analyzer.config.flashattention_transformer_layer_graph
    else:
        # Use dynamic graph generation based on model architecture
        if hasattr(analyzer.config, 'get_transformer_layer_graph'):
            layer_graph = analyzer.config.get_transformer_layer_graph(analyzer.model_params)
        else:
            # Fallback for configs without dynamic graph support
            layer_graph = analyzer.config.transformer_layer_graph
    stage = inference_config["stage"]
    total_results = result["total_results"]
    if stage != "chat":
        result = result[stage]
    else:
        result = result["prefill"]

    for name, input_names in layer_graph.items():
        if name in ["input", "output"]:
            OPs = 0
            memory_access = 0
            info = {}
        else:
            # Defensive handling: if node not in results, use zero values instead of crashing
            if name not in result:
                print(f"Warning: node '{name}' from layer_graph not found in analyzer results. This may indicate an architecture mismatch.")
                OPs = 0
                memory_access = 0
                info = {}
            else:
                OPs = result[name]["OPs"]
                memory_access = result[name]["memory_access"]
                info = result[name]
        write_to_node(name, OPs, memory_access, info, input_names)
    if stage == "chat":
        # seq_length:seq_length+gen_length
        total_results["chat"] = total_results["prefill"]
        n_divide = min(10, gen_length)
        for lengthi in np.linspace(seq_length + 1, seq_length + gen_length, n_divide):
            gen_result = analyzer.analyze(
                seqlen=lengthi,
                batchsize=batch_size,
                w_bit=w_bit,
                a_bit=a_bit,
                kv_bit=kv_bit,
                use_flashattention=use_flashattention,
            )
            for k, v in gen_result["total_results"]["decode"].items():
                total_results["chat"][k] += v * gen_length / n_divide
            for name, input_names in layer_graph.items():
                if name in gen_result["decode"]:
                    result[name]["OPs"] += (
                        gen_result["decode"][name]["OPs"] * gen_length / n_divide
                    )
                    result[name]["memory_access"] += (
                        gen_result["decode"][name]["memory_access"]
                        * gen_length
                        / n_divide
                    )
        for name, input_names in layer_graph.items():
            if name in ["input", "output"]:
                OPs = 0
                memory_access = 0
                info = {}
            else:
                # Defensive handling: if node not in results, use zero values
                if name not in result:
                    OPs = 0
                    memory_access = 0
                    info = {}
                else:
                    OPs = result[name]["OPs"]
                    memory_access = result[name]["memory_access"]
                    info = {}
            write_to_node(name, OPs, memory_access, info, input_names)
    # Compute graph metadata
    graph_info = compute_graph_metadata(layer_graph, analyzer)
    
    return nodes, edges, total_results, hardware_info, graph_info


def compute_graph_metadata(layer_graph, analyzer):
    """
    Compute graph metadata including node counts, loops, critical path, and layer count.
    """
    # Get num_hidden_layers from config
    num_hidden_layers = analyzer.config.get_num_hidden_layers(analyzer.model_params)
    
    # Node counts by type
    node_counts = {
        "total": len([n for n in layer_graph.keys() if n not in ["input", "output"]]),
        "input": 1 if "input" in layer_graph else 0,
        "output": 1 if "output" in layer_graph else 0,
    }
    
    # Categorize nodes by type
    linear_nodes = ["q_proj", "k_proj", "v_proj", "out_proj", "gate_proj", "up_proj", "down_proj", "lm_head"]
    attention_nodes = ["qk_matmul", "sv_matmul", "softmax", "fused_attention"]
    norm_nodes = ["attn_norm", "mlp_norm"]
    add_nodes = ["attn_add", "mlp_add"]
    activation_nodes = ["mlp_act"]
    
    node_counts["linear"] = sum(1 for n in layer_graph.keys() if n in linear_nodes)
    node_counts["attention"] = sum(1 for n in layer_graph.keys() if n in attention_nodes)
    node_counts["norm"] = sum(1 for n in layer_graph.keys() if n in norm_nodes)
    node_counts["add"] = sum(1 for n in layer_graph.keys() if n in add_nodes)
    node_counts["activation"] = sum(1 for n in layer_graph.keys() if n in activation_nodes)
    
    # Edge count
    edge_count = sum(len(inputs) for inputs in layer_graph.values())
    node_counts["edges"] = edge_count
    
    # Cycle detection using DFS
    has_cycles, cycles = detect_cycles(layer_graph)
    
    # Critical path / depth (longest path from input to output)
    critical_path_length = compute_critical_path(layer_graph)
    
    # Layer repetition count
    layer_repetition = num_hidden_layers
    
    return {
        "node_counts": node_counts,
        "has_cycles": has_cycles,
        "cycles": cycles,
        "critical_path_depth": critical_path_length,
        "layer_repetition_count": layer_repetition,
    }


def detect_cycles(graph):
    """Detect cycles in the graph using DFS. Returns (has_cycles, list_of_cycles)."""
    # Find all simple cycles using DFS with path tracking
    nodes = list(graph.keys())
    if "input" not in nodes or "output" not in nodes:
        return False, []
    
    cycles = []
    visited = set()
    rec_stack = set()
    
    def dfs(node, path, visited_local):
        visited_local.add(node)
        rec_stack.add(node)
        
        for neighbor in graph.get(node, []):
            if neighbor not in visited_local:
                result = dfs(neighbor, path + [neighbor], visited_local.copy())
                if result:
                    return True
            elif neighbor in rec_stack and neighbor in path:
                # Found a cycle
                cycle_start = path.index(neighbor)
                cycle = path[cycle_start:] + [neighbor]
                cycles.append(cycle)
                return True
        
        rec_stack.remove(node)
        return False
    
    # Start from input node
    if "input" in nodes:
        visited_local = set()
        dfs("input", ["input"], visited_local)
    
    return len(cycles) > 0, cycles[:3]  # Return first 3 cycles max


def compute_critical_path(graph):
    """
    Compute the critical path (longest path) from input to output in the DAG.
    Uses topological sort with DP.
    """
    if "input" not in graph or "output" not in graph:
        return 0
    
    nodes = list(graph.keys())
    in_degree = {n: 0 for n in nodes}
    
    for node, inputs in graph.items():
        for inp in inputs:
            if inp in in_degree:
                in_degree[node] += 1
    
    # Topological sort
    queue = deque([n for n in nodes if in_degree[n] == 0])
    
    # dist[node] = longest path length to reach this node
    dist = {n: 0 for n in nodes}
    
    while queue:
        node = queue.popleft()
        for neighbor in graph.get(node, []):
            if neighbor in dist:
                dist[neighbor] = max(dist[neighbor], dist[node] + 1)
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
    
    return dist.get("output", 0)
