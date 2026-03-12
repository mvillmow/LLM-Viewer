import copy
import re
from collections import defaultdict, deque

import numpy as np

from model_analyzer import ModelAnalyzer
from model_introspector import get_model_topology
from utils import str_number


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


def get_quant_bit(dtype):
    if dtype == "FP16":
        return 16
    if dtype == "INT8":
        return 8
    if dtype == "INT4":
        return 4
    if "bit" in dtype:
        bitwidth = int(re.findall(r"\d+", dtype)[0])
        return bitwidth
    raise ValueError(f"Unsupported dtype:{dtype}")


def get_model_graph(model_id, hardware, config_path, inference_config):
    w_bit = get_quant_bit(inference_config["w_quant"])
    a_bit = get_quant_bit(inference_config["a_quant"])
    kv_bit = get_quant_bit(inference_config["kv_quant"])
    seq_length = int(inference_config["seq_length"])
    batch_size = int(inference_config["batch_size"])
    use_flashattention = bool(inference_config["use_flashattention"])
    gen_length = int(inference_config["gen_length"])
    tp_size = int(inference_config["tp_size"])
    stage = inference_config["stage"]

    analyzer = get_analyer(model_id, hardware, config_path)
    results = analyzer.analyze(
        seqlen=seq_length,
        batchsize=batch_size,
        w_bit=w_bit,
        a_bit=a_bit,
        kv_bit=kv_bit,
        use_flashattention=use_flashattention,
        tp_size=tp_size,
    )

    bandwidth, max_OPS, onchip_buffer = analyzer.get_hardware_info()
    hardware_info = {
        "bandwidth": bandwidth,
        "max_OPS": max_OPS,
        "onchip_buffer": onchip_buffer,
    }

    total_results, stage_results = _select_stage_results(
        analyzer=analyzer,
        base_results=results,
        stage=stage,
        seq_length=seq_length,
        gen_length=gen_length,
        batch_size=batch_size,
        w_bit=w_bit,
        a_bit=a_bit,
        kv_bit=kv_bit,
        use_flashattention=use_flashattention,
        tp_size=tp_size,
    )

    topology = get_model_topology(
        model_id=model_id,
        model_params=analyzer.model_params,
        analyzer_config=analyzer.config,
        use_flashattention=use_flashattention,
    )
    nodes = _overlay_metrics_on_nodes(
        topology["nodes"],
        stage_results,
        analyzer.get_model_info()["GQA"],
    )
    edges = topology["edges"]
    combos = topology["combos"]
    graph_info = compute_graph_metadata(nodes, edges, analyzer, topology["graph_info"])

    return nodes, edges, combos, total_results, hardware_info, graph_info


def _select_stage_results(
    analyzer,
    base_results,
    stage,
    seq_length,
    gen_length,
    batch_size,
    w_bit,
    a_bit,
    kv_bit,
    use_flashattention,
    tp_size,
):
    total_results = copy.deepcopy(base_results["total_results"])
    if stage != "chat":
        return total_results, copy.deepcopy(base_results[stage])

    chat_results = copy.deepcopy(base_results["prefill"])
    total_results["chat"] = copy.deepcopy(base_results["total_results"]["prefill"])
    n_divide = min(10, gen_length)
    if n_divide <= 0:
        return total_results, chat_results

    for lengthi in np.linspace(seq_length + 1, seq_length + gen_length, n_divide):
        gen_result = analyzer.analyze(
            seqlen=int(lengthi),
            batchsize=batch_size,
            w_bit=w_bit,
            a_bit=a_bit,
            kv_bit=kv_bit,
            use_flashattention=use_flashattention,
            tp_size=tp_size,
        )
        for key, value in gen_result["total_results"]["decode"].items():
            total_results["chat"][key] += value * gen_length / n_divide
        for name, info in gen_result["decode"].items():
            if name not in chat_results:
                chat_results[name] = copy.deepcopy(info)
                continue
            for key, value in info.items():
                if isinstance(value, (int, float, np.number)):
                    chat_results[name][key] = chat_results[name].get(key, 0) + value * gen_length / n_divide

    return total_results, chat_results


def _overlay_metrics_on_nodes(topology_nodes, stage_results, gqa_enabled):
    nodes = []
    for topology_node in topology_nodes:
        node = copy.deepcopy(topology_node)
        node_id = node["id"]
        topology_info = copy.deepcopy(node.get("info", {}))
        metrics = copy.deepcopy(stage_results.get(node_id, {}))
        info = {}
        info.update(topology_info)
        info.update(metrics)
        node["info"] = info

        if metrics:
            node["description"] = (
                f"OPs:{str_number(metrics.get('OPs', 0))}, "
                f"Access:{str_number(metrics.get('memory_access', 0))}"
            )
        else:
            node["description"] = topology_info.get("module_type", topology_info.get("node_type", ""))

        if gqa_enabled and node_id in {"qk_matmul", "sv_matmul"}:
            node["label"] = f"{node['label']}(GQA)"
        nodes.append(node)
    return nodes


def compute_graph_metadata(nodes, edges, analyzer, topology_graph_info):
    node_ids = {node["id"] for node in nodes}
    node_counts = {
        "total": len([node_id for node_id in node_ids if node_id not in {"input", "output"}]),
        "input": 1 if "input" in node_ids else 0,
        "output": 1 if "output" in node_ids else 0,
        "edges": len(edges),
    }

    linear_nodes = {"q_proj", "k_proj", "v_proj", "out_proj", "gate_proj", "up_proj", "down_proj", "lm_head"}
    attention_nodes = {"qk_matmul", "sv_matmul", "softmax", "fused_attention"}
    norm_nodes = {"attn_norm", "mlp_norm", "final_norm"}
    add_nodes = {"attn_add", "mlp_add"}
    activation_nodes = {"mlp_act"}

    node_counts["linear"] = len(node_ids & linear_nodes)
    node_counts["attention"] = len(node_ids & attention_nodes)
    node_counts["norm"] = len(node_ids & norm_nodes)
    node_counts["add"] = len(node_ids & add_nodes)
    node_counts["activation"] = len(node_ids & activation_nodes)
    node_counts["residual_edges"] = sum(1 for edge in edges if edge.get("edgeType") == "residual")

    has_cycles, cycles = detect_cycles(edges)
    critical_path_length = compute_critical_path(edges)

    graph_info = copy.deepcopy(topology_graph_info)
    graph_info.update(
        {
            "node_counts": node_counts,
            "has_cycles": has_cycles,
            "cycles": cycles,
            "critical_path_depth": critical_path_length,
            "layer_repetition_count": topology_graph_info.get(
                "block_repetition_count",
                analyzer.config.get_num_hidden_layers(analyzer.model_params),
            ),
        }
    )
    return graph_info


def detect_cycles(edges):
    adjacency = defaultdict(list)
    nodes = set()
    for edge in edges:
        source = edge["source"]
        target = edge["target"]
        adjacency[source].append(target)
        nodes.add(source)
        nodes.add(target)

    cycles = []
    visiting = set()
    visited = set()

    def dfs(node, path):
        if node in visiting:
            if node in path:
                cycle_start = path.index(node)
                cycles.append(path[cycle_start:] + [node])
            return
        if node in visited:
            return
        visiting.add(node)
        for neighbor in adjacency.get(node, []):
            dfs(neighbor, path + [neighbor])
        visiting.remove(node)
        visited.add(node)

    for node in list(nodes):
        if node not in visited:
            dfs(node, [node])
        if cycles:
            break

    return len(cycles) > 0, cycles[:3]


def compute_critical_path(edges):
    adjacency = defaultdict(list)
    in_degree = defaultdict(int)
    nodes = set()
    for edge in edges:
        source = edge["source"]
        target = edge["target"]
        adjacency[source].append(target)
        in_degree[target] += 1
        nodes.add(source)
        nodes.add(target)

    for node in nodes:
        in_degree.setdefault(node, 0)

    queue = deque([node for node in nodes if in_degree[node] == 0])
    distance = {node: 0 for node in nodes}
    while queue:
        node = queue.popleft()
        for neighbor in adjacency.get(node, []):
            distance[neighbor] = max(distance[neighbor], distance[node] + 1)
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    return distance.get("output", 0)
