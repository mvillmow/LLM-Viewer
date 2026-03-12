import copy
import operator
from collections import defaultdict

import torch
import torch.nn as nn
from torch.fx import symbolic_trace
from transformers import AutoConfig, AutoModelForCausalLM


_TOPOLOGY_CACHE = {}


CANONICAL_BLOCK_NODE_ORDER = [
    "input",
    "attn_norm",
    "q_proj",
    "k_proj",
    "v_proj",
    "qk_matmul",
    "softmax",
    "sv_matmul",
    "fused_attention",
    "out_proj",
    "attn_add",
    "mlp_norm",
    "gate_proj",
    "up_proj",
    "moe_router",
    "mlp_act",
    "down_proj",
    "mlp_add",
    "output",
]

TOP_LEVEL_NODE_IDS = {"embedding", "final_norm", "lm_head"}

CANONICAL_NAME_HINTS = {
    "attn_norm": {"input_layernorm", "attention_norm", "attn_norm", "ln_1", "ln1", "norm1"},
    "mlp_norm": {"post_attention_layernorm", "ffn_norm", "mlp_norm", "ln_2", "ln2", "norm2"},
    "q_proj": {"q_proj", "query", "q_linear", "c_attn_q"},
    "k_proj": {"k_proj", "key", "k_linear", "c_attn_k"},
    "v_proj": {"v_proj", "value", "v_linear", "c_attn_v"},
    "out_proj": {"o_proj", "out_proj", "dense", "c_proj", "wo"},
    "gate_proj": {"gate_proj", "w1"},
    "up_proj": {"up_proj", "fc1", "dense_h_to_4h", "w3"},
    "down_proj": {"down_proj", "fc2", "dense_4h_to_h", "w2"},
    "moe_router": {"router", "gate", "router_logits", "switch"},
}

ATTENTION_PATH_HINTS = ("self_attn", "attention", "attn", "mixer")
MLP_PATH_HINTS = ("mlp", "ffn", "feed_forward", "feedforward", "block_sparse_moe", "moe")
MOE_PATH_HINTS = ("moe", "expert", "experts", "router")
EMBEDDING_HINTS = ("embed_tokens", "wte", "tok_embeddings", "word_embeddings", "embeddings")
FINAL_NORM_HINTS = ("norm", "ln_f", "final_layernorm", "model.norm", "transformer.ln_f")


def get_model_topology(model_id, model_params=None, analyzer_config=None, use_flashattention=False):
    cache_key = (model_id, bool(use_flashattention))
    if cache_key not in _TOPOLOGY_CACHE:
        _TOPOLOGY_CACHE[cache_key] = _build_model_topology(
            model_id=model_id,
            model_params=model_params,
            analyzer_config=analyzer_config,
            use_flashattention=use_flashattention,
        )
    return copy.deepcopy(_TOPOLOGY_CACHE[cache_key])


def _build_model_topology(model_id, model_params=None, analyzer_config=None, use_flashattention=False):
    if model_params is None:
        model_params = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    if analyzer_config is None:
        raise ValueError("analyzer_config is required for model introspection")

    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(model_params, trust_remote_code=True)

    named_modules = dict(model.named_modules())
    module_entries = _collect_module_entries(named_modules)
    block_info = _detect_repeated_block(module_entries)
    canonical_nodes = _build_canonical_nodes(
        module_entries,
        named_modules,
        block_info,
        model,
        analyzer_config,
        model_params,
        use_flashattention,
    )

    fallback_graph = analyzer_config.build_layer_graph(model_params, use_flashattention=use_flashattention)
    fallback_edges = analyzer_config.build_layer_edges(fallback_graph)

    fx_edges = []
    trace_error = None
    topology_source = "fallback"
    try:
        traced = symbolic_trace(model)
        fx_edges = _build_fx_edges(traced, canonical_nodes, block_info)
        if fx_edges:
            topology_source = "fx"
    except Exception as exc:  # pragma: no cover - exercised through runtime fallback
        trace_error = str(exc)

    if len(fx_edges) < max(4, len(fallback_edges) // 2):
        fx_edges = []
    edges = fx_edges if fx_edges else fallback_edges
    if not fx_edges:
        topology_source = "fallback"

    nodes = _materialize_nodes(canonical_nodes)
    combos = _build_combos(model, block_info, canonical_nodes)
    graph_info = {
        "architecture_name": model.__class__.__name__,
        "model_class": model.__class__.__name__,
        "block_repetition_count": block_info["count"],
        "topology_source": topology_source,
        "trace_error": trace_error,
        "module_count": len(module_entries),
    }
    if block_info["container_path"]:
        graph_info["block_container_path"] = block_info["container_path"]

    return {
        "nodes": nodes,
        "edges": _filter_edges_to_nodes(edges, nodes),
        "combos": combos,
        "graph_info": graph_info,
    }


def _collect_module_entries(named_modules):
    entries = []
    for path, module in named_modules.items():
        if path == "":
            continue
        if any(part.isdigit() for part in path.split(".")):
            parent_path = path.rsplit(".", 1)[0] if "." in path else ""
        else:
            parent_path = path.rsplit(".", 1)[0] if "." in path else ""
        entries.append(
            {
                "path": path,
                "module": module,
                "type": module.__class__.__name__,
                "parent_path": parent_path,
                "name": path.split(".")[-1],
            }
        )
    return entries


def _detect_repeated_block(module_entries):
    children_by_parent = defaultdict(list)
    type_by_path = {}
    for entry in module_entries:
        type_by_path[entry["path"]] = entry["type"]
        parts = entry["path"].split(".")
        for idx, part in enumerate(parts[:-1]):
            if part.isdigit():
                parent = ".".join(parts[:idx])
                child = ".".join(parts[: idx + 1])
                children_by_parent[parent].append(child)

    best = {"container_path": "", "representative_path": "", "count": 1}
    best_score = (-1, -1)
    for parent, child_paths in children_by_parent.items():
        indices = []
        child_types = []
        for child_path in child_paths:
            index = child_path.split(".")[-1]
            if index.isdigit():
                indices.append(int(index))
                child_types.append(type_by_path.get(child_path))
        if len(set(indices)) <= 1:
            continue
        if len(set(child_types)) != 1:
            continue
        score = (len(set(indices)), len(parent.split(".")))
        if score > best_score:
            best_score = score
            representative_path = f"{parent}.0" if parent else "0"
            best = {
                "container_path": parent,
                "representative_path": representative_path,
                "count": len(set(indices)),
            }
    return best


def _build_canonical_nodes(module_entries, named_modules, block_info, model, analyzer_config, model_params, use_flashattention):
    nodes = {}
    fallback_graph = analyzer_config.build_layer_graph(model_params, use_flashattention=use_flashattention)

    def add_node(node_id, label, combo_id, module_path=None, module=None, node_type="topology"):
        if node_id in nodes:
            return
        info = {"node_type": node_type}
        if module_path:
            info["module_path"] = module_path
        if module is not None:
            info["module_type"] = module.__class__.__name__
            dims = _extract_dimensions(module)
            if dims:
                info.update(dims)
        nodes[node_id] = {
            "id": node_id,
            "label": label,
            "title": label,
            "comboId": combo_id,
            "topology_info": info,
        }

    add_node("embedding", "Embedding", "combo_model", *_find_embedding(named_modules))
    add_node("final_norm", "Final Norm", "combo_model", *_find_final_norm(named_modules, block_info))
    lm_head_module = named_modules.get("lm_head")
    add_node("lm_head", "LM Head", "combo_model", "lm_head" if lm_head_module else None, lm_head_module)

    add_node("input", "Input", "combo_block", node_type="synthetic")
    add_node("output", "Output", "combo_block", node_type="synthetic")

    block_modules = _block_descendants(module_entries, block_info["representative_path"])
    canonical_paths = _map_block_modules_to_canonical(block_modules, fallback_graph)
    combo_by_node = _infer_combo_assignments(canonical_paths)

    for node_id in fallback_graph.keys():
        if node_id in TOP_LEVEL_NODE_IDS:
            continue
        module_path = canonical_paths.get(node_id)
        module = named_modules.get(module_path) if module_path else None
        node_type = "synthetic" if module is None else "module"
        label = _pretty_label(node_id)
        add_node(node_id, label, combo_by_node.get(node_id, "combo_block"), module_path, module, node_type)

    activation_path = _find_first_matching_path(block_modules, lambda entry: _is_activation_module(entry["module"]))
    if activation_path and "mlp_act" in nodes:
        nodes["mlp_act"]["topology_info"]["module_path"] = activation_path
        nodes["mlp_act"]["topology_info"]["module_type"] = named_modules[activation_path].__class__.__name__

    return nodes


def _find_embedding(named_modules):
    for path, module in named_modules.items():
        if not path or "." in path and any(part.isdigit() for part in path.split(".")):
            continue
        if isinstance(module, nn.Embedding) and any(hint in path for hint in EMBEDDING_HINTS):
            return path, module
    for path, module in named_modules.items():
        if isinstance(module, nn.Embedding):
            return path, module
    return None, None


def _find_final_norm(named_modules, block_info):
    for path, module in named_modules.items():
        if not path or path == block_info["container_path"] or path.startswith(block_info["representative_path"] + "."):
            continue
        lower = path.lower()
        if isinstance(module, nn.LayerNorm) or "rmsnorm" in module.__class__.__name__.lower():
            if any(hint in lower for hint in FINAL_NORM_HINTS):
                return path, module
    return None, None


def _block_descendants(module_entries, representative_path):
    descendants = []
    prefix = representative_path + "." if representative_path else ""
    for entry in module_entries:
        if entry["path"].startswith(prefix):
            descendants.append(entry)
    return descendants


def _map_block_modules_to_canonical(block_modules, fallback_graph):
    path_lookup = {entry["path"]: entry for entry in block_modules}
    candidates = defaultdict(list)
    for entry in block_modules:
        leaf = entry["name"].lower()
        for canonical_id, hints in CANONICAL_NAME_HINTS.items():
            if leaf in hints:
                candidates[canonical_id].append(entry["path"])

    canonical_paths = {}
    for canonical_id in fallback_graph.keys():
        if canonical_id in {"input", "output", "qk_matmul", "softmax", "sv_matmul", "fused_attention", "attn_add", "mlp_add", "mlp_act"}:
            continue
        if canonical_id in candidates:
            canonical_paths[canonical_id] = sorted(candidates[canonical_id], key=lambda path: (path.count("."), path))[0]

    for canonical_id in ("attn_norm", "mlp_norm"):
        if canonical_id in canonical_paths:
            continue
        path = _find_norm_fallback_path(block_modules, canonical_id)
        if path:
            canonical_paths[canonical_id] = path

    return canonical_paths


def _find_norm_fallback_path(block_modules, canonical_id):
    norm_entries = [
        entry for entry in block_modules
        if isinstance(entry["module"], nn.LayerNorm) or "rmsnorm" in entry["type"].lower()
    ]
    if not norm_entries:
        return None
    norm_entries = sorted(norm_entries, key=lambda entry: entry["path"])
    if canonical_id == "attn_norm":
        return norm_entries[0]["path"]
    if len(norm_entries) > 1:
        return norm_entries[1]["path"]
    return None


def _infer_combo_assignments(canonical_paths):
    combo_by_node = {}
    for node_id in CANONICAL_BLOCK_NODE_ORDER:
        if node_id in {"input", "output", "attn_add", "mlp_add", "attn_norm", "mlp_norm"}:
            combo_by_node[node_id] = "combo_block"
            continue
        path = canonical_paths.get(node_id, "")
        lower = path.lower()
        if any(hint in lower for hint in MOE_PATH_HINTS) and node_id in {"moe_router", "gate_proj", "up_proj", "down_proj", "mlp_act"}:
            combo_by_node[node_id] = "combo_moe"
        elif any(hint in lower for hint in ATTENTION_PATH_HINTS) or node_id in {"q_proj", "k_proj", "v_proj", "qk_matmul", "softmax", "sv_matmul", "fused_attention", "out_proj"}:
            combo_by_node[node_id] = "combo_attention"
        elif any(hint in lower for hint in MLP_PATH_HINTS) or node_id in {"gate_proj", "up_proj", "moe_router", "mlp_act", "down_proj"}:
            combo_by_node[node_id] = "combo_mlp"
        else:
            combo_by_node[node_id] = "combo_block"
    return combo_by_node


def _materialize_nodes(canonical_nodes):
    nodes = []
    for node_id, node in canonical_nodes.items():
        topology_info = node.get("topology_info", {})
        description = topology_info.get("module_type", topology_info.get("node_type", ""))
        nodes.append(
            {
                "id": node_id,
                "label": node["label"],
                "title": node["title"],
                "description": description,
                "comboId": node.get("comboId"),
                "info": topology_info,
            }
        )
    return nodes


def _build_combos(model, block_info, canonical_nodes):
    combos = [
        {
            "id": "combo_model",
            "label": model.__class__.__name__,
            "type": "rect",
            "style": {"fill": "#F8FAFC", "stroke": "#CBD5E1"},
        }
    ]
    if block_info["count"] > 1:
        combos.append(
            {
                "id": "combo_block",
                "parentId": "combo_model",
                "label": f"Transformer Block x{block_info['count']}",
                "type": "rect",
                "collapsed": True,
                "style": {"fill": "#EFF6FF", "stroke": "#60A5FA"},
            }
        )
        combos.append(
            {
                "id": "combo_attention",
                "parentId": "combo_block",
                "label": "Self Attention",
                "type": "rect",
                "style": {"fill": "#F8FAFC", "stroke": "#93C5FD"},
            }
        )
        combos.append(
            {
                "id": "combo_mlp",
                "parentId": "combo_block",
                "label": "MLP",
                "type": "rect",
                "style": {"fill": "#FEFCE8", "stroke": "#FACC15"},
            }
        )
        if any(node.get("comboId") == "combo_moe" for node in canonical_nodes.values()):
            combos.append(
                {
                    "id": "combo_moe",
                    "parentId": "combo_block",
                    "label": "MoE",
                    "type": "rect",
                    "style": {"fill": "#FFF7ED", "stroke": "#FB923C"},
                }
            )
    else:
        combos.append(
            {
                "id": "combo_block",
                "parentId": "combo_model",
                "label": "Transformer Block",
                "type": "rect",
                "style": {"fill": "#EFF6FF", "stroke": "#60A5FA"},
            }
        )
        combos.append(
            {
                "id": "combo_attention",
                "parentId": "combo_block",
                "label": "Self Attention",
                "type": "rect",
                "style": {"fill": "#F8FAFC", "stroke": "#93C5FD"},
            }
        )
        combos.append(
            {
                "id": "combo_mlp",
                "parentId": "combo_block",
                "label": "MLP",
                "type": "rect",
                "style": {"fill": "#FEFCE8", "stroke": "#FACC15"},
            }
        )
    return combos


def _build_fx_edges(traced, canonical_nodes, block_info):
    top_level_targets = {
        canonical_nodes[node_id]["topology_info"].get("module_path"): node_id
        for node_id in TOP_LEVEL_NODE_IDS
        if node_id in canonical_nodes and canonical_nodes[node_id]["topology_info"].get("module_path")
    }
    block_target_prefix = block_info["representative_path"] + "." if block_info["representative_path"] else ""
    target_to_canonical = {}
    for node_id, node in canonical_nodes.items():
        module_path = node["topology_info"].get("module_path")
        if not module_path:
            continue
        target_to_canonical[module_path] = node_id

    node_map = {}
    add_sequence = 0
    for fx_node in traced.graph.nodes:
        mapped = None
        if fx_node.op == "placeholder":
            mapped = "input"
        elif fx_node.op == "call_module":
            target = str(fx_node.target)
            if target in top_level_targets:
                mapped = top_level_targets[target]
            elif target in target_to_canonical:
                mapped = target_to_canonical[target]
            elif block_target_prefix and target.startswith(block_target_prefix):
                remainder = target[len(block_target_prefix):]
                for module_path, canonical_id in target_to_canonical.items():
                    if module_path.endswith(remainder):
                        mapped = canonical_id
                        break
        elif fx_node.op == "call_function" and fx_node.target in {operator.add, torch.add}:
            if add_sequence == 0:
                mapped = "attn_add"
            elif add_sequence == 1:
                mapped = "mlp_add"
            add_sequence += 1
        elif fx_node.op == "output":
            mapped = "output"

        if mapped in canonical_nodes:
            node_map[fx_node] = mapped

    edges = []
    seen = set()
    for fx_node, target_id in node_map.items():
        if fx_node.op == "placeholder":
            continue
        source_ids = []
        for input_node in fx_node.all_input_nodes:
            source_id = node_map.get(input_node)
            if source_id:
                source_ids.append(source_id)
        for source_id in source_ids:
            edge_type = _infer_edge_type_from_add(target_id, source_id)
            edge_key = (source_id, target_id, edge_type)
            if edge_key not in seen:
                seen.add(edge_key)
                edges.append({"source": source_id, "target": target_id, "edgeType": edge_type})
    return edges


def _infer_edge_type_from_add(target_id, source_id):
    if target_id == "attn_add" and source_id == "input":
        return "residual"
    if target_id == "mlp_add" and source_id in {"attn_add", "input"}:
        return "residual"
    return "data_flow"


def _filter_edges_to_nodes(edges, nodes):
    valid_ids = {node["id"] for node in nodes}
    filtered = []
    for edge in edges:
        if edge["source"] in valid_ids and edge["target"] in valid_ids:
            filtered.append(edge)
    return filtered


def _extract_dimensions(module):
    if isinstance(module, nn.Linear):
        return {"in_features": module.in_features, "out_features": module.out_features}
    if isinstance(module, nn.Embedding):
        return {"num_embeddings": module.num_embeddings, "embedding_dim": module.embedding_dim}
    if isinstance(module, nn.LayerNorm):
        normalized_shape = module.normalized_shape
        if isinstance(normalized_shape, tuple):
            normalized_shape = "x".join(str(v) for v in normalized_shape)
        return {"normalized_shape": normalized_shape}
    weight = getattr(module, "weight", None)
    if weight is not None and hasattr(weight, "shape"):
        return {"weight_shape": list(weight.shape)}
    return {}


def _pretty_label(node_id):
    mapping = {
        "attn_norm": "Attention Norm",
        "mlp_norm": "MLP Norm",
        "q_proj": "Q Projection",
        "k_proj": "K Projection",
        "v_proj": "V Projection",
        "qk_matmul": "QK MatMul",
        "softmax": "Softmax",
        "sv_matmul": "SV MatMul",
        "fused_attention": "Fused Attention",
        "out_proj": "Output Projection",
        "attn_add": "Attention Residual Add",
        "gate_proj": "Gate Projection",
        "up_proj": "Up Projection",
        "moe_router": "MoE Router",
        "mlp_act": "Activation",
        "down_proj": "Down Projection",
        "mlp_add": "MLP Residual Add",
    }
    return mapping.get(node_id, node_id.replace("_", " ").title())


def _is_activation_module(module):
    return isinstance(
        module,
        (
            nn.GELU,
            nn.ReLU,
            nn.SiLU,
            nn.Sigmoid,
            nn.Tanh,
            nn.LeakyReLU,
        ),
    )


def _find_first_matching_path(entries, predicate):
    matches = [entry["path"] for entry in entries if predicate(entry)]
    if not matches:
        return None
    return sorted(matches)[0]
