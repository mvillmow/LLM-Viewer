import copy
from collections import defaultdict

import torch
import torch.nn as nn
import transformers
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM


_TOPOLOGY_CACHE = {}


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
    "mlp_act": {"act", "activation"},
}

EMBEDDING_HINTS = ("embed_tokens", "wte", "tok_embeddings", "word_embeddings", "embeddings")
FINAL_NORM_HINTS = ("norm", "ln_f", "final_layernorm", "model.norm", "transformer.ln_f")
ROLE_ORDER = [
    "attn_norm",
    "q_proj",
    "k_proj",
    "v_proj",
    "out_proj",
    "mlp_norm",
    "gate_proj",
    "up_proj",
    "moe_router",
    "mlp_act",
    "down_proj",
]
HIDDEN_ROLE_IDS = {"input", "output", "qk_matmul", "softmax", "sv_matmul", "fused_attention", "attn_add", "mlp_add"}


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

    model, model_build_error = _build_meta_model(model_params, analyzer_config)
    if model is None:
        return {
            "nodes": [],
            "edges": [],
            "combos": [],
            "graph_info": {
                "architecture_name": _get_architecture_name(None, model_params),
                "model_class": _get_architecture_name(None, model_params),
                "block_repetition_count": analyzer_config.get_num_hidden_layers(model_params),
                "topology_source": "unavailable",
                "connectivity_status": "partial",
                "trace_error": model_build_error,
                "module_count": 0,
                "stats_source": "dynamic_topology",
                "functional_op_attribution": "folded_into_modules",
            },
        }

    named_modules = dict(model.named_modules())
    module_entries = _collect_module_entries(named_modules)
    block_info = _detect_repeated_block(module_entries)
    visible_paths = _select_visible_module_paths(module_entries, block_info)
    role_to_path = _infer_metric_roles(module_entries, named_modules, block_info, analyzer_config, model_params, use_flashattention)
    nodes = _build_nodes(visible_paths, named_modules, block_info, role_to_path, model, model_params)
    edges, connectivity_status = _build_edges(nodes, role_to_path, block_info, analyzer_config, model_params, use_flashattention)
    combos = _build_combos(nodes, block_info, model, model_params)

    graph_info = {
        "architecture_name": _get_architecture_name(model, model_params),
        "model_class": _get_architecture_name(model, model_params),
        "block_repetition_count": block_info["count"],
        "topology_source": "dynamic_fallback",
        "connectivity_status": connectivity_status,
        "trace_error": model_build_error,
        "module_count": len(module_entries),
        "stats_source": "dynamic_topology",
        "functional_op_attribution": "folded_into_modules",
    }
    if block_info["container_path"]:
        graph_info["block_container_path"] = block_info["container_path"]

    return {
        "nodes": nodes,
        "edges": edges,
        "combos": combos,
        "graph_info": graph_info,
    }


def _build_meta_model(model_params, analyzer_config):
    normalized_config = _prepare_config_for_model_build(model_params, analyzer_config)
    for builder in (_build_architecture_model, _build_causallm_model, _build_base_model):
        try:
            with torch.device("meta"):
                return builder(normalized_config), None
        except Exception as exc:  # pragma: no cover - runtime compatibility path
            last_error = str(exc)
    return None, last_error


def _build_architecture_model(model_params):
    architectures = getattr(model_params, "architectures", None) or []
    for architecture_name in architectures:
        model_cls = getattr(transformers, architecture_name, None)
        if model_cls is None:
            continue
        return model_cls(model_params)
    raise ValueError("No loadable architecture class found")


def _build_causallm_model(model_params):
    return AutoModelForCausalLM.from_config(model_params, trust_remote_code=True)


def _build_base_model(model_params):
    return AutoModel.from_config(model_params, trust_remote_code=True)


def _prepare_config_for_model_build(model_params, analyzer_config):
    normalized_config = copy.deepcopy(model_params)
    fallback_values = {
        "vocab_size": analyzer_config.get_vocab_size(model_params),
        "hidden_size": analyzer_config.get_hidden_size(model_params),
        "intermediate_size": analyzer_config.get_intermediate_size(model_params),
        "num_hidden_layers": analyzer_config.get_num_hidden_layers(model_params),
        "num_attention_heads": analyzer_config.get_num_attention_heads(model_params),
        "num_key_value_heads": analyzer_config.get_num_key_value_heads(model_params),
    }
    for attr_name, attr_value in fallback_values.items():
        if attr_value is None:
            continue
        if not hasattr(normalized_config, attr_name) or getattr(normalized_config, attr_name, None) is None:
            setattr(normalized_config, attr_name, attr_value)
    if not hasattr(normalized_config, "pad_token_id"):
        normalized_config.pad_token_id = None
    return normalized_config


def _collect_module_entries(named_modules):
    child_counts = defaultdict(int)
    for path in named_modules:
        if not path:
            continue
        parent = path.rsplit(".", 1)[0] if "." in path else ""
        child_counts[parent] += 1

    entries = []
    for path, module in named_modules.items():
        if path == "":
            continue
        entries.append(
            {
                "path": path,
                "module": module,
                "type": module.__class__.__name__,
                "name": path.split(".")[-1],
                "parent_path": path.rsplit(".", 1)[0] if "." in path else "",
                "is_leaf": child_counts.get(path, 0) == 0,
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
            if not index.isdigit():
                continue
            indices.append(int(index))
            child_types.append(type_by_path.get(child_path))
        if len(set(indices)) <= 1 or len(set(child_types)) != 1:
            continue
        score = (len(set(indices)), len(parent.split(".")))
        if score > best_score:
            best_score = score
            best = {
                "container_path": parent,
                "representative_path": f"{parent}.0" if parent else "0",
                "count": len(set(indices)),
            }
    return best


def _select_visible_module_paths(module_entries, block_info):
    visible_paths = []
    representative_prefix = block_info["representative_path"] + "." if block_info["representative_path"] else ""
    repeated_prefix = block_info["container_path"] + "." if block_info["container_path"] else ""

    for entry in module_entries:
        if not entry["is_leaf"]:
            continue
        path = entry["path"]
        if block_info["container_path"] and path.startswith(repeated_prefix):
            if representative_prefix and path.startswith(representative_prefix):
                visible_paths.append(path)
            continue
        visible_paths.append(path)
    return sorted(set(visible_paths), key=lambda path: (path.count("."), path))


def _infer_metric_roles(module_entries, named_modules, block_info, analyzer_config, model_params, use_flashattention):
    fallback_graph = analyzer_config.build_layer_graph(model_params, use_flashattention=use_flashattention)
    block_modules = _block_descendants(module_entries, block_info["representative_path"])
    role_to_path = _map_block_modules_to_roles(block_modules, fallback_graph)

    embedding_path, _ = _find_embedding(named_modules)
    if embedding_path:
        role_to_path["embedding"] = embedding_path

    final_norm_path, _ = _find_final_norm(named_modules, block_info)
    if final_norm_path:
        role_to_path["final_norm"] = final_norm_path

    if "lm_head" in named_modules:
        role_to_path["lm_head"] = "lm_head"

    activation_path = _find_first_matching_path(block_modules, lambda entry: _is_activation_module(entry["module"]))
    if activation_path:
        role_to_path["mlp_act"] = activation_path
    return role_to_path


def _block_descendants(module_entries, representative_path):
    if not representative_path:
        return []
    prefix = representative_path + "."
    return [entry for entry in module_entries if entry["path"].startswith(prefix)]


def _map_block_modules_to_roles(block_modules, fallback_graph):
    candidates = defaultdict(list)
    for entry in block_modules:
        leaf = entry["name"].lower()
        for canonical_id, hints in CANONICAL_NAME_HINTS.items():
            if leaf in hints:
                candidates[canonical_id].append(entry["path"])

    role_to_path = {}
    for role in fallback_graph.keys():
        if role in HIDDEN_ROLE_IDS:
            continue
        if role in candidates:
            role_to_path[role] = sorted(candidates[role], key=lambda path: (path.count("."), path))[0]

    for role in ("attn_norm", "mlp_norm"):
        if role in role_to_path:
            continue
        path = _find_norm_fallback_path(block_modules, role)
        if path:
            role_to_path[role] = path
    return role_to_path


def _find_embedding(named_modules):
    for path, module in named_modules.items():
        if not path:
            continue
        if isinstance(module, nn.Embedding) and any(hint in path for hint in EMBEDDING_HINTS):
            return path, module
    for path, module in named_modules.items():
        if isinstance(module, nn.Embedding):
            return path, module
    return None, None


def _find_final_norm(named_modules, block_info):
    repeated_prefix = block_info["container_path"] + "." if block_info["container_path"] else ""
    for path, module in named_modules.items():
        if not path:
            continue
        if repeated_prefix and path.startswith(repeated_prefix):
            continue
        lower = path.lower()
        if isinstance(module, nn.LayerNorm) or "rmsnorm" in module.__class__.__name__.lower():
            if any(hint in lower for hint in FINAL_NORM_HINTS):
                return path, module
    return None, None


def _find_norm_fallback_path(block_modules, role):
    norm_entries = [
        entry for entry in block_modules
        if isinstance(entry["module"], nn.LayerNorm) or "rmsnorm" in entry["type"].lower()
    ]
    if not norm_entries:
        return None
    norm_entries = sorted(norm_entries, key=lambda entry: entry["path"])
    if role == "attn_norm":
        return norm_entries[0]["path"]
    if len(norm_entries) > 1:
        return norm_entries[1]["path"]
    return None


def _build_nodes(visible_paths, named_modules, block_info, role_to_path, model, model_params):
    path_to_role = {path: role for role, path in role_to_path.items()}
    nodes = []
    subtree_combo_ids = _get_subtree_combo_ids(visible_paths, block_info)
    for path in visible_paths:
        module = named_modules[path]
        info = {
            "node_type": "module",
            "module_path": path,
            "module_type": module.__class__.__name__,
            "parent_path": path.rsplit(".", 1)[0] if "." in path else "",
            "stats_source": "dynamic_topology",
            "repeat_factor": block_info["count"] if _is_repeated_block_path(path, block_info) else 1,
        }
        metric_key = path_to_role.get(path)
        if metric_key:
            info["metric_key"] = metric_key
        info.update(_extract_dimensions(module))
        nodes.append(
            {
                "id": path,
                "label": path.split(".")[-1],
                "title": path,
                "description": module.__class__.__name__,
                "comboId": _determine_combo_id(path, block_info, subtree_combo_ids),
                "info": info,
            }
        )
    return nodes


def _get_subtree_combo_ids(visible_paths, block_info):
    subtree_combo_ids = {}
    representative_parts = block_info["representative_path"].split(".") if block_info["representative_path"] else []
    prefix_len = len(representative_parts)
    subtree_counts = defaultdict(int)
    for path in visible_paths:
        parts = path.split(".")
        if prefix_len and parts[:prefix_len] == representative_parts and len(parts) > prefix_len + 1:
            subtree = parts[prefix_len]
            subtree_counts[subtree] += 1
    for subtree, count in subtree_counts.items():
        if count > 1:
            subtree_combo_ids[subtree] = f"combo_subtree_{subtree}"
    return subtree_combo_ids


def _determine_combo_id(path, block_info, subtree_combo_ids):
    if not _is_repeated_block_path(path, block_info):
        return "combo_model"
    representative_parts = block_info["representative_path"].split(".") if block_info["representative_path"] else []
    parts = path.split(".")
    prefix_len = len(representative_parts)
    if len(parts) > prefix_len + 1:
        subtree = parts[prefix_len]
        if subtree in subtree_combo_ids:
            return subtree_combo_ids[subtree]
    return "combo_block"


def _is_repeated_block_path(path, block_info):
    prefix = block_info["representative_path"] + "." if block_info["representative_path"] else ""
    return bool(prefix and path.startswith(prefix))


def _build_edges(nodes, role_to_path, block_info, analyzer_config, model_params, use_flashattention):
    if role_to_path:
        collapsed_edges = _build_role_based_edges(role_to_path, analyzer_config, model_params, use_flashattention)
        if collapsed_edges:
            return collapsed_edges, "full"
    return _build_sequential_edges(nodes), "partial"


def _build_role_based_edges(role_to_path, analyzer_config, model_params, use_flashattention):
    role_graph = analyzer_config.build_layer_graph(model_params, use_flashattention=use_flashattention)
    adjacency = defaultdict(list)
    for target_role, source_roles in role_graph.items():
        for source_role in source_roles:
            adjacency[source_role].append((target_role, _get_role_edge_type(source_role, target_role)))

    visible_roles = {role for role in role_to_path if role not in HIDDEN_ROLE_IDS}
    edges = []
    seen = set()

    def walk_visible_targets(role, edge_type, visited):
        if role in visible_roles:
            return {(role, edge_type)}
        if role in visited:
            return set()
        results = set()
        next_visited = set(visited)
        next_visited.add(role)
        for target_role, next_type in adjacency.get(role, []):
            results.update(
                walk_visible_targets(
                    target_role,
                    "residual" if edge_type == "residual" or next_type == "residual" else "data_flow",
                    next_visited,
                )
            )
        return results

    for source_role in visible_roles:
        for target_role, edge_type in adjacency.get(source_role, []):
            for visible_target_role, visible_edge_type in walk_visible_targets(target_role, edge_type, set()):
                source_id = role_to_path.get(source_role)
                target_id = role_to_path.get(visible_target_role)
                if not source_id or not target_id or source_id == target_id:
                    continue
                edge_key = (source_id, target_id, visible_edge_type)
                if edge_key in seen:
                    continue
                seen.add(edge_key)
                edges.append(
                    {
                        "source": source_id,
                        "target": target_id,
                        "edgeType": visible_edge_type,
                    }
                )

    if role_to_path.get("embedding") and role_to_path.get("attn_norm"):
        edges.append({"source": role_to_path["embedding"], "target": role_to_path["attn_norm"], "edgeType": "data_flow"})
    if role_to_path.get("final_norm") and role_to_path.get("lm_head"):
        edges.append({"source": role_to_path["final_norm"], "target": role_to_path["lm_head"], "edgeType": "data_flow"})
    elif role_to_path.get("down_proj") and role_to_path.get("lm_head"):
        edges.append({"source": role_to_path["down_proj"], "target": role_to_path["lm_head"], "edgeType": "data_flow"})

    return _normalize_edges(edges)


def _build_sequential_edges(nodes):
    sorted_nodes = sorted(nodes, key=lambda node: (node["id"].count("."), node["id"]))
    edges = []
    for index in range(len(sorted_nodes) - 1):
        edges.append(
            {
                "source": sorted_nodes[index]["id"],
                "target": sorted_nodes[index + 1]["id"],
                "edgeType": "data_flow",
            }
        )
    return edges


def _get_role_edge_type(source_role, target_role):
    if target_role == "attn_add" and source_role == "input":
        return "residual"
    if target_role == "mlp_add" and source_role in {"input", "attn_add"}:
        return "residual"
    return "data_flow"


def _build_combos(nodes, block_info, model, model_params):
    architecture_name = _get_architecture_name(model, model_params)
    combos = [
        {
            "id": "combo_model",
            "label": architecture_name,
            "type": "rect",
            "style": {"fill": "#F8FAFC", "stroke": "#CBD5E1"},
        }
    ]
    if block_info["count"] > 1:
        block_label = _prettify_segment(block_info["container_path"].split(".")[-1] if block_info["container_path"] else "block")
        combos.append(
            {
                "id": "combo_block",
                "parentId": "combo_model",
                "label": f"{block_label} x{block_info['count']}",
                "type": "rect",
                "style": {"fill": "#EFF6FF", "stroke": "#60A5FA"},
            }
        )
    elif any(node["comboId"] == "combo_block" for node in nodes):
        combos.append(
            {
                "id": "combo_block",
                "parentId": "combo_model",
                "label": _prettify_segment(block_info["representative_path"].split(".")[-1] if block_info["representative_path"] else "block"),
                "type": "rect",
                "style": {"fill": "#EFF6FF", "stroke": "#60A5FA"},
            }
        )

    subtree_ids = sorted({node["comboId"] for node in nodes if node["comboId"] not in {"combo_model", "combo_block"}})
    for combo_id in subtree_ids:
        subtree_name = combo_id.replace("combo_subtree_", "")
        combos.append(
            {
                "id": combo_id,
                "parentId": "combo_block",
                "label": _prettify_segment(subtree_name),
                "type": "rect",
                "style": {"fill": "#F8FAFC", "stroke": "#93C5FD"},
            }
        )
    return combos


def _extract_dimensions(module):
    info = {}
    if isinstance(module, nn.Linear):
        info.update({"in_features": module.in_features, "out_features": module.out_features})
    if isinstance(module, nn.Embedding):
        info.update({"num_embeddings": module.num_embeddings, "embedding_dim": module.embedding_dim})
    normalized_shape = getattr(module, "normalized_shape", None)
    if normalized_shape is not None:
        if isinstance(normalized_shape, tuple):
            normalized_shape = list(normalized_shape)
        info["normalized_shape"] = normalized_shape
    weight = getattr(module, "weight", None)
    if weight is not None and hasattr(weight, "shape"):
        info["weight_shape"] = list(weight.shape)
    return info


def _normalize_edges(edges):
    normalized = []
    seen = set()
    for edge in edges:
        key = (edge["source"], edge["target"], edge.get("edgeType", "data_flow"))
        if key in seen or edge["source"] == edge["target"]:
            continue
        seen.add(key)
        normalized.append(
            {
                "source": edge["source"],
                "target": edge["target"],
                "edgeType": edge.get("edgeType", "data_flow"),
            }
        )
    return normalized


def _get_architecture_name(model, model_params):
    if model is not None:
        return model.__class__.__name__
    architectures = getattr(model_params, "architectures", None)
    if architectures:
        return architectures[0]
    return model_params.__class__.__name__


def _prettify_segment(segment):
    return segment.replace("_", " ").replace("-", " ").title()


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
