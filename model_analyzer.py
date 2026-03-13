import os
import importlib
from hardwares.hardware_params import hardware_params
from roofline_model import roofline_analyze
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from utils import str_number, str_number_time
import math

ALL_DATA_NAMES = [
    "OPs",
    "memory_access",
    "load_weight",
    "load_act",
    "store_act",
    "load_kv_cache",
    "store_kv_cache",
    "inference_time",
]


class ModelAnalyzer:
    def __init__(self, model_id, hardware, config_file=None):
        """
        HuggingFace-only model analyzer.
        """
        self.model_id = model_id
        self.hardware = hardware
        if config_file is None:
            # Always use generic.py for HuggingFace models - it auto-detects architecture
            config_file = "configs/generic.py"
        print(f"use config file {config_file} for {model_id}")
        # Always load from HuggingFace using AutoConfig
        self.model_params = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        self.config = importlib.import_module(config_file.replace("/", ".").replace(".py", ""))

        # temporary variables
        self.results = None
        self.w_bit = None
        self.a_bit = None
        self.kv_bit = None
        self.batchsize = None
        self.seqlen = None

    def _analyze_to_results(
        self,
        stage,
        name,
        OPs=0,
        load_weight=0,
        load_act=0,
        store_act=0,
        load_kv_cache=0,
        store_kv_cache=0,
    ):

        bandwidth, max_OPS, onchip_buffer = self.get_hardware_info()
        memory_access = load_weight + load_act + store_act + load_kv_cache + store_kv_cache
        arithmetic_intensity, performance, bound = roofline_analyze(bandwidth, max_OPS, OPs, memory_access)
        inference_time = OPs / performance
        self.results[stage][name] = {
            "OPs": OPs,
            "memory_access": memory_access,
            "arithmetic_intensity": arithmetic_intensity,
            "performance": performance,
            "bound": bound,
            "load_weight": load_weight,
            "load_act": load_act,
            "store_act": store_act,
            "load_kv_cache": load_kv_cache,
            "store_kv_cache": store_kv_cache,
            "inference_time": inference_time,
        }

    def save_csv(self, save_path=None):
        if save_path is None:
            save_path = f"output/{self.model_id[:self.model_id.rfind('/')]}"
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            save_path += f"{self.model_id[self.model_id.rfind('/'):]}"

        decode_file_name = f"{save_path}_decode.csv"
        prefill_file_name = f"{save_path}_prefill.csv"
        print(f"save to {decode_file_name} and {prefill_file_name}")

        for file_name, stage in [
            (decode_file_name, "decode"),
            (prefill_file_name, "prefill"),
        ]:
            with open(file_name, "a+") as f:

                f.write(
                    f"\n\n=== {self.model_id} {self.hardware} w_bit={self.w_bit} a_bit={self.a_bit} kv_bit={self.kv_bit} batchsize={self.batchsize} seqlen={self.seqlen} tp_size={self.tp_size} ===\n"
                )
                # legend
                f.write(
                    f"layer_name,OPs,Access,arithmetic_intensity,performance,bound,load_weight,load_act,store_act,load_kv_cache,store_kv_cache,inference_time\n"
                )
            with open(file_name, "a+") as f:
                for layer_name, result in self.results[stage].items():
                    f.write(
                        f"{layer_name},{str_number(result['OPs'])},{str_number(result['memory_access'])}B,{str_number(result['arithmetic_intensity'])},{str_number(result['performance'])},"
                        f"{result['bound']},{str_number(result['load_weight'])}B,{str_number(result['load_act'])}B,{str_number(result['store_act'])}B,{str_number(result['load_kv_cache'])}B,"
                        f"{str_number(result['store_kv_cache'])}B,{str_number_time(result['inference_time'])}s\n"
                    )

    def analyze(
        self,
        seqlen,
        batchsize,
        w_bit=16,
        a_bit=16,
        kv_bit=None,
        use_flashattention=False,
        kv_token_ratio=1,
        tp_size: int = 1
    ):
        """
        seqlen: sequence length
        batchsize: batch size
        w_bit: weight bit
        a_bit: activation bit
        kv_bit: key and value bit. if it is None, it will be the same as a_bit
        use_flashattention: use flash attention/flash decoding
        kv_token_ratio: use this for KV compression
        tp_size: the number of devices for tensor parallelism to use

        return is a dict with the following format:
        {
            "decode": {
                    "layer_name": {
                            "OPs": "",
                            "memory_access": "",
                            "arithmetic_intensity": "",
                            "performance": "",
                            "bound": "",
                            "load_weight": "",
                            "load_act": "",
                            "store_act": "",
                            "load_kv_cache": "",
                            "store_kv_cache": "",
                            "inference_time": ""
                    }
            },
            "prefill": {
                    "layer_name": {
                            "OPs": "",
                            "memory_access": "",
                            "arithmetic_intensity": "",
                            "performance": "",
                            "bound": "",
                            "load_weight": "",
                            "load_act": "",
                            "store_act": "",
                            "load_kv_cache": "",
                            "store_kv_cache": "",
                            "inference_time": ""
                    }
            },
            "total_results": {
                "decode": {},
                "prefill": {}
            }
        }
        """
        assert seqlen > 0
        assert batchsize > 0
        self.results = {"decode": {}, "prefill": {}}
        if kv_bit is None:
            kv_bit = a_bit
        self.w_bit = w_bit
        self.a_bit = a_bit
        self.kv_bit = kv_bit
        self.batchsize = batchsize
        self.seqlen = seqlen
        self.tp_size = tp_size

        w_byte = self.w_bit / 8
        a_byte = self.a_bit / 8
        kv_byte = self.kv_bit / 8

        config = self.config
        model_params = self.model_params
        num_attention_heads = config.get_num_attention_heads(model_params)
        hidden_size = config.get_hidden_size(model_params)
        num_key_value_heads = config.get_num_key_value_heads(model_params)
        num_hidden_layers = config.get_num_hidden_layers(model_params)

        for name, (ic, oc) in config.get_linear_layers(model_params, tp_size).items():
            # for linear layers
            is_kv_proj = name in ["k_proj", "v_proj"]
            is_normal_proj = not is_kv_proj
            self._analyze_to_results(
                "decode",
                name,
                OPs=ic * oc * batchsize * 2,
                load_weight=ic * oc * w_byte,
                load_act=ic * batchsize * a_byte,
                store_act=0 if is_kv_proj else oc * batchsize * a_byte,
                load_kv_cache=0,
                store_kv_cache=(0 if is_normal_proj else oc * batchsize * kv_byte),
            )
            # for prefill
            self._analyze_to_results(
                "prefill",
                name,
                OPs=ic * oc * batchsize * seqlen * 2,
                load_weight=ic * oc * w_byte,
                load_act=ic * batchsize * seqlen * a_byte,
                store_act=(0 if is_kv_proj else oc * batchsize * seqlen * a_byte),
                load_kv_cache=0,
                store_kv_cache=(0 if is_normal_proj else oc * batchsize * seqlen * kv_byte),
            )

        # for attention
        head_size = hidden_size // num_attention_heads
        # for decode
        qk_matmul_OPs = seqlen * head_size * num_attention_heads * batchsize * 2
        sv_matmul_OPs = 1 * head_size * seqlen * num_attention_heads * batchsize * 2
        # the softmax operation takes five steps:
        # max_x=max(x)
        # x=x-max_x
        # x_exp=exp(x)
        # sum_x_exp=sum(x_exp)
        # y=x_exp/sum(x_exp)
        softmax_OPs = batchsize * num_attention_heads * seqlen * 1 * 5
        if use_flashattention:
            name = f"fused_attention"
            bandwidth, max_OPS, onchip_buffer = self.get_hardware_info()
            # flashattention-2 https://arxiv.org/pdf/2307.08691.pdf
            block_size_r = min(math.ceil(onchip_buffer / (kv_byte * head_size)), head_size)
            n_blocks_r = math.ceil(1 / block_size_r)
            q_numel = (1) * head_size * batchsize * num_attention_heads * a_byte
            o_numel = 1 * seqlen * batchsize * num_attention_heads * a_byte
            self._analyze_to_results(
                "decode",
                name,
                OPs=qk_matmul_OPs + sv_matmul_OPs + softmax_OPs,
                load_weight=0,
                load_act=q_numel,
                store_act=o_numel * 2,  # initialize O and save O
                load_kv_cache=n_blocks_r * (seqlen) * head_size * batchsize * num_key_value_heads * kv_byte * 2,
                store_kv_cache=0,
            )

        else:
            name = f"qk_matmul"
            self._analyze_to_results(
                "decode",
                name,
                OPs=qk_matmul_OPs,
                load_weight=0,
                load_act=(1) * head_size * batchsize * num_attention_heads * a_byte,
                store_act=1 * seqlen * batchsize * num_attention_heads * a_byte,
                load_kv_cache=(seqlen) * head_size * batchsize * num_key_value_heads * kv_byte,
                store_kv_cache=0,
            )
            name = f"sv_matmul"
            self._analyze_to_results(
                "decode",
                name,
                OPs=sv_matmul_OPs,
                load_weight=0,
                load_act=(1 * seqlen * batchsize * num_attention_heads) * a_byte,
                store_act=1 * head_size * batchsize * num_attention_heads * a_byte,
                load_kv_cache=(seqlen * head_size * batchsize * num_key_value_heads) * kv_byte,
                store_kv_cache=0,
            )

            name = f"softmax"
            # max sub exp sum div
            self._analyze_to_results(
                "decode",
                name,
                OPs=softmax_OPs,
                load_weight=0,
                load_act=batchsize * num_attention_heads * seqlen * 1 * a_byte,
                store_act=batchsize * num_attention_heads * seqlen * 1 * a_byte,
                load_kv_cache=0,
                store_kv_cache=0,
            )

        # Get elementwise layer specs from config for dynamic analysis
        elementwise_layers = config.get_elementwise_layers(model_params)
        layer_formulas = getattr(config, 'ELEMENTWISE_LAYER_FORMULAS', {})
        
        for name, layer_spec in elementwise_layers.items():
            layer_type = layer_spec.get("type", "unknown")
            formula = layer_formulas.get(layer_type, {})
            ops_per_element = formula.get("OPs_per_element", 1)
            
            # Compute OPs based on layer type
            OPs = batchsize * hidden_size * ops_per_element
            load_act = batchsize * hidden_size * a_byte
            # Activation layers typically have 2x activation load
            if layer_type == "activation":
                load_act = batchsize * hidden_size * a_byte * 2
            store_act = batchsize * hidden_size * a_byte
            
            self._analyze_to_results(
                "decode",
                name,
                OPs=OPs,
                load_weight=0,
                load_act=load_act,
                store_act=store_act,
                load_kv_cache=0,
                store_kv_cache=0,
            )

        # for prefill
        qk_matmul_OPs = seqlen * seqlen * head_size * num_attention_heads * batchsize * 2
        sv_matmul_OPs = seqlen * head_size * seqlen * num_attention_heads * batchsize * 2
        softmax_OPs = batchsize * num_attention_heads * seqlen * seqlen * 5
        if use_flashattention:
            name = f"fused_attention"
            bandwidth, max_OPS, onchip_buffer = self.get_hardware_info()
            # flashattention-2 https://arxiv.org/pdf/2307.08691.pdf
            block_size_r = min(math.ceil(onchip_buffer / (kv_byte * head_size)), head_size)
            n_blocks_r = math.ceil(seqlen / block_size_r)
            q_numel = seqlen * head_size * batchsize * num_attention_heads * a_byte
            o_numel = seqlen * seqlen * batchsize * num_attention_heads * a_byte
            self._analyze_to_results(
                "prefill",
                name,
                OPs=qk_matmul_OPs + sv_matmul_OPs + softmax_OPs,
                load_weight=0,
                load_act=q_numel,
                store_act=o_numel * 2,  # initialize O and save O
                load_kv_cache=n_blocks_r * (seqlen) * head_size * batchsize * num_key_value_heads * kv_byte * 2,
                store_kv_cache=0,
            )
        else:
            name = f"qk_matmul"
            self._analyze_to_results(
                "prefill",
                name,
                OPs=qk_matmul_OPs,
                load_weight=0,
                load_act=seqlen * head_size * batchsize * num_key_value_heads * a_byte,
                store_act=seqlen * seqlen * batchsize * num_attention_heads * a_byte,
                load_kv_cache=seqlen * head_size * batchsize * num_key_value_heads * kv_byte,
                store_kv_cache=0,
            )
            name = f"sv_matmul"
            self._analyze_to_results(
                "prefill",
                name,
                OPs=sv_matmul_OPs,
                load_weight=0,
                load_act=seqlen * seqlen * batchsize * num_attention_heads * a_byte,
                store_act=seqlen * head_size * batchsize * num_attention_heads * a_byte,
                load_kv_cache=seqlen * head_size * batchsize * num_key_value_heads * kv_byte,
                store_kv_cache=0,
            )
            name = f"softmax"
            self._analyze_to_results(
                "prefill",
                name,
                OPs=softmax_OPs,
                load_weight=0,
                load_act=batchsize * num_attention_heads * seqlen * seqlen * a_byte,
                store_act=batchsize * num_attention_heads * seqlen * seqlen * a_byte,
                load_kv_cache=0,
                store_kv_cache=0,
            )
        # Use elementwise layer specs for prefill (same as decode but with seqlen)
        for name, layer_spec in elementwise_layers.items():
            layer_type = layer_spec.get("type", "unknown")
            formula = layer_formulas.get(layer_type, {})
            ops_per_element = formula.get("OPs_per_element", 1)
            
            OPs = batchsize * hidden_size * seqlen * ops_per_element
            load_act = batchsize * hidden_size * seqlen * a_byte
            if layer_type == "activation":
                load_act = batchsize * hidden_size * seqlen * a_byte * 2
            store_act = batchsize * hidden_size * seqlen * a_byte
            
            self._analyze_to_results(
                "prefill",
                name,
                OPs=OPs,
                load_weight=0,
                load_act=load_act,
                store_act=store_act,
                load_kv_cache=0,
                store_kv_cache=0,
            )

        # compute total
        total_results = {"decode": {}, "prefill": {}}
        for data_name in ALL_DATA_NAMES:
            total_results["decode"][data_name] = 0
            total_results["prefill"][data_name] = 0
        for stage in ["decode", "prefill"]:
            for layer_name, result in self.results[stage].items():
                for data_name in ALL_DATA_NAMES:
                    total_results[stage][data_name] += result[data_name] * num_hidden_layers

        # memory footprint
        weight_kv_footprint = total_results["prefill"]["load_weight"] + total_results["prefill"]["store_kv_cache"]
        decode_tmp_act = 0
        for layer_name, result in self.results["decode"].items():
            decode_tmp_act += result["store_act"]
        total_results["decode"]["memory_consumption"] = decode_tmp_act + weight_kv_footprint
        total_results["decode"]["memory_consumption_tmp_act"] = decode_tmp_act
        total_results["decode"]["memory_consumption_weight"] = total_results["prefill"]["load_weight"]
        total_results["decode"]["memory_consumption_kv_cache"] = total_results["prefill"]["store_kv_cache"]
        prefill_tmp_act = 0
        for layer_name, result in self.results["prefill"].items():
            prefill_tmp_act += result["store_act"]
        total_results["prefill"]["memory_consumption"] = prefill_tmp_act + weight_kv_footprint
        total_results["prefill"]["memory_consumption_tmp_act"] = prefill_tmp_act
        total_results["prefill"]["memory_consumption_weight"] = total_results["prefill"]["load_weight"]
        total_results["prefill"]["memory_consumption_kv_cache"] = total_results["prefill"]["store_kv_cache"]

        # lm_head
        name = "lm_head"
        args = {"batchsize": batchsize, "a_byte": a_byte, "w_byte": w_byte}
        for layer_info in self.config.post_process(self.model_params, args):
            self._analyze_to_results(**layer_info)
            for data_name in ALL_DATA_NAMES:
                total_results[layer_info["stage"]][data_name] += self.results[layer_info["stage"]][layer_info["name"]][
                    data_name
                ]
        # for stage in ["prefill", "decode"]:
        #     self._analyze_to_results(
        #         stage,
        #         name,
        #         OPs=batchsize * hidden_size * vocab_size * 1,
        #         load_weight=hidden_size * vocab_size,
        #         load_act=hidden_size * a_byte,
        #         store_act=vocab_size * a_byte,
        #         load_kv_cache=0,
        #         store_kv_cache=0,
        #     )
        #     for data_name in ALL_DATA_NAMES:
        #         total_results[stage][data_name] += self.results[stage][name][data_name]

        self.results["total_results"] = total_results
        return self.results

    def analyze_generate_task(
        self,
        prompt_len,
        gen_len,
        batchsize,
        w_bit=16,
        a_bit=16,
        kv_bit=None,
        use_flashattention = False,
        tp_size: int = 1
    ):
        prefill_result = self.analyze(
            prompt_len,
            batchsize,
            w_bit,
            a_bit,
            kv_bit,
            use_flashattention=use_flashattention,
            tp_size=tp_size
        )
        prefill_time = inference_time = prefill_result["total_results"]["prefill"]["inference_time"]

        for i in range(prompt_len, prompt_len + gen_len):
            result = self.analyze(i, batchsize, w_bit, a_bit, kv_bit, use_flashattention=use_flashattention, tp_size=tp_size)
            inference_time += result["total_results"]["decode"]["inference_time"]
        return {"inference_time": inference_time, "prefill_time": prefill_time}

    def analyze_dynamic(
        self,
        topology,
        seqlen,
        batchsize,
        w_bit=16,
        a_bit=16,
        kv_bit=None,
        use_flashattention=False,
        kv_token_ratio=1,
        tp_size: int = 1,
    ):
        assert seqlen > 0
        assert batchsize > 0
        self.results = {"decode": {}, "prefill": {}}
        if kv_bit is None:
            kv_bit = a_bit
        self.w_bit = w_bit
        self.a_bit = a_bit
        self.kv_bit = kv_bit
        self.batchsize = batchsize
        self.seqlen = seqlen
        self.tp_size = tp_size

        context = self._build_dynamic_context(batchsize, seqlen, w_bit, a_bit, kv_bit, use_flashattention)
        for stage in ["decode", "prefill"]:
            stage_factor = 1 if stage == "decode" else seqlen
            for node in topology["nodes"]:
                info = node.get("info", {})
                if info.get("node_type") != "module":
                    continue
                metrics = self._compute_dynamic_node_metrics(info, context, stage_factor)
                self._store_dynamic_metrics(stage, node["id"], metrics)

        self._apply_folded_ops(topology, context)
        self.results["total_results"] = self._compute_dynamic_totals(topology["nodes"])
        return self.results

    def get_hardware_info(self):
        bandwidth = hardware_params[self.hardware]["bandwidth"]
        if self.w_bit <= 8 and self.a_bit <= 8 and self.kv_bit <= 8:
            max_OPS = hardware_params[self.hardware]["INT8"]
        else:
            max_OPS = hardware_params[self.hardware]["FP16"]
        onchip_buffer = hardware_params[self.hardware]["onchip_buffer"]
        return bandwidth, max_OPS, onchip_buffer

    def get_model_info(self):
        if self.config.get_num_attention_heads(self.model_params) != self.config.get_num_key_value_heads(
            self.model_params
        ):
            GQA = True
        else:
            GQA = False

        info = {"GQA": GQA}  # group query attention
        return info

    def _build_dynamic_context(self, batchsize, seqlen, w_bit, a_bit, kv_bit, use_flashattention):
        hidden_size = self.config.get_hidden_size(self.model_params)
        intermediate_size = self.config.get_intermediate_size(self.model_params)
        num_attention_heads = self.config.get_num_attention_heads(self.model_params)
        num_key_value_heads = self.config.get_num_key_value_heads(self.model_params)
        head_size = hidden_size // num_attention_heads if hidden_size and num_attention_heads else 0
        return {
            "batchsize": batchsize,
            "seqlen": seqlen,
            "hidden_size": hidden_size,
            "intermediate_size": intermediate_size,
            "num_attention_heads": num_attention_heads,
            "num_key_value_heads": num_key_value_heads,
            "head_size": head_size,
            "w_byte": w_bit / 8,
            "a_byte": a_bit / 8,
            "kv_byte": kv_bit / 8,
            "use_flashattention": use_flashattention,
        }

    def _compute_dynamic_node_metrics(self, info, context, stage_factor):
        module_type = str(info.get("module_type", "")).lower()
        metric_key = info.get("metric_key")
        batchsize = context["batchsize"]
        a_byte = context["a_byte"]
        w_byte = context["w_byte"]
        kv_byte = context["kv_byte"]
        seq_multiplier = batchsize * stage_factor

        if "embedding" in module_type:
            width = info.get("embedding_dim") or context["hidden_size"] or 0
            weight_rows = info.get("num_embeddings") or 0
            return self._build_metrics_dict(
                OPs=0,
                load_weight=seq_multiplier * width * w_byte,
                load_act=0,
                store_act=seq_multiplier * width * a_byte,
                stats_source="dynamic_topology",
            )

        if "linear" in module_type or metric_key == "lm_head":
            ic = info.get("in_features")
            oc = info.get("out_features")
            if (ic is None or oc is None) and len(info.get("weight_shape", [])) >= 2:
                oc, ic = info["weight_shape"][:2]
            if ic is None or oc is None:
                return self._build_metrics_dict(stats_source="dynamic_topology")
            is_kv_proj = metric_key in {"k_proj", "v_proj"}
            return self._build_metrics_dict(
                OPs=ic * oc * seq_multiplier * 2,
                load_weight=ic * oc * w_byte,
                load_act=ic * seq_multiplier * a_byte,
                store_act=0 if is_kv_proj else oc * seq_multiplier * a_byte,
                load_kv_cache=0,
                store_kv_cache=(oc * seq_multiplier * kv_byte) if is_kv_proj else 0,
                stats_source="dynamic_topology",
            )

        if "norm" in module_type:
            width = self._infer_feature_width(info, context)
            return self._build_metrics_dict(
                OPs=seq_multiplier * width * 7,
                load_weight=0,
                load_act=seq_multiplier * width * a_byte,
                store_act=seq_multiplier * width * a_byte,
                stats_source="dynamic_topology",
            )

        if module_type in {"gelu", "relu", "silu", "sigmoid", "tanh", "leakyrelu"}:
            width = self._infer_activation_width(info, context)
            return self._build_metrics_dict(
                OPs=seq_multiplier * width * 2,
                load_weight=0,
                load_act=seq_multiplier * width * a_byte * 2,
                store_act=seq_multiplier * width * a_byte,
                stats_source="dynamic_topology",
            )

        return self._build_metrics_dict(stats_source="dynamic_topology")

    def _infer_feature_width(self, info, context):
        normalized_shape = info.get("normalized_shape")
        if isinstance(normalized_shape, list) and normalized_shape:
            return normalized_shape[-1]
        if isinstance(normalized_shape, tuple) and normalized_shape:
            return normalized_shape[-1]
        if isinstance(normalized_shape, int):
            return normalized_shape
        if info.get("out_features"):
            return info["out_features"]
        if info.get("embedding_dim"):
            return info["embedding_dim"]
        return context["hidden_size"] or 0

    def _infer_activation_width(self, info, context):
        return info.get("out_features") or context["intermediate_size"] or context["hidden_size"] or 0

    def _build_metrics_dict(
        self,
        OPs=0,
        load_weight=0,
        load_act=0,
        store_act=0,
        load_kv_cache=0,
        store_kv_cache=0,
        folded_ops=0,
        folded_ops_detail="",
        stats_source="dynamic_topology",
    ):
        bandwidth, max_OPS, onchip_buffer = self.get_hardware_info()
        memory_access = load_weight + load_act + store_act + load_kv_cache + store_kv_cache
        if OPs == 0 and memory_access == 0:
            arithmetic_intensity = 0
            performance = 0
            bound = "none"
            inference_time = 0
        else:
            arithmetic_intensity, performance, bound = roofline_analyze(bandwidth, max_OPS, OPs, memory_access)
            inference_time = OPs / performance if performance else 0
        return {
            "OPs": OPs,
            "memory_access": memory_access,
            "arithmetic_intensity": arithmetic_intensity,
            "performance": performance,
            "bound": bound,
            "load_weight": load_weight,
            "load_act": load_act,
            "store_act": store_act,
            "load_kv_cache": load_kv_cache,
            "store_kv_cache": store_kv_cache,
            "inference_time": inference_time,
            "folded_ops": folded_ops,
            "folded_ops_detail": folded_ops_detail,
            "stats_source": stats_source,
        }

    def _store_dynamic_metrics(self, stage, node_id, metrics):
        self.results[stage][node_id] = metrics

    def _apply_folded_ops(self, topology, context):
        role_to_node_id = {}
        for node in topology["nodes"]:
            metric_key = node.get("info", {}).get("metric_key")
            if metric_key:
                role_to_node_id[metric_key] = node["id"]

        attention_target = role_to_node_id.get("out_proj") or role_to_node_id.get("v_proj")
        attn_residual_target = role_to_node_id.get("mlp_norm") or role_to_node_id.get("gate_proj") or role_to_node_id.get("up_proj")
        mlp_residual_target = role_to_node_id.get("down_proj") or role_to_node_id.get("mlp_norm")
        if not attention_target and not attn_residual_target and not mlp_residual_target:
            return

        for stage in ["decode", "prefill"]:
            stage_factor = 1 if stage == "decode" else context["seqlen"]
            batchsize = context["batchsize"]
            num_attention_heads = context["num_attention_heads"] or 0
            num_key_value_heads = context["num_key_value_heads"] or num_attention_heads or 0
            head_size = context["head_size"] or 0
            seqlen = context["seqlen"]
            a_byte = context["a_byte"]
            kv_byte = context["kv_byte"]
            hidden_size = context["hidden_size"] or 0

            if context["use_flashattention"]:
                q_numel = stage_factor * head_size * batchsize * num_attention_heads * a_byte
                o_numel = stage_factor * seqlen * batchsize * num_attention_heads * a_byte
                self._merge_folded_metrics(
                    stage,
                    attention_target,
                    "fused_attention",
                    OPs=self._attention_qk_ops(stage, context) + self._attention_sv_ops(stage, context) + self._attention_softmax_ops(stage, context),
                    load_act=q_numel,
                    store_act=o_numel * 2,
                    load_kv_cache=stage_factor * seqlen * head_size * batchsize * num_key_value_heads * kv_byte * 2,
                )
            else:
                self._merge_folded_metrics(
                    stage,
                    attention_target,
                    "qk_matmul",
                    OPs=self._attention_qk_ops(stage, context),
                    load_act=stage_factor * head_size * batchsize * num_attention_heads * a_byte,
                    store_act=stage_factor * seqlen * batchsize * num_attention_heads * a_byte,
                    load_kv_cache=seqlen * head_size * batchsize * num_key_value_heads * kv_byte,
                )
                self._merge_folded_metrics(
                    stage,
                    attention_target,
                    "softmax",
                    OPs=self._attention_softmax_ops(stage, context),
                    load_act=batchsize * num_attention_heads * stage_factor * seqlen * a_byte,
                    store_act=batchsize * num_attention_heads * stage_factor * seqlen * a_byte,
                )
                self._merge_folded_metrics(
                    stage,
                    attention_target,
                    "sv_matmul",
                    OPs=self._attention_sv_ops(stage, context),
                    load_act=stage_factor * seqlen * batchsize * num_attention_heads * a_byte,
                    store_act=stage_factor * head_size * batchsize * num_attention_heads * a_byte,
                    load_kv_cache=seqlen * head_size * batchsize * num_key_value_heads * kv_byte,
                )

            self._merge_folded_metrics(
                stage,
                attn_residual_target,
                "attn_add",
                OPs=batchsize * hidden_size * stage_factor,
                load_act=batchsize * hidden_size * stage_factor * a_byte,
                store_act=batchsize * hidden_size * stage_factor * a_byte,
            )
            self._merge_folded_metrics(
                stage,
                mlp_residual_target,
                "mlp_add",
                OPs=batchsize * hidden_size * stage_factor,
                load_act=batchsize * hidden_size * stage_factor * a_byte,
                store_act=batchsize * hidden_size * stage_factor * a_byte,
            )

    def _merge_folded_metrics(
        self,
        stage,
        node_id,
        detail,
        OPs=0,
        load_weight=0,
        load_act=0,
        store_act=0,
        load_kv_cache=0,
        store_kv_cache=0,
    ):
        if not node_id or node_id not in self.results[stage]:
            return
        current = self.results[stage][node_id]
        updated = self._build_metrics_dict(
            OPs=current["OPs"] + OPs,
            load_weight=current["load_weight"] + load_weight,
            load_act=current["load_act"] + load_act,
            store_act=current["store_act"] + store_act,
            load_kv_cache=current["load_kv_cache"] + load_kv_cache,
            store_kv_cache=current["store_kv_cache"] + store_kv_cache,
            folded_ops=current.get("folded_ops", 0) + OPs,
            folded_ops_detail=self._append_folded_detail(current.get("folded_ops_detail", ""), detail),
            stats_source=current.get("stats_source", "dynamic_topology"),
        )
        self.results[stage][node_id] = updated

    def _append_folded_detail(self, current, detail):
        if not current:
            return detail
        details = set(current.split(","))
        details.add(detail)
        return ",".join(sorted(detail_name for detail_name in details if detail_name))

    def _attention_qk_ops(self, stage, context):
        seqlen = context["seqlen"]
        head_size = context["head_size"]
        num_attention_heads = context["num_attention_heads"]
        batchsize = context["batchsize"]
        if stage == "decode":
            return seqlen * head_size * num_attention_heads * batchsize * 2
        return seqlen * seqlen * head_size * num_attention_heads * batchsize * 2

    def _attention_sv_ops(self, stage, context):
        seqlen = context["seqlen"]
        head_size = context["head_size"]
        num_attention_heads = context["num_attention_heads"]
        batchsize = context["batchsize"]
        if stage == "decode":
            return head_size * seqlen * num_attention_heads * batchsize * 2
        return seqlen * head_size * seqlen * num_attention_heads * batchsize * 2

    def _attention_softmax_ops(self, stage, context):
        seqlen = context["seqlen"]
        num_attention_heads = context["num_attention_heads"]
        batchsize = context["batchsize"]
        if stage == "decode":
            return batchsize * num_attention_heads * seqlen * 5
        return batchsize * num_attention_heads * seqlen * seqlen * 5

    def _compute_dynamic_totals(self, topology_nodes):
        total_results = {"decode": {}, "prefill": {}}
        scaled_store_act = {"decode": 0, "prefill": 0}
        for stage in ["decode", "prefill"]:
            for data_name in ALL_DATA_NAMES:
                total_results[stage][data_name] = 0
            for node in topology_nodes:
                info = node.get("info", {})
                repeat_factor = info.get("repeat_factor", 1)
                node_result = self.results[stage].get(node["id"])
                if not node_result:
                    continue
                for data_name in ALL_DATA_NAMES:
                    total_results[stage][data_name] += node_result.get(data_name, 0) * repeat_factor
                scaled_store_act[stage] += node_result.get("store_act", 0) * repeat_factor
                node_result["repeat_factor"] = repeat_factor
                for data_name in ALL_DATA_NAMES:
                    node_result[f"total_contribution_{data_name}"] = node_result.get(data_name, 0) * repeat_factor

        weight_kv_footprint = total_results["prefill"]["load_weight"] + total_results["prefill"]["store_kv_cache"]
        decode_tmp_act = scaled_store_act["decode"]
        prefill_tmp_act = scaled_store_act["prefill"]
        total_results["decode"]["memory_consumption"] = decode_tmp_act + weight_kv_footprint
        total_results["decode"]["memory_consumption_tmp_act"] = decode_tmp_act
        total_results["decode"]["memory_consumption_weight"] = total_results["prefill"]["load_weight"]
        total_results["decode"]["memory_consumption_kv_cache"] = total_results["prefill"]["store_kv_cache"]
        total_results["prefill"]["memory_consumption"] = prefill_tmp_act + weight_kv_footprint
        total_results["prefill"]["memory_consumption_tmp_act"] = prefill_tmp_act
        total_results["prefill"]["memory_consumption_weight"] = total_results["prefill"]["load_weight"]
        total_results["prefill"]["memory_consumption_kv_cache"] = total_results["prefill"]["store_kv_cache"]
        return total_results
