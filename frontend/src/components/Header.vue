<template>
    <div class="title">
        <a href="https://github.com/mvillmow/LLM-Viewer" target="_blank" class="hover-bold">LLM-Viewer</a>
        v{{ version }}
    </div>
    <div class="header_button">
        |
        <span>Model: </span>
        <input 
            type="text" 
            v-model="select_model_id" 
            list="model_suggestions" 
            placeholder="Enter HuggingFace model ID..."
            class="model-input"
            @keyup.enter="loadModel"
        />
        <datalist id="model_suggestions">
            <option v-for="model_id in suggested_model_ids" :value="model_id">{{ model_id }}</option>
        </datalist>
        <button @click="loadModel" class="load-btn">Load</button>
        <span> | </span>
        <span>Hardware: </span>
        <select v-model="select_hardware">
            <option v-for="hardware in avaliable_hardwares" :value="hardware">{{ hardware }}</option>
        </select>
    </div>
    <div>
        <span> | </span>
        <span>Server: </span>
        <select v-model="ip_port">
            <option value="api.llm-viewer.com">api.llm-viewer.com</option>
            <option value="127.0.0.1:5000">127.0.0.1</option>
        </select>
    </div>
    <div>
        <span> | </span>
        <span class="hover-bold" @click="is_show_help = ! is_show_help">Help</span>
    </div>
    <div>
        <span> | </span>
        <a href="https://github.com/mvillmow/LLM-Viewer" target="_blank" class="hover-bold">Github Project</a>
    </div>
    <div>
        <span> | </span>
        <a href="https://arxiv.org/pdf/2402.16363.pdf" target="_blank" class="hover-bold">Paper</a>
    </div>
    <div v-if="is_show_help" class="float-info-window">
        <!-- item -->
        <p>LLM-Viewer is a open-sourced tool to visualize the LLM model and analyze the deployment on hardware devices.</p>
        <p>
            At the center of the page, you can see the graph of the LLM model. Click the node to see the detail of the node.
        </p>
        <p>↑ At the top of the page, you can set the LLM model, hardware devices, and server.
            If you deploy the LLM-Viewer localhost, you can select the localhost server.
        </p>
        <p>
            ← At the left of the page, you can see the configuration pannel. You can set the inference config and optimization config.
        </p>
        <p>
            ↙ The Network-wise Analysis result is demonstrated in the left pannel.
        </p>
        <p>
            We invite you to read our paper <a class="hover-bold" href="https://arxiv.org/pdf/2402.16363.pdf" target="_blank">LLM Inference Unveiled: Survey and Roofline Model Insights</a>.
            In this paper, we provide a comprehensive analysis of the latest advancements in efficient LLM inference using LLM-Viewer. 
            Citation bibtext:
        </p>
        @article{yuan2024llm,<br/>
            &nbsp    title={LLM Inference Unveiled: Survey and Roofline Model Insights},<br/>
            &nbsp    author={Yuan, Zhihang and Shang, Yuzhang and Zhou, Yang and Dong, Zhen and Xue, Chenhao and Wu, Bingzhe and Li, Zhikai and Gu, Qingyi and Lee, Yong Jae and Yan, Yan and others},<br/>
            &nbsp    journal={arXiv preprint arXiv:2402.16363},<br/>
            &nbsp    year={2024}<br/>
        }
    </div>
</template>

<script setup>
import { inject, ref, watch, computed, onMounted } from 'vue';
import axios from 'axios'
const model_id = inject('model_id');
const hardware = inject('hardware');
const global_update_trigger = inject('global_update_trigger');
const ip_port = inject('ip_port');

const avaliable_hardwares = ref([]);
const suggested_model_ids = ref([]);

const version = ref(llm_viewer_frontend_version)

const is_show_help = ref(false)
const is_loading_model = ref(false)

function update_avaliable() {
    const url = 'http://' + ip_port.value + '/get_avaliable'
    axios.get(url).then(function (response) {
        console.log(response);
        avaliable_hardwares.value = response.data.avaliable_hardwares
        suggested_model_ids.value = response.data.suggested_model_ids
        // Set default model to first suggested model if not already set
        if (suggested_model_ids.value.length > 0 && !select_model_id.value) {
            select_model_id.value = suggested_model_ids.value[0]
        }
    })
        .catch(function (error) {
            console.log("error in get_avaliable");
            console.log(error);
        });
}

onMounted(() => {
    console.log("Header mounted")
    update_avaliable()
})

var select_model_id = ref(model_id.value);

function loadModel() {
    if (!select_model_id.value || select_model_id.value.trim() === '') {
        return;
    }
    console.log("Loading model:", select_model_id.value)
    model_id.value = select_model_id.value
    is_loading_model.value = true
    global_update_trigger.value += 1
    // Reset loading state after a delay (the graph component will handle the actual loading state)
    setTimeout(() => {
        is_loading_model.value = false
    }, 3000)
}

watch(select_model_id, (n) => {
    console.log("select_model_id", n)
})

var select_hardware = ref(hardware.value);
watch(select_hardware, (n) => {
    console.log("select_hardware", n)
    hardware.value = n
    global_update_trigger.value += 1
})

watch(ip_port, (n) => {
    console.log("ip_port", n)
    update_avaliable()
})


</script>

<style scoped>
.header_button,
.title,
div {
    color: var(--text);
}

.header_button {
    display: flex;
    align-items: center;
    gap: 6px;
}

.header_button button,
.model-input,
select {
    font-family: inherit;
}

.header_button button {
    font-size: 1.0rem;
    margin: 5px;
    padding: 7px 12px;
    border-radius: 10px;
    border: 1px solid var(--border);
    background-color: var(--panel);
    color: var(--text);
    cursor: pointer;
    transition: border-color 0.2s ease, background-color 0.2s ease, color 0.2s ease, transform 0.2s ease;
}

.header_button button:hover {
    color: var(--accent);
    border-color: var(--accent-border-soft);
    background-color: var(--surface-hover);
    transform: translateY(-1px);
}

.header_button button:active {
    color: var(--accent-active);
    background-color: var(--surface-active);
}

.model-input {
    padding: 8px 12px;
    border: 1px solid var(--border);
    border-radius: 10px;
    font-size: 14px;
    width: 280px;
    margin-right: 5px;
    background-color: var(--panel-input);
    color: var(--text);
}

.model-input:focus,
select:focus,
.load-btn:focus {
    outline: none;
    border-color: var(--accent);
    box-shadow: 0 0 0 4px var(--focus-ring);
}

.load-btn {
    padding: 8px 16px;
    background: linear-gradient(180deg, var(--accent-strong) 0%, var(--accent) 100%);
    color: var(--accent-contrast);
    border: 1px solid var(--accent-border-strong);
    border-radius: 10px;
    cursor: pointer;
    font-size: 14px;
    transition: background-color 0.2s ease, transform 0.2s ease, box-shadow 0.2s ease;
    box-shadow: 0 10px 18px var(--accent-shadow);
}

.load-btn:hover {
    background: linear-gradient(180deg, var(--accent-strong-hover) 0%, var(--accent-hover) 100%);
    transform: translateY(-1px);
}

.load-btn:active {
    background: linear-gradient(180deg, var(--accent-strong-active) 0%, var(--accent-active) 100%);
}

.active {
    color: var(--accent-contrast);
    background-color: var(--accent-active);
}

.title {
    font-size: 18px;
    text-align: left;
    font-weight: 700;
    letter-spacing: 0.01em;
}

.hover-bold{
    color: inherit;
    text-decoration-color: var(--accent-underline);
    text-underline-offset: 0.15em;
}

.hover-bold:hover {
    color: var(--accent);
}

.float-info-window {
    position: absolute;
    top: 80px;
    left: 40%;
    height: auto;
    width: 30%;
    background: var(--panel-elevated);
    padding: 20px;
    border: 1px solid var(--border);
    border-radius: 16px;
    color: var(--text);
    box-shadow: var(--shadow-soft);
    backdrop-filter: blur(12px);
    z-index: 999;
}

select {
    padding: 8px 12px;
    border: 1px solid var(--border);
    border-radius: 10px;
    background-color: var(--panel-input);
}
</style>

