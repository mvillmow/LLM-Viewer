<script setup>
// import { defineAsyncComponent } from 'vue'

// const Graph = defineAsyncComponent(() =>
//   import('./components/Graph.vue')
// )

import Graph from "./components/Graph.vue"
import LeftPannel from "./components/LeftPannel.vue"
import Header from "./components/Header.vue"
import { ref, computed, provide } from 'vue';

const model_id = ref("LLM360/K2-Think-V2");
const hardware = ref("nvidia_A6000");
const global_update_trigger = ref(1);
const total_results = ref({});
const ip_port = ref("127.0.0.1:5000");

provide("model_id", model_id);
provide("hardware", hardware);
provide("global_update_trigger", global_update_trigger);
provide("total_results", total_results);
provide("ip_port", ip_port);


const global_inference_config = ref({ 
  "stage": "decode", 
  batch_size: 1, 
  seq_length: 1024, 
  gen_length: 1,
  tp_size: 1,
  w_quant: "FP16", 
  a_quant: "FP16", 
  kv_quant: "FP16", 
  use_flashattention: false
});
provide("global_inference_config", global_inference_config);

</script>

<template>
  <div class="app_container">
    <div class="upper_header">
      <Header></Header>
    </div>
    <div class="bottom-block">
      <LeftPannel></LeftPannel>
      <Graph></Graph>
    </div>

  </div>
</template>

<style>
html,
body,
#app {
  margin: 0;
  width: 100%;
  height: 100%;
}

:root {
  --background: #f6f1e8;
  --background-top: #f9f4eb;
  --background-bottom: #f4ede3;
  --background-accent: rgba(47, 111, 109, 0.1);
  --background-accent-soft: rgba(47, 111, 109, 0.08);
  --panel: #fbf7f1;
  --panel-elevated: rgba(255, 252, 247, 0.94);
  --panel-elevated-strong: rgba(255, 252, 247, 0.96);
  --panel-elevated-soft: rgba(248, 242, 234, 0.92);
  --panel-input: rgba(255, 252, 247, 0.88);
  --panel-overlay: rgba(255, 252, 247, 0.9);
  --surface-hover: #f5eee4;
  --surface-active: #efe4d4;
  --border: #d8cbbb;
  --text: #2f261d;
  --text-muted: #6d6156;
  --accent: #2f6f6d;
  --accent-hover: #265c5a;
  --accent-active: #1f4a48;
  --accent-strong: #3d8481;
  --accent-strong-hover: #337472;
  --accent-strong-active: #255654;
  --accent-contrast: #f8f4ed;
  --accent-border-soft: rgba(47, 111, 109, 0.35);
  --accent-border-strong: rgba(24, 64, 62, 0.18);
  --accent-shadow: rgba(47, 111, 109, 0.18);
  --accent-underline: rgba(47, 111, 109, 0.28);
  --accent-tint: rgba(47, 111, 109, 0.1);
  --focus-ring: rgba(47, 111, 109, 0.24);
  --shadow-soft: 0 18px 40px rgba(65, 47, 31, 0.08);
  --header-shadow: rgba(88, 67, 48, 0.05);
  --graph-node-linear: #c16a4a;
  --graph-node-nonlinear: #58806d;
  --graph-node-title: var(--accent-contrast);
  --graph-node-label-muted: rgba(47, 38, 29, 0.55);
  --graph-node-value: #5f564d;
  --graph-arrow: #97897b;
  --graph-edge: #5f564d;
  --graph-edge-residual: #5f8fbd;
  --graph-label: #2f261d;
  --graph-label-stroke: #ece1d3;
  --graph-combo-fill: #f6efe6;
  --graph-combo-stroke: #cdbdab;
  --graph-combo-shadow: rgba(148, 163, 184, 0.32);
  --graph-selection: #d8ebe5;
  --chart-line: #3f352c;
}

body {
  overflow-x: hidden;
  overflow-y: hidden;
  background: linear-gradient(180deg, var(--background-top) 0%, var(--background-bottom) 100%);
  color: var(--text);
  font-family: "Avenir Next", "Segoe UI", sans-serif;
}

.app_container {
  width: 100%;
  height: 100vh;
  background:
    radial-gradient(circle at top left, var(--background-accent), transparent 32%),
    linear-gradient(180deg, var(--background-top) 0%, var(--background-bottom) 100%);
}

.upper_header {
  flex: 1;
  display: flex;
  flex-direction: row;
  align-items: center;
  justify-content: center;
  width: 100%;
  height: 50px;
  background-color: var(--panel);
  border-bottom: 1px solid var(--border);
  box-shadow: 0 8px 20px var(--header-shadow);
}

.bottom-block {
  display: flex;
  flex-direction: row;
  height: calc(100% - 60px);
  gap: 16px;
  padding: 16px;
  box-sizing: border-box;
}
</style>
