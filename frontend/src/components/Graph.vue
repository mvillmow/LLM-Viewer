<template>
    <div class="main_graph" ref="graphContainer">
        <div id="info-window" class="float-info-window" v-if="info_window_str.length>0">
            <h3> {{ info_window_str }}</h3>
        </div>
        <div id="graphContainer" @resize="handleResize"></div>

        <div class="float-search-window">
            <input type="text" v-model.lazy="searchText" placeholder="Search" />
            <div>
                <div v-for="(value) in searchResult" :key="value" @click="SelectNode(value, true)">
                    {{ value }}
                </div>
            </div>
        </div>
        <div class="float-node-info-window">
            <div v-if="selected_node_id" class="float-node-info-item">
                <strong>{{ selected_node_id }}</strong>
            </div>
            <div v-for="(value, key) in all_node_info[selected_node_id]" :key="key" class="float-node-info-item">
                <span v-if="typeof value === 'string'">{{ key }}: {{ value }}</span>
                <span v-else-if="['bound'].includes(key)">{{ key }}: {{ value }}</span>
                <span v-else-if="['inference_time'].includes(key)">{{ key }}: {{ strNumberTime(value) }}</span>
                <span v-else>{{ key }}: {{ strNumber(value) }}</span>
            </div>
            <div class="float-node-info-item">
                <canvas id="lineChart" width="300" height="200"></canvas>
            </div>
        </div>
    </div>
</template>

<script setup>
import G6 from "@antv/g6"

import { onMounted } from 'vue'
import { watch, inject, ref } from 'vue'
import { getGraphConfig, getGraphTheme } from "./graphs/graph_config.js"
// import { get_roofline_options } from "./graphs/roofline_config.js"
import axios from 'axios'
import { strNumber, strNumberTime } from '@/utils.js';
import { Chart, registerables } from 'chart.js';

import annotationPlugin from 'chartjs-plugin-annotation';

const model_id = inject('model_id')
const hardware = inject('hardware')
const global_update_trigger = inject('global_update_trigger')
const global_inference_config = inject('global_inference_config')
const ip_port = inject('ip_port')
const total_results = inject('total_results')
var hardware_info = {}
var nowFocusNode = null
var nowFocusNodePrevColor = null


var graph = null;
var graph_data;
const all_node_info = ref({})
Chart.register(...registerables, annotationPlugin);

const searchText = ref('')
const searchResult = ref([])

const selected_node_id = ref("")
var roofline_chart = null

const info_window_str = ref('')
let pendingRefitHandle = null

const changeGraphSizeWaitTimer = ref(false);
window.onresize = () => {
    if (!changeGraphSizeWaitTimer.value & graph != null) {
        // console.log("handleResize", window.innerWidth, window.innerHeight)
        var leftControlDiv = document.querySelector('.left_control');
        var width = leftControlDiv.offsetWidth;
        graph.changeSize(window.innerWidth - width, window.innerHeight)
        changeGraphSizeWaitTimer.value = true;
        setTimeout(function () {
            changeGraphSizeWaitTimer.value = false;
        }, 100);
    }
};

function graphUpdate() {
    const url = 'http://' + ip_port.value + '/get_graph'
    console.log("graphUpdate", url)
    info_window_str.value="Loading from server..."
    var is_init=false
    axios.post(url, { model_id: model_id.value, hardware: hardware.value, inference_config: global_inference_config.value }).then(function (response) {
        console.log(response);
        info_window_str.value=""
        graph_data = normalizeGraphData(response.data)
        all_node_info.value = {}
        for (let i = 0; i < graph_data.nodes.length; i++) {
            all_node_info.value[graph_data.nodes[i].id] = graph_data.nodes[i].info;
        }
        total_results.value = graph_data.total_results
        hardware_info = graph_data.hardware_info

        nowFocusNode=null
        graph.clear()
        graph.data(graph_data)
        graph.render()
        scheduleGraphRefit(true)
        console.log(graph_data)
        setTimeout(() => {
            update_roofline_model();
        }, 10);

    })
        .catch(function (error) {
            info_window_str.value="Error in get_graph"
            console.log("error in graphUpdate");
            console.log(error);
        });

}

watch(() => global_update_trigger.value, () => graphUpdate(false))
// watch(() => global_update_trigger.value, () => update_roofline_model())
// watch(() => global_update_trigger.value, () => release_select())

function handleSearch(newText, oldText) {
    console.log("handleSearch", newText)
    const nodes = graph.findAll('node', (node) => {
        const nodeId = node.get('id');
        // console.log("handleSearch", node)
        return nodeId.includes(newText)
    });
    console.log("handleSearch", nodes)
    searchResult.value = []
    for (let i = 0; i < nodes.length; i++) {
        const node = nodes[i];
        const nodeId = node.get('id');
        searchResult.value.push(nodeId)
        if (i > 100) {
            break
        }

    }
}
watch(searchText, handleSearch)


function SelectNode(nodeId, moveView = false) {
    if (moveView) {
        console.log("graph.focusItem", nodeId)
        graph.focusItem(nodeId, true)
    }
    if (nowFocusNode) {
        // console.log("nowFocusNodePrevColor", nowFocusNodePrevColor)
        nowFocusNode.update({
            style: {
                fill: nowFocusNodePrevColor,
            },
        });
    }
    const node = graph.findById(nodeId)
    if (node) {
        const theme = getGraphTheme();
        // 高亮
        if (node.getModel().style.fill) {
            nowFocusNodePrevColor = node.getModel().style.fill
        } else {
            nowFocusNodePrevColor = theme.panel
        }
        node.update({
            style: {
                fill: theme.selection,
            },
        });
        nowFocusNode = node
    }

    selected_node_id.value = nodeId
}


function scheduleGraphRefit(runLayout = false) {
    if (!graph) {
        return;
    }
    if (pendingRefitHandle !== null) {
        cancelAnimationFrame(pendingRefitHandle);
    }
    pendingRefitHandle = requestAnimationFrame(() => {
        pendingRefitHandle = null;
        if (runLayout && graph.get && graph.get('layoutController')) {
            graph.layout();
        }
        setTimeout(() => {
            if (graph) {
                graph.fitView(20);
            }
        }, 40);
    });
}


function update_roofline_model() {
    const ctx = document.getElementById('lineChart');
    if (ctx) {
        const theme = getGraphTheme();
        if (roofline_chart) {
            roofline_chart.destroy();
        }
        const bandwidth = hardware_info["bandwidth"];
        const max_OPS = hardware_info["max_OPS"];
        const turningPoint = max_OPS / bandwidth;

        var annotation
        var x_max
        if (selected_node_id.value){
            const node_arithmetic_intensity = all_node_info.value[selected_node_id.value]["arithmetic_intensity"];
            x_max = Math.max(turningPoint * 3, node_arithmetic_intensity+1);
            annotation={
                        annotations: {
                            lineX: {
                                type: 'line',
                                xMin: node_arithmetic_intensity,
                                xMax: node_arithmetic_intensity,
                                yMin: 0,
                                yMax: max_OPS * 1.1,
                                borderColor: theme.accent,
                                borderWidth: 2,
                                borderDash: [5, 5], // 虚线样式
                                label: {
                                    enabled: true,
                                    content: 'Node AI',
                                    position: 'top',
                                    backgroundColor: theme.accent,
                                    color: theme.accentContrast
                                }
                            }
                        }
                    }
        }else{
            annotation={}
            x_max = turningPoint * 3
        }
        roofline_chart = new Chart(ctx, {
            type: 'line',
            data:
            {
                // labels: [0, turningPoint, 321],
                datasets: [{
                    label: 'Roofline',
                    data: [
                        { x: 0, y: 0 },
                        { x: turningPoint, y: max_OPS },
                        { x: x_max, y: max_OPS }
                    ],
                    // [0, max_OPS, max_OPS],
                    borderColor: theme.chartLine,
                    borderWidth: 2,
                    fill: false,
                    pointRadius: 0 // 不显示数据点
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,

                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Arithmetic Intensity (OPs/byte)',
                            color: theme.text
                        },
                        type: 'linear',
                        ticks: {
                            callback: function (value, index, values) {
                                return value.toFixed(1);
                            },
                            color: theme.textMuted,
                        },
                        beginAtZero: true,
                        max: x_max,
                        grid: {
                            color: theme.border,
                        },
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Performance (OPS)',
                            color: theme.text
                        },
                        ticks: {
                            callback: function (value, index, values) {
                                return value.toExponential(1);
                            },
                            color: theme.textMuted,
                        },
                        beginAtZero: true,
                        max: max_OPS * 1.1,
                        grid: {
                            color: theme.border,
                        },
                    }
                },
                plugins: {
                    title: {
                        display: true,
                        text: 'Roofline Model', // 这里是你想要的标题
                        color: theme.text,
                        position: 'top' // 标题的位置，可以是'top', 'left', 'bottom', 或 'right'
                    },
                    legend: {
                        display: false
                    },
                    annotation: annotation
                }
            }
        });
    }
}

function release_select(){
    if (nowFocusNode) {
        nowFocusNode.update({
            style: {
                fill: nowFocusNodePrevColor,
            },
        });
        nowFocusNode = null;
        nowFocusNodePrevColor = null;
    }
    selected_node_id.value = ""
    update_roofline_model()
}


function getComboAncestors(item) {
    const ancestors = [];
    let current = item;
    while (current) {
        const comboId = current.getModel()?.comboId;
        if (!comboId) {
            break;
        }
        const combo = graph.findById(comboId);
        if (!combo) {
            break;
        }
        ancestors.push(combo);
        current = combo;
    }
    return ancestors;
}


function isSelectedNodeHiddenByCollapsedCombo() {
    if (!selected_node_id.value || !graph) {
        return false;
    }
    const node = graph.findById(selected_node_id.value);
    if (!node) {
        return true;
    }
    const comboAncestors = getComboAncestors(node);
    return comboAncestors.some((combo) => combo.getModel()?.collapsed === true);
}


function handleComboCollapseExpand() {
    if (isSelectedNodeHiddenByCollapsedCombo()) {
        release_select();
    }
    scheduleGraphRefit(true);
}

function normalizeGraphData(serverGraphData) {
    const theme = getGraphTheme();
    const edgeStyles = {
        data_flow: {
            stroke: theme.edge,
            lineDash: null,
        },
        residual: {
            stroke: theme.edgeResidual,
            lineDash: [5, 5],
        },
    };

    const nodes = (serverGraphData.nodes || []).map((node) => ({
        ...node,
        comboId: node.comboId || undefined,
    }));
    const edges = (serverGraphData.edges || []).map((edge) => {
        const edgeType = edge.edgeType || 'data_flow';
        const style = edgeStyles[edgeType] || edgeStyles.data_flow;
        return {
            ...edge,
            edgeType,
            style: {
                ...(edge.style || {}),
                stroke: style.stroke,
                ...(style.lineDash ? { lineDash: style.lineDash } : {}),
            },
            labelCfg: edgeType === 'residual'
                ? { autoRotate: true, style: { fill: theme.edgeResidual, fontSize: 12 } }
                : undefined,
        };
    });
    const combos = (serverGraphData.combos || []).map((combo) => ({
        ...combo,
        collapsed: combo.collapsed === true,
        style: {
            ...(combo.style || {}),
        },
    }));
    return {
        ...serverGraphData,
        nodes,
        edges,
        combos,
    };
}

onMounted(() => {
    graph = new G6.Graph(getGraphConfig()); 
    graph.on('node:click', (event) => {
        const { item } = event;
        const node = item.getModel();
        clickNode(node);
    });
    graph.on('node:touchstart', (event) => {
        const { item } = event;
        const node = item.getModel();
        clickNode(node);
    });
    graph.on('canvas:click', (event) => {
        release_select()
    });
    graph.on('aftercollapseexpandcombo', () => {
        handleComboCollapseExpand();
    });
    graphUpdate(true);
})

function clickNode(node) {
    console.log(node);
    const nodeId = node.id;
    SelectNode(nodeId);
    // sleep 100ms
    setTimeout(() => {
        update_roofline_model();
    }, 100);
}
</script>

<style scoped>
.main_graph {
    width: 75%;
    height: 100%;

    position: relative;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    float: right;
    flex-grow: 1;
    background:
        radial-gradient(circle at top right, var(--background-accent-soft), transparent 26%),
        linear-gradient(180deg, var(--panel-elevated) 0%, var(--panel-elevated-soft) 100%);
    border: 1px solid var(--border);
    border-radius: 22px;
    box-shadow: var(--shadow-soft);
    overflow: hidden;
}

.float-search-window {
    position: absolute;
    top: 10px;
    right: 10px;
    height: auto;
    max-height: 50vh;
    min-width: 220px;
    background: var(--panel-elevated);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 10px;
    overflow-y: auto;
    box-shadow: var(--shadow-soft);
    backdrop-filter: blur(12px);
    color: var(--text);
}


.float-info-window {
    position: absolute;
    top: 10px;
    left: 40%;
    height: auto;
    width: 20%;
    background: var(--panel-elevated);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 10px 14px;
    overflow-y: auto;
    box-shadow: var(--shadow-soft);
    backdrop-filter: blur(12px);
    color: var(--text);
}

.float-node-info-window {
    position: absolute;
    top: 10px;
    left: 10px;
    min-width: 320px;
    background: var(--panel-elevated);
    border: 1px solid var(--border);
    border-radius: 16px;
    box-shadow: var(--shadow-soft);
    backdrop-filter: blur(12px);
    color: var(--text);
    overflow: hidden;
}

.float-node-info-item {
    padding: 8px 12px;
    border-top: 1px solid var(--border);
}

.float-node-info-item:first-child {
    border-top: 0;
    padding-top: 12px;
}

.float-search-window input {
    width: 100%;
    box-sizing: border-box;
    margin-bottom: 8px;
    padding: 8px 10px;
    border: 1px solid var(--border);
    border-radius: 10px;
    background: var(--panel-overlay);
    color: var(--text);
}

.float-search-window input:focus {
    outline: none;
    border-color: var(--accent);
    box-shadow: 0 0 0 4px var(--focus-ring);
}

.float-search-window > div > div {
    padding: 7px 9px;
    border-radius: 10px;
    cursor: pointer;
    color: var(--text);
}

.float-search-window > div > div:hover {
    background-color: var(--accent-tint);
    color: var(--accent-active);
}
</style>
