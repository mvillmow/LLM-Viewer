<template>
    <div class="left_control">
        <Config></Config>
    </div>
</template>

<script setup>
import Config from "./left_controls/Config.vue"
import { inject, ref, watch, computed, onMounted } from 'vue';

const model_id = inject('model_id');

onMounted(() => {
    console.log("LeftControl onMounted", model_id.value)
})

</script>

<style>
.input_config {
    border: 1px solid var(--border);
    margin: 3px;
}

.left_control {
    align-items: left;
    width: 21%;
    min-width: 280px;
    height: 100%;
    background: linear-gradient(180deg, var(--panel-elevated-strong) 0%, var(--panel-elevated-soft) 100%);
    border: 1px solid var(--border);
    border-radius: 18px;
    box-shadow: var(--shadow-soft);
    overflow-y: auto;
    box-sizing: border-box;
    padding: 18px 18px 22px;
}
</style>
