import G6 from "@antv/g6"
import { getThemeColors } from "@/theme.js"

const ICON_MAP = {
  normal: 'https://gw.alipayobjects.com/mdn/rms_8fd2eb/afts/img/A*0HC-SawWYUoAAAAAAAAAAABkARQnAQ',
  b: 'https://gw.alipayobjects.com/mdn/rms_8fd2eb/afts/img/A*sxK0RJ1UhNkAAAAAAAAAAABkARQnAQ',
};

export function getGraphTheme() {
  const theme = getThemeColors([
    'accent',
    'accent-active',
    'accent-contrast',
    'accent-tint',
    'border',
    'chart-line',
    'graph-arrow',
    'graph-combo-fill',
    'graph-combo-shadow',
    'graph-combo-stroke',
    'graph-edge',
    'graph-edge-residual',
    'graph-label',
    'graph-label-stroke',
    'graph-node-label-muted',
    'graph-node-linear',
    'graph-node-nonlinear',
    'graph-node-title',
    'graph-node-value',
    'graph-selection',
    'panel',
    'text',
    'text-muted',
  ]);

  return {
    accent: theme.accent,
    accentActive: theme['accent-active'],
    accentContrast: theme['accent-contrast'],
    accentTint: theme['accent-tint'],
    arrow: theme['graph-arrow'],
    border: theme.border,
    chartLine: theme['chart-line'],
    comboFill: theme['graph-combo-fill'],
    comboShadow: theme['graph-combo-shadow'],
    comboStroke: theme['graph-combo-stroke'],
    edge: theme['graph-edge'],
    edgeResidual: theme['graph-edge-residual'],
    label: theme['graph-label'],
    labelStroke: theme['graph-label-stroke'],
    nodeLabelMuted: theme['graph-node-label-muted'],
    nodeLinear: theme['graph-node-linear'],
    nodeNonlinear: theme['graph-node-nonlinear'],
    nodeTitle: theme['graph-node-title'],
    nodeValue: theme['graph-node-value'],
    panel: theme.panel,
    selection: theme['graph-selection'],
    text: theme.text,
    textMuted: theme['text-muted'],
  };
}

G6.registerNode(
  'card-node',
  {
    drawShape: function drawShape(cfg, group) {
      const theme = getGraphTheme();
      const color = cfg.is_linear ? theme.nodeLinear : theme.nodeNonlinear;
      const r = 2;
      const shape = group.addShape('rect', {
        attrs: {
          x: 0,
          y: 0,
          width: 150,
          height: 60,
          stroke: color,
          radius: r,
        },
        name: 'main-box',
        draggable: true,
      });

      group.addShape('rect', {
        attrs: {
          x: 0,
          y: 0,
          width: 150,
          height: 21,
          fill: color,
          radius: [r, r, 0, 0],
        },
        name: 'title-box',
        draggable: true,
      });

      group.addShape('image', {
        attrs: {
          x: 4,
          y: 2,
          height: 16,
          width: 16,
          cursor: 'pointer',
          img: ICON_MAP[cfg.nodeType || 'app'],
        },
        name: 'node-icon',
      });

      group.addShape('text', {
        attrs: {
          textBaseline: 'top',
          y: 5,
          x: 24,
          lineHeight: 20,
          text: cfg.title,
          fill: theme.nodeTitle,
        },
        name: 'title',
      });

      cfg.panels.forEach((item, index) => {
        group.addShape('text', {
          attrs: {
            textBaseline: 'top',
            y: 27,
            x: 24 + index * 60,
            lineHeight: 20,
            text: item.title,
            fill: theme.nodeLabelMuted,
          },
          name: `index-title-${index}`,
        });

        group.addShape('text', {
          attrs: {
            textBaseline: 'top',
            y: 45,
            x: 24 + index * 60,
            lineHeight: 20,
            text: item.value,
            fill: theme.nodeValue,
          },
          name: `index-value-${index}`,
        });
      });

      return shape;
    },
  },
  'single-node',
);

export function getGraphConfig() {
  const theme = getGraphTheme();

  return {
    container: 'graphContainer',
    width: window.innerWidth,
    height: window.innerHeight,
    defaultEdge: {
      type: 'polyline',
      sourceAnchor: 1,
      targetAnchor: 0,
      style: {
        endArrow: {
          path: G6.Arrow.triangle(5, 10),
          fill: theme.arrow,
          opacity: 50,
        },
        stroke: theme.edge,
        radius: 6,
      },
    },
    defaultNode: {
      type: 'modelRect',
      size: [190, 60],
      anchorPoints: [
        [0.5, 0],
        [0.5, 1]
      ],
      logoIcon: {
        show: false,
      },
      stateIcon: {
        show: false,
        img:
          'https://gw.alipayobjects.com/zos/basement_prod/c781088a-c635-452a-940c-0173663456d4.svg',
      },
      labelCfg: {
        offset: 15,
        style: {
          fill: theme.label,
          fontSize: 20,
          stroke: theme.labelStroke,
        }
      },
      descriptionCfg: {
        style: {
          fill: theme.nodeValue,
          fontSize: 14,
        },
      },
    },
    defaultCombo: {
      type: 'rect',
      padding: [20, 16, 16, 16],
      style: {
        fill: theme.comboFill,
        stroke: theme.comboStroke,
        radius: 10,
        lineWidth: 1.5,
      },
      labelCfg: {
        refY: 10,
        style: {
          fontSize: 16,
          fill: theme.label,
          fontWeight: 600,
        },
      },
    },
    comboStateStyles: {
      collapsed: {
        lineWidth: 2.5,
        shadowColor: theme.comboShadow,
        shadowBlur: 12,
      },
    },
    modes: {
      default: [
        'drag-canvas',
        'zoom-canvas',
        'lasso-select',
        {
          type: 'collapse-expand-combo',
          trigger: 'click',
          relayout: true,
        },
      ],
    },
    layout: {
      type: 'dagre',
      rankdir: 'TB',
      nodesep: 10,
      ranksep: 20,
      controlPoints: false,
    },
  };
}
