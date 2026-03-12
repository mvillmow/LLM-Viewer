import G6 from "@antv/g6"

const ICON_MAP = {
  normal: 'https://gw.alipayobjects.com/mdn/rms_8fd2eb/afts/img/A*0HC-SawWYUoAAAAAAAAAAABkARQnAQ',
  b: 'https://gw.alipayobjects.com/mdn/rms_8fd2eb/afts/img/A*sxK0RJ1UhNkAAAAAAAAAAABkARQnAQ',
};

G6.registerNode(
  'card-node',
  {
    drawShape: function drawShape(cfg, group) {
      const color = cfg.is_linear ? '#F4664A' : '#30BF78';
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
        // must be assigned in G6 3.3 and later versions. it can be any string you want, but should be unique in a custom item type
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
        // must be assigned in G6 3.3 and later versions. it can be any string you want, but should be unique in a custom item type
        name: 'title-box',
        draggable: true,
      });

      // left icon
      group.addShape('image', {
        attrs: {
          x: 4,
          y: 2,
          height: 16,
          width: 16,
          cursor: 'pointer',
          img: ICON_MAP[cfg.nodeType || 'app'],
        },
        // must be assigned in G6 3.3 and later versions. it can be any string you want, but should be unique in a custom item type
        name: 'node-icon',
      });

      // title text
      group.addShape('text', {
        attrs: {
          textBaseline: 'top',
          y: 5,
          x: 24,
          lineHeight: 20,
          text: cfg.title,
          fill: '#fff',
        },
        // must be assigned in G6 3.3 and later versions. it can be any string you want, but should be unique in a custom item type
        name: 'title',
      });

      // if (cfg.nodeLevel > 0) {
      //   group.addShape('marker', {
      //     attrs: {
      //       x: 184,
      //       y: 30,
      //       r: 6,
      //       cursor: 'pointer',
      //       symbol: cfg.collapse ? G6.Marker.expand : G6.Marker.collapse,
      //       stroke: '#666',
      //       lineWidth: 1,
      //     },
      //     // must be assigned in G6 3.3 and later versions. it can be any string you want, but should be unique in a custom item type
      //     name: 'collapse-icon',
      //   });
      // }

      // The content list
      cfg.panels.forEach((item, index) => {
        // name text
        group.addShape('text', {
          attrs: {
            textBaseline: 'top',
            y: 27,
            x: 24 + index * 60,
            lineHeight: 20,
            text: item.title,
            fill: 'rgba(0,0,0, 0.4)',
          },
          // must be assigned in G6 3.3 and later versions. it can be any string you want, but should be unique in a custom item type
          name: `index-title-${index}`,
        });

        // value text
        group.addShape('text', {
          attrs: {
            textBaseline: 'top',
            y: 45,
            x: 24 + index * 60,
            lineHeight: 20,
            text: item.value,
            fill: '#595959',
          },
          // must be assigned in G6 3.3 and later versions. it can be any string you want, but should be unique in a custom item type
          name: `index-value-${index}`,
        });
      });
      return shape;
    },
  },
  'single-node',
);


export const graph_config = {
    container: 'graphContainer', // String | HTMLElement，必须，在 Step 1 中创建的容器 id 或容器本身
    width: window.innerWidth, // Number，必须，图的宽度
    height: window.innerHeight, // Number，必须，图的高度
    defaultEdge: {
        type: 'polyline',
        sourceAnchor: 1,
        targetAnchor: 0,
        style: {
            endArrow: {
                path: G6.Arrow.triangle(5, 10),
                fill: "#aaaaaa",
                opacity: 50,
            },
            stroke: "#000000",
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
        
        labelCfg:{
          offset: 15,
          style: {
            fill: '#000000',
            fontSize: 20,
            stroke: '#E7E7E7',
          }
        },
        descriptionCfg: {
          style: {
            fill: '#656565',
            fontSize: 14,
          },
        },
    },
    defaultCombo: {
      type: 'rect',
      padding: [20, 16, 16, 16],
      style: {
        fill: '#F8FAFC',
        stroke: '#CBD5E1',
        radius: 10,
        lineWidth: 1.5,
      },
      labelCfg: {
        refY: 10,
        style: {
          fontSize: 16,
          fill: '#0F172A',
          fontWeight: 600,
        },
      },
    },
    comboStateStyles: {
      collapsed: {
        lineWidth: 2.5,
        shadowColor: '#94A3B8',
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
}
