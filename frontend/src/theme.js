const THEME_FALLBACKS = {
  accent: '#2f6f6d',
  'accent-active': '#1f4a48',
  'accent-border-soft': 'rgba(47, 111, 109, 0.35)',
  'accent-contrast': '#f8f4ed',
  'accent-tint': 'rgba(47, 111, 109, 0.1)',
  background: '#f6f1e8',
  border: '#d8cbbb',
  'chart-line': '#3f352c',
  'graph-arrow': '#97897b',
  'graph-combo-fill': '#f6efe6',
  'graph-combo-shadow': 'rgba(148, 163, 184, 0.32)',
  'graph-combo-stroke': '#cdbdab',
  'graph-edge': '#5f564d',
  'graph-edge-residual': '#5f8fbd',
  'graph-label': '#2f261d',
  'graph-label-stroke': '#ece1d3',
  'graph-node-label-muted': 'rgba(47, 38, 29, 0.55)',
  'graph-node-linear': '#c16a4a',
  'graph-node-nonlinear': '#58806d',
  'graph-node-title': '#f8f4ed',
  'graph-node-value': '#5f564d',
  'graph-selection': '#d8ebe5',
  panel: '#fbf7f1',
  text: '#2f261d',
  'text-muted': '#6d6156',
};

export function getThemeColor(token) {
  const key = token.replace(/^--/, '');
  const variableName = `--${key}`;

  if (typeof document === 'undefined') {
    return THEME_FALLBACKS[key] || '';
  }

  const value = getComputedStyle(document.documentElement).getPropertyValue(variableName).trim();
  return value || THEME_FALLBACKS[key] || '';
}

export function getThemeColors(tokens) {
  return Object.fromEntries(tokens.map((token) => [token, getThemeColor(token)]));
}
