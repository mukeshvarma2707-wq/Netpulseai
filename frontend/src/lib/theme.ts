// Colour language. Red means ANOMALOUS and nothing else anywhere in the UI.
export const COLORS = {
  bg: "#060b14",
  routine: "#f59e0b",
  anomalous: "#dc2626",
  resolved: "#10b981",
  beam: "#06b6d4",
  tile: "#0d1726",
  tileSource: "#0e7490",
} as const;

export const STATUS_COLOR = {
  routine: COLORS.routine,
  anomalous: COLORS.anomalous,
  resolved: COLORS.resolved,
} as const;
