import type { CaseSummary } from "./api";

export const GRID_SIZE = 100;

/** Real Milan grid position of a cell (1-indexed, x = column, y = row). */
export function cellToXY(cellId: number): { x: number; y: number } {
  return {
    x: ((cellId - 1) % GRID_SIZE) + 1,
    y: Math.floor((cellId - 1) / GRID_SIZE) + 1,
  };
}

/**
 * World-space position of a cell's centre on the ground plane. One grid unit
 * per cell, centred on the origin. Grid y grows "north", which maps to -z so
 * the map reads the right way up from the default camera.
 */
export function cellToWorld(cellId: number): [number, number] {
  const { x, y } = cellToXY(cellId);
  return [x - (GRID_SIZE + 1) / 2, -(y - (GRID_SIZE + 1) / 2)];
}

/** How far the forecast sits above the congestion threshold. */
export function excessOf(c: Pick<CaseSummary, "naive_forecast" | "congestion_threshold" | "deficit">): number {
  if (c.naive_forecast != null && c.congestion_threshold != null && c.naive_forecast > 0) {
    return Math.max(0, c.naive_forecast - c.congestion_threshold);
  }
  return Math.max(0, c.deficit ?? 0);
}

export type TowerStatus = "routine" | "anomalous" | "resolved";

export function statusOf(c: Pick<CaseSummary, "classification" | "fully_resolved">): TowerStatus {
  if (c.classification === "ANOMALOUS") return "anomalous";
  return c.fully_resolved ? "resolved" : "routine";
}
