// Typed client for the BalanceGrid FastAPI backend (src/api/main.py).

export type Classification = "ROUTINE" | "ANOMALOUS";

export interface CaseSummary {
  id: number;
  cell_id: number;
  target_datetime: string;
  classification: Classification;
  naive_forecast: number | null;
  congestion_threshold: number | null;
  deficit: number | null;
  covered: number | null;
  fully_resolved: boolean | null;
  has_report: boolean;
  /** Present when requested with include_sources. */
  sources?: ReallocationSource[];
}

export interface HourSummary {
  target_datetime: string;
  total: number;
  routine: number;
  anomalous: number;
}

export interface CaseQuery {
  limit?: number;
  offset?: number;
  classification?: Classification;
  target_datetime?: string;
  date?: string;
  include_sources?: boolean;
}

export interface ReallocationSource {
  source_cell_id: number;
  amount_moved: number;
}

export interface CaseDetail extends Omit<CaseSummary, "has_report"> {
  reason: string | null;
  sources: ReallocationSource[];
  report: { report_text: string; generated_at: string } | null;
}

export interface ReportSummary {
  case_id: number;
  cell_id: number;
  target_datetime: string;
  classification: Classification;
  generated_at: string;
}

export const API_BASE = (import.meta.env.VITE_API_BASE_URL as string | undefined)?.replace(/\/$/, "") || "/api";
// Upper bound on cases fetched for one hour (the busiest hour has ~4k).
export const CASE_LIMIT = Number(import.meta.env.VITE_CASE_LIMIT) || 10000;

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number | null,
  ) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE}${path}`, init);
  } catch (err) {
    if ((err as Error).name === "AbortError") throw err;
    throw new ApiError(`Could not reach the BalanceGrid API at ${API_BASE}.`, null);
  }
  if (!res.ok) {
    let detail = res.statusText;
    try {
      detail = (await res.json()).detail ?? detail;
    } catch {
      /* non-JSON error body */
    }
    // The dev proxy also answers 5xx when nothing is listening on the target port.
    if (res.status >= 500) {
      throw new ApiError(`The API behind ${API_BASE} failed (${res.status}). Is the BalanceGrid backend running?`, res.status);
    }
    throw new ApiError(`${res.status} ${detail}`, res.status);
  }
  return res.json() as Promise<T>;
}

export const api = {
  listCases: (q: CaseQuery = {}, signal?: AbortSignal) => {
    const params = new URLSearchParams();
    for (const [k, v] of Object.entries({ limit: CASE_LIMIT, ...q })) if (v !== undefined && v !== "") params.set(k, String(v));
    return request<CaseSummary[]>(`/cases?${params}`, { signal });
  },
  listHours: (signal?: AbortSignal) => request<HourSummary[]>(`/hours`, { signal }),
  getCase: (id: number, signal?: AbortSignal) => request<CaseDetail>(`/cases/${id}`, { signal }),
  generateReport: (id: number) =>
    request<{ status: string; report_text: string }>(`/cases/${id}/report`, { method: "POST" }),
  listReports: () => request<ReportSummary[]>(`/reports`),
};
