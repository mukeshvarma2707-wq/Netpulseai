import { useCallback, useEffect, useRef, useState } from "react";
import { api, ApiError, type CaseDetail, type CaseSummary, type HourSummary } from "./api";

export type LoadState<T> =
  | { status: "loading" }
  | { status: "error"; error: string }
  | { status: "ready"; data: T };

function describe(err: unknown): string {
  if (err instanceof ApiError && err.status === 404) {
    return "The server answered, but it has no /cases endpoint. Another app may be running on the API port.";
  }
  return err instanceof Error ? err.message : String(err);
}

/** Every forecast hour with case counts (GET /hours): drives the date/time pickers. */
export function useHours() {
  const [state, setState] = useState<LoadState<HourSummary[]>>({ status: "loading" });
  const [nonce, setNonce] = useState(0);

  useEffect(() => {
    const ctrl = new AbortController();
    setState({ status: "loading" });
    api
      .listHours(ctrl.signal)
      .then((data) => {
        if (!Array.isArray(data)) throw new Error("Unexpected response from /hours.");
        setState({ status: "ready", data });
      })
      .catch((err) => {
        if (err.name === "AbortError") return;
        const outdated = err instanceof ApiError && err.status === 404;
        setState({
          status: "error",
          error: outdated
            ? "The server has no /hours endpoint. Restart uvicorn so it picks up the updated src/api/main.py (or another app is running on the API port)."
            : describe(err),
        });
      });
    return () => ctrl.abort();
  }, [nonce]);

  const retry = useCallback(() => setNonce((n) => n + 1), []);
  return { state, retry };
}

/**
 * All cases for one forecast hour, with reallocation sources inline. The
 * previous hour's data stays on screen while the next one loads.
 */
export function useHourCases(hour: string | null) {
  const [data, setData] = useState<CaseSummary[]>([]);
  const [loadedHour, setLoadedHour] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [nonce, setNonce] = useState(0);

  useEffect(() => {
    if (!hour) return;
    const ctrl = new AbortController();
    setError(null);
    // Short delay so scrubbing through hours doesn't fire a request per step.
    const t = setTimeout(() => {
      api
        .listCases({ target_datetime: hour, include_sources: true }, ctrl.signal)
        .then((rows) => {
          setData(rows);
          setLoadedHour(hour);
        })
        .catch((err) => err.name !== "AbortError" && setError(describe(err)));
    }, 120);
    return () => {
      clearTimeout(t);
      ctrl.abort();
    };
  }, [hour, nonce]);

  const retry = useCallback(() => setNonce((n) => n + 1), []);

  // Patch a single case in place (e.g. has_report after generating one).
  const patch = useCallback((id: number, fields: Partial<CaseSummary>) => {
    setData((rows) => rows.map((c) => (c.id === id ? { ...c, ...fields } : c)));
  }, []);

  return { cases: data, loadedHour, loading: hour !== loadedHour && !error, error, retry, patch };
}

// Shared cache of GET /cases/{id} responses, used by both the detail panel and
// the beam layer (the list endpoint does not include reallocation sources).
const detailCache = new Map<number, CaseDetail>();

export function useCaseDetail(id: number | null) {
  const [state, setState] = useState<LoadState<CaseDetail> | null>(null);
  const [nonce, setNonce] = useState(0);

  useEffect(() => {
    if (id == null) return setState(null);
    const cached = detailCache.get(id);
    if (cached) return setState({ status: "ready", data: cached });
    const ctrl = new AbortController();
    setState({ status: "loading" });
    api
      .getCase(id, ctrl.signal)
      .then((d) => {
        detailCache.set(id, d);
        setState({ status: "ready", data: d });
      })
      .catch((err) => {
        if (err.name !== "AbortError") setState({ status: "error", error: describe(err) });
      });
    return () => ctrl.abort();
  }, [id, nonce]);

  const refresh = useCallback(() => {
    if (id != null) detailCache.delete(id);
    setNonce((n) => n + 1);
  }, [id]);

  return { state, refresh };
}

/**
 * Fetches details (for their reallocation sources) of the given ROUTINE cases,
 * a few at a time, so beams appear progressively instead of all at once.
 */
export function useSourceDetails(caseIds: number[], concurrency = 8) {
  const [, setVersion] = useState(0);
  const [failed, setFailed] = useState(0);
  const key = caseIds.join(",");
  const idsRef = useRef(caseIds);
  idsRef.current = caseIds;

  useEffect(() => {
    const ctrl = new AbortController();
    const queue = idsRef.current.filter((id) => !detailCache.has(id));
    let frame = 0;
    let errors = 0;
    setFailed(0);
    const bump = () => {
      if (!frame) frame = requestAnimationFrame(() => ((frame = 0), setVersion((v) => v + 1)));
    };
    const worker = async () => {
      while (queue.length && !ctrl.signal.aborted) {
        const id = queue.shift()!;
        try {
          detailCache.set(id, await api.getCase(id, ctrl.signal));
          bump();
        } catch (err) {
          if ((err as Error).name === "AbortError") return;
          setFailed(++errors);
        }
      }
    };
    for (let i = 0; i < concurrency; i++) void worker();
    return () => {
      ctrl.abort();
      cancelAnimationFrame(frame);
    };
  }, [key, concurrency]);

  const loaded = caseIds.filter((id) => detailCache.has(id)).length;
  return { details: detailCache, loaded, total: caseIds.length, failed };
}

export function cacheDetail(detail: CaseDetail) {
  detailCache.set(detail.id, detail);
}
