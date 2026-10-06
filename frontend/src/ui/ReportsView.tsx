import { useCallback, useEffect, useMemo, useState } from "react";
import { motion } from "framer-motion";
import { ChevronDown, Loader2, LocateFixed, RefreshCw, Search } from "lucide-react";
import { api, type ReportSummary } from "../lib/api";
import { useCaseDetail } from "../lib/useData";
import { COLORS } from "../lib/theme";
import { formatHour } from "./format";

interface Props {
  onOpenCase: (id: number, targetDatetime: string) => void;
}

/** GET /reports as an incident log: newest first, expandable to the full report text. */
export function ReportsView({ onOpenCase }: Props) {
  const [reports, setReports] = useState<ReportSummary[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [query, setQuery] = useState("");
  const [open, setOpen] = useState<number | null>(null);

  const load = useCallback(() => {
    setLoading(true);
    setError(null);
    api
      .listReports()
      .then((r) => setReports(r))
      .catch((err) => setError(err instanceof Error ? err.message : String(err)))
      .finally(() => setLoading(false));
  }, []);

  useEffect(load, [load]);

  const rows = useMemo(() => {
    const q = query.trim();
    return (reports ?? [])
      .filter((r) => !q || String(r.cell_id).includes(q) || String(r.case_id).includes(q))
      .sort((a, b) => b.generated_at.localeCompare(a.generated_at));
  }, [reports, query]);

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="absolute inset-0 top-14 z-20 overflow-y-auto bg-[#060b14]/85 backdrop-blur-md"
    >
      <motion.div
        initial={{ y: 16, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ delay: 0.05 }}
        className="mx-auto max-w-5xl px-4 py-8"
      >
        <div className="flex flex-wrap items-end justify-between gap-4">
          <div>
            <div className="font-mono text-[10px] uppercase tracking-[0.22em] text-slate-500">Incident log</div>
            <h1 className="mt-1 text-2xl font-semibold text-slate-50">Generated reports</h1>
            <p className="mt-1 text-sm text-slate-400">
              {reports ? `${reports.length} report${reports.length === 1 ? "" : "s"} on file.` : "Every plain-language report generated for a case."}
            </p>
          </div>
          <div className="flex items-center gap-2">
            <label className="flex items-center gap-2 rounded-md border border-white/10 bg-black/30 px-2.5 py-1.5">
              <Search className="h-3.5 w-3.5 text-slate-500" />
              <input
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Cell or case ID"
                className="w-32 bg-transparent font-mono text-xs text-slate-200 outline-none placeholder:text-slate-600"
              />
            </label>
            <button
              onClick={load}
              className="flex items-center gap-1.5 rounded-md border border-white/10 px-2.5 py-1.5 text-xs text-slate-300 hover:bg-white/5"
            >
              <RefreshCw className={`h-3.5 w-3.5 ${loading ? "animate-spin" : ""}`} /> Refresh
            </button>
          </div>
        </div>

        <div className="mt-6 overflow-hidden rounded-xl border border-white/[0.07] bg-[#08111f]/80">
          <div className="hidden grid-cols-[90px_90px_1fr_120px_190px_32px] gap-3 border-b border-white/[0.06] px-4 py-2.5 font-mono text-[10px] uppercase tracking-wider text-slate-500 md:grid">
            <span>Case</span>
            <span>Cell</span>
            <span>Forecast window</span>
            <span>Class</span>
            <span>Generated (UTC)</span>
            <span />
          </div>

          {error ? (
            <div className="px-4 py-10 text-center font-mono text-xs text-slate-400">
              {error}
              <div>
                <button onClick={load} className="mt-3 rounded-md border border-white/10 px-3 py-1.5 text-slate-200 hover:bg-white/5">
                  Retry
                </button>
              </div>
            </div>
          ) : loading && !reports ? (
            <div className="flex items-center justify-center gap-2 px-4 py-10 font-mono text-xs text-slate-400">
              <Loader2 className="h-4 w-4 animate-spin" /> Loading incident log
            </div>
          ) : rows.length === 0 ? (
            <div className="px-4 py-10 text-center text-sm text-slate-400">
              {query ? "No reports match that ID." : "No reports yet. Open a case in the Command Center and generate one."}
            </div>
          ) : (
            rows.map((r) => (
              <Row
                key={r.case_id}
                r={r}
                open={open === r.case_id}
                onToggle={() => setOpen((o) => (o === r.case_id ? null : r.case_id))}
                onOpenCase={() => onOpenCase(r.case_id, r.target_datetime)}
              />
            ))
          )}
        </div>
      </motion.div>
    </motion.div>
  );
}

function Row({ r, open, onToggle, onOpenCase }: { r: ReportSummary; open: boolean; onToggle: () => void; onOpenCase?: () => void }) {
  const color = r.classification === "ANOMALOUS" ? COLORS.anomalous : COLORS.routine;
  return (
    <div className="border-b border-white/[0.04] last:border-0">
      <button
        onClick={onToggle}
        className="grid w-full grid-cols-[1fr_auto] items-center gap-3 px-4 py-3 text-left font-mono text-xs transition-colors hover:bg-white/[0.03] md:grid-cols-[90px_90px_1fr_120px_190px_32px]"
      >
        <span className="text-slate-500 max-md:hidden">#{r.case_id}</span>
        <span className="text-slate-100">
          <span className="md:hidden">Cell </span>
          {r.cell_id}
        </span>
        <span className="text-slate-300 max-md:hidden">{formatHour(r.target_datetime)}</span>
        <span className="max-md:hidden">
          <span className="rounded px-1.5 py-0.5 text-[10px] font-semibold tracking-wider" style={{ color, background: `${color}1f` }}>
            {r.classification}
          </span>
        </span>
        <span className="text-slate-400 max-md:hidden">{r.generated_at.replace(/\.\d+$/, "")}</span>
        <ChevronDown className={`h-4 w-4 justify-self-end text-slate-500 transition-transform ${open ? "rotate-180" : ""}`} />
      </button>
      {open && <ReportText caseId={r.case_id} onOpenCase={onOpenCase} />}
    </div>
  );
}

function ReportText({ caseId, onOpenCase }: { caseId: number; onOpenCase?: () => void }) {
  const { state } = useCaseDetail(caseId);
  return (
    <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: "auto", opacity: 1 }} className="overflow-hidden">
      <div className="mx-4 mb-4 rounded-lg border border-emerald-400/15 bg-[#030a12] p-3">
        {state?.status === "ready" ? (
          <pre className="whitespace-pre-wrap font-mono text-[12px] leading-relaxed text-emerald-100/90">
            {state.data.report?.report_text ?? "Report text unavailable."}
          </pre>
        ) : state?.status === "error" ? (
          <div className="font-mono text-xs text-slate-400">{state.error}</div>
        ) : (
          <div className="flex items-center gap-2 font-mono text-xs text-slate-400">
            <Loader2 className="h-3.5 w-3.5 animate-spin" /> Fetching report
          </div>
        )}
        <div className="mt-3 flex justify-end">
          {onOpenCase ? (
            <button onClick={onOpenCase} className="flex items-center gap-1.5 rounded-md border border-sky-300/20 px-2.5 py-1 text-[11px] text-sky-200 hover:bg-sky-400/10">
              <LocateFixed className="h-3.5 w-3.5" /> Show on map
            </button>
          ) : (
            <span className="font-mono text-[10px] text-slate-600">Outside the loaded case window</span>
          )}
        </div>
      </div>
    </motion.div>
  );
}
