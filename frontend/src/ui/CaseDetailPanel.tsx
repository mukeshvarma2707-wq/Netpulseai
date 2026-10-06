import { useState, type ReactNode } from "react";
import { motion } from "framer-motion";
import {
  CalendarDays,
  CheckCircle2,
  Crosshair,
  FileText,
  Loader2,
  Network,
  RefreshCw,
  ShieldAlert,
  Radar,
  X,
} from "lucide-react";
import { api, type CaseDetail } from "../lib/api";
import { cellToXY, statusOf } from "../lib/grid";
import { COLORS, STATUS_COLOR } from "../lib/theme";
import { useCaseDetail } from "../lib/useData";
import { fmtNum, formatHour } from "./format";

interface Props {
  caseId: number;
  onClose: () => void;
  onLocateCell: (cellId: number) => void;
  onReportGenerated: (id: number) => void;
}

export function CaseDetailPanel({ caseId, onClose, onLocateCell, onReportGenerated }: Props) {
  const { state, refresh } = useCaseDetail(caseId);

  return (
    <motion.aside
      initial={{ x: "100%", opacity: 0.4 }}
      animate={{ x: 0, opacity: 1 }}
      exit={{ x: "100%", opacity: 0 }}
      transition={{ type: "spring", damping: 30, stiffness: 260 }}
      className="absolute bottom-0 right-0 top-14 z-20 flex w-full flex-col border-l border-white/[0.07] bg-[#07101d]/92 shadow-[-20px_0_60px_rgba(0,0,0,0.5)] backdrop-blur-xl sm:w-[420px]"
    >
      {state?.status === "ready" ? (
        <Detail c={state.data} onClose={onClose} onLocateCell={onLocateCell} onReportGenerated={() => {
          onReportGenerated(caseId);
          refresh();
        }} />
      ) : (
        <div className="flex flex-1 flex-col">
          <div className="flex justify-end p-3">
            <CloseButton onClick={onClose} />
          </div>
          <div className="grid flex-1 place-items-center px-8 text-center font-mono text-xs text-slate-400">
            {state?.status === "error" ? (
              <div>
                <div className="text-slate-300">Could not load case #{caseId}</div>
                <div className="mt-1 text-slate-500">{state.error}</div>
                <button onClick={refresh} className="mt-4 rounded-md border border-white/10 px-3 py-1.5 text-slate-200 hover:bg-white/5">
                  Retry
                </button>
              </div>
            ) : (
              <div className="flex items-center gap-2">
                <Loader2 className="h-4 w-4 animate-spin" /> Loading case #{caseId}
              </div>
            )}
          </div>
        </div>
      )}
    </motion.aside>
  );
}

function Detail({
  c,
  onClose,
  onLocateCell,
  onReportGenerated,
}: {
  c: CaseDetail;
  onClose: () => void;
  onLocateCell: (cellId: number) => void;
  onReportGenerated: () => void;
}) {
  const status = statusOf(c);
  const { x, y } = cellToXY(c.cell_id);
  const isRoutine = c.classification === "ROUTINE";

  return (
    <div className="flex-1 overflow-y-auto">
      <header className="sticky top-0 z-10 border-b border-white/[0.06] bg-[#07101d]/95 px-5 pb-4 pt-4 backdrop-blur">
        <div className="flex items-start justify-between gap-3">
          <div>
            <div className="font-mono text-[10px] uppercase tracking-[0.22em] text-slate-500">
              Case #{c.id} · grid ({x}, {y})
            </div>
            <h2 className="mt-1 flex items-center gap-2 text-xl font-semibold text-slate-50">
              Cell {c.cell_id}
              <ClassificationBadge c={c} />
            </h2>
            <div className="mt-1 font-mono text-xs text-slate-400">{formatHour(c.target_datetime)}</div>
          </div>
          <CloseButton onClick={onClose} />
        </div>
      </header>

      <div className="space-y-6 px-5 py-5">
        <Section title="Forecast vs. threshold">
          <Gauge forecast={c.naive_forecast} threshold={c.congestion_threshold} color={STATUS_COLOR[status === "resolved" ? "routine" : status]} />
        </Section>

        <Section title="Diagnosis">
          <Reason reason={c.reason} />
        </Section>

        {isRoutine ? (
          <Section title="Capacity reallocation">
            <Reallocation c={c} onLocateCell={onLocateCell} />
          </Section>
        ) : (
          <Escalated />
        )}

        <Section title="Operator report">
          <ReportBlock c={c} onGenerated={onReportGenerated} />
        </Section>
      </div>
    </div>
  );
}

export function ClassificationBadge({ c }: { c: Pick<CaseDetail, "classification"> }) {
  const color = c.classification === "ANOMALOUS" ? COLORS.anomalous : COLORS.routine;
  return (
    <span
      className="rounded px-1.5 py-0.5 font-mono text-[10px] font-semibold tracking-wider"
      style={{ color, background: `${color}1f`, border: `1px solid ${color}55` }}
    >
      {c.classification}
    </span>
  );
}

export function Section({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section>
      <h3 className="mb-2.5 font-mono text-[10px] uppercase tracking-[0.22em] text-slate-500">{title}</h3>
      {children}
    </section>
  );
}

function CloseButton({ onClick }: { onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      aria-label="Close panel"
      className="grid h-8 w-8 place-items-center rounded-md text-slate-400 hover:bg-white/5 hover:text-slate-100"
    >
      <X className="h-4 w-4" />
    </button>
  );
}

/** Horizontal gauge: the forecast marker sitting past the threshold line is the problem at a glance. */
export function Gauge({ forecast, threshold, color }: { forecast: number | null; threshold: number | null; color: string }) {
  if (forecast == null || threshold == null || forecast <= 0) {
    return <div className="font-mono text-xs text-slate-500">Forecast value unavailable for this case.</div>;
  }
  const max = Math.max(forecast, threshold) * 1.15;
  const tp = (threshold / max) * 100;
  const fp = (forecast / max) * 100;
  const over = forecast - threshold;

  return (
    <div>
      <div className="relative mt-7 h-3 rounded-full bg-white/[0.05]">
        <div className="absolute inset-y-0 left-0 rounded-l-full bg-slate-500/40" style={{ width: `${Math.min(tp, fp)}%` }} />
        {over > 0 && (
          <motion.div
            initial={{ width: 0 }}
            animate={{ width: `${fp - tp}%` }}
            transition={{ duration: 0.8, delay: 0.15, ease: "easeOut" }}
            className="absolute inset-y-0 rounded-r-full"
            style={{ left: `${tp}%`, background: `linear-gradient(90deg, ${color}66, ${color})`, boxShadow: `0 0 14px ${color}88` }}
          />
        )}
        {/* threshold line */}
        <div className="absolute -bottom-1.5 -top-1.5 w-px bg-slate-200" style={{ left: `${tp}%` }} />
        <div className="absolute -top-6 -translate-x-1/2 whitespace-nowrap font-mono text-[10px] text-slate-400" style={{ left: `${tp}%` }}>
          threshold {fmtNum(threshold)}
        </div>
        {/* forecast marker */}
        <motion.div
          initial={{ left: `${tp}%` }}
          animate={{ left: `${fp}%` }}
          transition={{ duration: 0.8, delay: 0.15, ease: "easeOut" }}
          className="absolute top-1/2 h-4 w-4 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 bg-[#07101d]"
          style={{ borderColor: color, boxShadow: `0 0 10px ${color}` }}
        />
      </div>
      <div className="mt-3 flex items-baseline justify-between font-mono text-xs">
        <span className="text-slate-400">
          forecast <span className="text-slate-100">{fmtNum(forecast)}</span>
        </span>
        <span style={{ color: over > 0 ? color : undefined }}>
          {over > 0 ? "+" : ""}
          {fmtNum(over)} ({over > 0 ? "+" : ""}
          {fmtNum((over / threshold) * 100, 0)}%)
        </span>
      </div>
    </div>
  );
}

const REASON_KINDS = [
  { test: /isolated/i, label: "Isolated", Icon: Radar, hint: "Few neighbouring towers share the overload." },
  { test: /holiday/i, label: "Holiday", Icon: CalendarDays, hint: "Calendar effect in play." },
  { test: /neighbou?rs? also at risk/i, label: "Neighbour-correlated", Icon: Network, hint: "Nearby towers are busy too." },
];

export function Reason({ reason }: { reason: string | null }) {
  if (!reason) return <div className="font-mono text-xs text-slate-500">No reason recorded.</div>;
  const kinds = REASON_KINDS.filter((k) => k.test.test(reason));
  return (
    <div className="rounded-lg border border-white/[0.06] bg-white/[0.02] p-3">
      {kinds.length > 0 && (
        <div className="mb-2 flex flex-wrap gap-1.5">
          {kinds.map(({ label, Icon, hint }) => (
            <span
              key={label}
              title={hint}
              className="flex items-center gap-1.5 rounded-md border border-sky-300/15 bg-sky-300/[0.06] px-2 py-1 text-[11px] text-sky-100"
            >
              <Icon className="h-3.5 w-3.5 text-sky-300" />
              {label}
            </span>
          ))}
        </div>
      )}
      <p className="text-sm leading-relaxed text-slate-300">{reason}</p>
    </div>
  );
}

export function Reallocation({ c, onLocateCell }: { c: CaseDetail; onLocateCell?: (cellId: number) => void }) {
  if (c.deficit == null) {
    return <div className="font-mono text-xs text-slate-500">No solver coverage record for this case.</div>;
  }
  const covered = c.covered ?? 0;
  const pct = c.deficit > 0 ? Math.min(100, (covered / c.deficit) * 100) : 100;
  const sources = [...c.sources].sort((a, b) => b.amount_moved - a.amount_moved);

  return (
    <div className="space-y-3">
      <div className="rounded-lg border border-white/[0.06] bg-white/[0.02] p-3">
        <div className="flex items-baseline justify-between font-mono text-xs">
          <span className="text-slate-400">
            covered <span className="text-slate-100">{fmtNum(covered)}</span> / {fmtNum(c.deficit)}
          </span>
          {c.fully_resolved ? (
            <span className="flex items-center gap-1 rounded px-1.5 py-0.5 text-[10px] font-semibold tracking-wider" style={{ color: COLORS.resolved, background: `${COLORS.resolved}1a` }}>
              <CheckCircle2 className="h-3 w-3" /> FULLY RESOLVED
            </span>
          ) : (
            <span className="text-[11px]" style={{ color: COLORS.routine }}>
              {fmtNum(c.deficit - covered)} short
            </span>
          )}
        </div>
        <div className="mt-2 h-2 overflow-hidden rounded-full bg-white/5">
          <motion.div
            initial={{ width: 0 }}
            animate={{ width: `${pct}%` }}
            transition={{ duration: 0.9, ease: "easeOut" }}
            className="h-full rounded-full"
            style={{ background: `linear-gradient(90deg, ${COLORS.beam}, ${COLORS.resolved})` }}
          />
        </div>
      </div>

      {sources.length === 0 ? (
        <div className="font-mono text-xs text-slate-500">No neighbouring capacity was available to move.</div>
      ) : (
        <ul className="space-y-1">
          {sources.map((s) => (
            <li key={s.source_cell_id} className="group flex items-center gap-3 rounded-md px-2 py-1.5 hover:bg-white/[0.03]">
              {onLocateCell ? (
                <button
                  onClick={() => onLocateCell(s.source_cell_id)}
                  className="flex w-[92px] shrink-0 items-center gap-1.5 font-mono text-xs text-slate-200 hover:text-cyan-200"
                  title="Fly to source cell"
                >
                  <Crosshair className="h-3 w-3 text-cyan-400/70 group-hover:text-cyan-300" />
                  {s.source_cell_id}
                </button>
              ) : (
                <span className="w-[92px] shrink-0 font-mono text-xs text-slate-200">Cell {s.source_cell_id}</span>
              )}
              <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-white/5">
                <div
                  className="h-full rounded-full"
                  style={{ width: `${Math.min(100, (s.amount_moved / c.deficit!) * 100)}%`, background: COLORS.beam }}
                />
              </div>
              <span className="w-16 text-right font-mono text-xs text-slate-300">{fmtNum(s.amount_moved)}</span>
            </li>
          ))}
        </ul>
      )}
      <p className="text-[11px] text-slate-500">
        {sources.length > 1 ? `${sources.length} neighbours pooled capacity in one joint solve.` : "Capacity moved from a neighbouring tower."}
      </p>
    </div>
  );
}

export function Escalated() {
  return (
    <div
      className="flex gap-3 rounded-lg border p-4"
      style={{ borderColor: `${COLORS.anomalous}55`, background: `${COLORS.anomalous}12` }}
    >
      <ShieldAlert className="mt-0.5 h-5 w-5 shrink-0" style={{ color: COLORS.anomalous }} />
      <div>
        <div className="text-sm font-semibold text-slate-100">Escalated — no automatic action taken</div>
        <p className="mt-1 text-xs leading-relaxed text-slate-400">
          This overload doesn't match a known pattern, so the solver was not run. A human operator needs to review it.
        </p>
      </div>
    </div>
  );
}

export function ReportBlock({ c, onGenerated }: { c: CaseDetail; onGenerated: () => void }) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const generate = async () => {
    setBusy(true);
    setError(null);
    try {
      await api.generateReport(c.id);
      onGenerated();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };

  if (!c.report) {
    return (
      <div>
        <button
          onClick={generate}
          disabled={busy}
          className="flex w-full items-center justify-center gap-2 rounded-lg border border-sky-300/25 bg-sky-400/10 px-4 py-2.5 text-sm font-medium text-sky-100 transition-colors hover:bg-sky-400/15 disabled:cursor-wait disabled:opacity-70"
        >
          {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <FileText className="h-4 w-4" />}
          {busy ? "Generating report…" : "Generate report"}
        </button>
        {error && <div className="mt-2 font-mono text-[11px] text-amber-200/80">{error}</div>}
      </div>
    );
  }

  return (
    <div className="overflow-hidden rounded-lg border border-emerald-400/15 bg-[#030a12]">
      <div className="flex items-center justify-between border-b border-emerald-400/10 bg-emerald-400/[0.04] px-3 py-1.5 font-mono text-[10px] text-emerald-300/80">
        <span>LOG · {c.report.generated_at.replace(/\.\d+$/, "")} UTC</span>
        <button onClick={generate} disabled={busy} className="flex items-center gap-1 text-slate-400 hover:text-slate-200 disabled:opacity-50" title="Regenerate report">
          <RefreshCw className={`h-3 w-3 ${busy ? "animate-spin" : ""}`} />
          {busy ? "…" : "regen"}
        </button>
      </div>
      <pre className="max-h-80 overflow-y-auto whitespace-pre-wrap p-3 font-mono text-[12px] leading-relaxed text-emerald-100/90">
        {c.report.report_text}
      </pre>
      {error && <div className="px-3 pb-2 font-mono text-[11px] text-amber-200/80">{error}</div>}
    </div>
  );
}
