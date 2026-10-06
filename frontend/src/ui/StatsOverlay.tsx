import type { CaseSummary } from "../lib/api";
import { COLORS } from "../lib/theme";
import { Panel } from "./Panel";
import { formatHour } from "./format";

interface Props {
  hour: string | null;
  cases: CaseSummary[];
  sources: { loaded: number; total: number; failed: number };
  beamCount: number;
}

export function StatsOverlay({ hour, cases, sources, beamCount }: Props) {
  const routine = cases.filter((c) => c.classification === "ROUTINE");
  const anomalous = cases.length - routine.length;
  const resolved = routine.filter((c) => c.fully_resolved).length;
  const pct = routine.length ? Math.round((resolved / routine.length) * 100) : 0;
  const syncing = sources.loaded < sources.total;

  return (
    <Panel className="p-4">
      <div className="font-mono text-[10px] uppercase tracking-[0.22em] text-slate-500">Forecast window</div>
      <div className="mt-0.5 font-mono text-sm text-slate-100">{hour ? formatHour(hour) : "—"}</div>

      <div className="mt-4 grid grid-cols-3 gap-2">
        <Stat label="Routine" value={routine.length} color={COLORS.routine} />
        <Stat label="Anomalous" value={anomalous} color={COLORS.anomalous} />
        <Stat label="Resolved" value={`${pct}%`} color={COLORS.resolved} />
      </div>

      <div className="mt-3 h-1.5 overflow-hidden rounded-full bg-white/5">
        <div className="h-full rounded-full transition-all duration-700" style={{ width: `${pct}%`, background: COLORS.resolved }} />
      </div>
      <div className="mt-1.5 font-mono text-[10px] text-slate-500">
        {resolved}/{routine.length} routine cases fully covered by reallocation
      </div>

      <div className="mt-3 flex items-center gap-2 border-t border-white/5 pt-3 font-mono text-[10px] text-slate-400">
        <span
          className={`h-1.5 w-1.5 rounded-full ${syncing ? "animate-pulse" : ""}`}
          style={{ background: COLORS.beam }}
        />
        {syncing
          ? `Tracing reallocations ${sources.loaded}/${sources.total}`
          : `${beamCount} reallocation flow${beamCount === 1 ? "" : "s"} active`}
        {sources.failed > 0 && <span className="text-slate-500">· {sources.failed} unavailable</span>}
      </div>
    </Panel>
  );
}

function Stat({ label, value, color }: { label: string; value: number | string; color: string }) {
  return (
    <div className="rounded-md border border-white/5 bg-white/[0.02] px-2.5 py-2">
      <div className="font-mono text-lg font-semibold leading-none" style={{ color }}>
        {value}
      </div>
      <div className="mt-1 text-[10px] uppercase tracking-wider text-slate-500">{label}</div>
    </div>
  );
}
