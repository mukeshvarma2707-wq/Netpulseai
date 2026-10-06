import { COLORS } from "../lib/theme";
import { Panel } from "./Panel";

const ITEMS = [
  { color: COLORS.routine, label: "Routine · being rebalanced" },
  { color: COLORS.anomalous, label: "Anomalous · escalated" },
  { color: COLORS.resolved, label: "Fully resolved" },
  { color: COLORS.beam, label: "Capacity flow", beam: true },
];

export function Legend() {
  return (
    <div className="pointer-events-none absolute bottom-36 right-4 hidden xl:block">
      <Panel className="px-3 py-2.5">
        <div className="mb-1.5 font-mono text-[10px] uppercase tracking-[0.22em] text-slate-500">Legend</div>
        {ITEMS.map((i) => (
          <div key={i.label} className="flex items-center gap-2 py-0.5 text-[11px] text-slate-300">
            {i.beam ? (
              <span className="h-0.5 w-3 rounded-full" style={{ background: `linear-gradient(90deg, ${COLORS.beam}, ${COLORS.resolved})` }} />
            ) : (
              <span className="h-3 w-1.5 rounded-sm" style={{ background: i.color, boxShadow: `0 0 6px ${i.color}` }} />
            )}
            {i.label}
          </div>
        ))}
        <div className="mt-1.5 border-t border-white/5 pt-1.5 text-[10px] text-slate-500">Height = forecast over threshold</div>
      </Panel>
    </div>
  );
}
