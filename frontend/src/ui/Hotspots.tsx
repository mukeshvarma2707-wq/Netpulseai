import type { CaseSummary } from "../lib/api";
import { excessOf, statusOf } from "../lib/grid";
import { STATUS_COLOR } from "../lib/theme";
import type { Filter } from "./Controls";
import { Panel } from "./Panel";
import { fmtNum } from "./format";

/** The hour's biggest overloads: a quick way in when towers are tiny from overhead. */
export function Hotspots({
  cases,
  filter,
  selectedId,
  onSelect,
}: {
  cases: CaseSummary[];
  filter: Filter;
  selectedId: number | null;
  onSelect: (id: number) => void;
}) {
  const top = cases
    .filter((c) => filter === "ALL" || c.classification === filter)
    .sort((a, b) => excessOf(b) - excessOf(a))
    .slice(0, 6);

  if (top.length === 0) return null;

  return (
    <Panel className="hidden p-2 sm:block">
      <div className="px-2 pb-1.5 pt-1 font-mono text-[10px] uppercase tracking-[0.22em] text-slate-500">
        Largest overloads
      </div>
      {top.map((c) => {
        const color = STATUS_COLOR[statusOf(c)];
        return (
          <button
            key={c.id}
            onClick={() => onSelect(c.id)}
            className={`flex w-full items-center gap-2.5 rounded-md px-2 py-1.5 text-left font-mono text-xs transition-colors hover:bg-white/5 ${
              selectedId === c.id ? "bg-white/[0.06]" : ""
            }`}
          >
            <span className="h-2 w-2 shrink-0 rounded-sm" style={{ background: color, boxShadow: `0 0 8px ${color}` }} />
            <span className="text-slate-200">Cell {c.cell_id}</span>
            <span className="ml-auto text-slate-400">+{fmtNum(excessOf(c))}</span>
          </button>
        );
      })}
    </Panel>
  );
}
