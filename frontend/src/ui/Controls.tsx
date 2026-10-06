import { useEffect, useMemo, useState } from "react";
import { CalendarDays, ChevronLeft, ChevronRight, Home, Loader2, Pause, Play } from "lucide-react";
import type { HourSummary } from "../lib/api";
import { COLORS } from "../lib/theme";
import { Panel } from "./Panel";

export type Filter = "ALL" | "ROUTINE" | "ANOMALOUS";

const FILTERS: { value: Filter; label: string; color?: string }[] = [
  { value: "ALL", label: "All" },
  { value: "ROUTINE", label: "Routine", color: COLORS.routine },
  { value: "ANOMALOUS", label: "Anomalous", color: COLORS.anomalous },
];

interface Props {
  filter: Filter;
  onFilter: (f: Filter) => void;
  hours: HourSummary[];
  hour: string;
  onHour: (h: string) => void;
  loading: boolean;
  onHome: () => void;
  panelOpen: boolean;
}

/** "2013-11-02 08:00:00" -> ["2013-11-02", 8] */
const splitHour = (ts: string): [string, number] => [ts.slice(0, 10), Number(ts.slice(11, 13))];

export function Controls({ filter, onFilter, hours, hour, onHour, loading, onHome, panelOpen }: Props) {
  const [playing, setPlaying] = useState(false);
  const index = useMemo(() => new Map(hours.map((h, i) => [h.target_datetime, i])), [hours]);
  const days = useMemo(() => {
    const m = new Map<string, HourSummary[]>();
    for (const h of hours) {
      const d = splitHour(h.target_datetime)[0];
      m.set(d, [...(m.get(d) ?? []), h]);
    }
    return m;
  }, [hours]);

  const i = index.get(hour) ?? 0;
  const [day, hh] = splitHour(hour);
  const dayHours = days.get(day) ?? [];
  const byHourOfDay = new Map(dayHours.map((h) => [splitHour(h.target_datetime)[1], h]));
  const dayMax = Math.max(1, ...dayHours.map((h) => h.total));
  const current = hours[i];
  const first = hours[0]?.target_datetime.slice(0, 10);
  const last = hours[hours.length - 1]?.target_datetime.slice(0, 10);

  const step = (d: number) => onHour(hours[Math.min(hours.length - 1, Math.max(0, i + d))].target_datetime);

  useEffect(() => {
    if (!playing || loading) return;
    const t = setTimeout(() => onHour(hours[i >= hours.length - 1 ? 0 : i + 1].target_datetime), 3500);
    return () => clearTimeout(t);
  }, [playing, loading, i, hours, onHour]);

  // Picking a day keeps the same hour-of-day when that day has it.
  const pickDay = (d: string) => {
    let list = days.get(d);
    if (!list) {
      // Snap to the nearest day that has data.
      const next = hours.find((h) => h.target_datetime.slice(0, 10) >= d) ?? hours[hours.length - 1];
      list = days.get(next.target_datetime.slice(0, 10))!;
    }
    const same = list.find((h) => splitHour(h.target_datetime)[1] === hh);
    onHour((same ?? list[0]).target_datetime);
  };

  const btn =
    "grid h-8 w-8 shrink-0 place-items-center rounded-md text-slate-400 transition-colors hover:bg-white/5 hover:text-slate-100 disabled:opacity-30";

  return (
    <div
      className={`pointer-events-none absolute inset-x-0 bottom-4 flex justify-center px-4 transition-[padding] duration-300 ${
        panelOpen ? "sm:pr-[436px]" : ""
      }`}
    >
      <Panel className="w-full max-w-[880px] px-3 pb-2 pt-2.5">
        <div className="flex flex-wrap items-center gap-x-3 gap-y-2">
          <div className="flex rounded-lg bg-black/30 p-0.5" role="radiogroup" aria-label="Filter by classification">
            {FILTERS.map((f) => (
              <button
                key={f.value}
                role="radio"
                aria-checked={filter === f.value}
                onClick={() => onFilter(f.value)}
                className={`flex items-center gap-1.5 rounded-md px-2.5 py-1 text-xs font-medium transition-colors ${
                  filter === f.value ? "bg-white/10 text-slate-100" : "text-slate-400 hover:text-slate-200"
                }`}
              >
                {f.color && <span className="h-1.5 w-1.5 rounded-full" style={{ background: f.color }} />}
                {f.label}
              </button>
            ))}
          </div>

          <label className="flex items-center gap-2 rounded-md border border-white/10 bg-black/30 px-2 py-1">
            <CalendarDays className="h-3.5 w-3.5 text-slate-400" />
            <input
              type="date"
              value={day}
              min={first}
              max={last}
              onChange={(e) => e.target.value && pickDay(e.target.value)}
              className="bg-transparent font-mono text-xs text-slate-100 outline-none [color-scheme:dark]"
              aria-label="Forecast date"
            />
          </label>

          <div className="flex items-center">
            <button className={btn} disabled={i <= 0} onClick={() => step(-1)} aria-label="Previous hour">
              <ChevronLeft className="h-4 w-4" />
            </button>
            <button className={btn} onClick={() => setPlaying((p) => !p)} aria-label={playing ? "Pause" : "Play hours"}>
              {playing ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
            </button>
            <button className={btn} disabled={i >= hours.length - 1} onClick={() => step(1)} aria-label="Next hour">
              <ChevronRight className="h-4 w-4" />
            </button>
          </div>

          <div className="flex min-w-[120px] items-center gap-2 font-mono text-xs">
            <span className="text-base text-slate-100">{String(hh).padStart(2, "0")}:00</span>
            {loading ? (
              <Loader2 className="h-3.5 w-3.5 animate-spin text-slate-400" />
            ) : (
              current && (
                <span className="text-[10px] text-slate-500">
                  <span style={{ color: COLORS.routine }}>{current.routine}</span> /{" "}
                  <span style={{ color: COLORS.anomalous }}>{current.anomalous}</span>
                </span>
              )
            )}
          </div>

          <button className={`${btn} ml-auto`} onClick={onHome} aria-label="Reset view" title="Reset view">
            <Home className="h-4 w-4" />
          </button>
        </div>

        {/* 24-hour strip for the selected day: bar height = cases that hour */}
        <div className="mt-2 grid items-end gap-[3px]" style={{ gridTemplateColumns: "repeat(24, minmax(0, 1fr))" }}>
          {Array.from({ length: 24 }, (_, h) => {
            const s = byHourOfDay.get(h);
            const selected = h === hh;
            const height = s ? 6 + (s.total / dayMax) * 26 : 3;
            const anomShare = s && s.total ? s.anomalous / s.total : 0;
            return (
              <button
                key={h}
                disabled={!s}
                onClick={() => s && onHour(s.target_datetime)}
                title={s ? `${String(h).padStart(2, "0")}:00 · ${s.routine} routine, ${s.anomalous} anomalous` : `${h}:00 · no cases`}
                className={`group flex h-10 flex-col items-stretch justify-end rounded-sm px-[1px] pb-[1px] transition-colors ${
                  selected ? "bg-sky-400/15 ring-1 ring-sky-300/50" : "hover:bg-white/5"
                } disabled:cursor-default`}
              >
                <div className="flex flex-col overflow-hidden rounded-[2px]" style={{ height }}>
                  <div style={{ flex: anomShare, background: COLORS.anomalous, opacity: selected ? 1 : 0.7 }} />
                  <div style={{ flex: s ? 1 - anomShare : 1, background: s ? COLORS.routine : "#1e293b", opacity: selected ? 1 : 0.55 }} />
                </div>
              </button>
            );
          })}
        </div>
        <div className="mt-0.5 flex justify-between px-0.5 font-mono text-[9px] text-slate-600">
          <span>00</span>
          <span>06</span>
          <span>12</span>
          <span>18</span>
          <span>23</span>
        </div>
      </Panel>
    </div>
  );
}
