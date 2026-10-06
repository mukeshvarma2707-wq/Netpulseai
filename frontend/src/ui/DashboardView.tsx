import { useEffect, useMemo, useState } from "react";
import { motion } from "framer-motion";
import { ArrowDown, ArrowUp, Loader2, LocateFixed, RefreshCw, SlidersHorizontal } from "lucide-react";
import { api, type CaseSummary, type Classification, type HourSummary } from "../lib/api";
import { statusOf } from "../lib/grid";
import { COLORS, STATUS_COLOR } from "../lib/theme";
import { useCaseDetail } from "../lib/useData";
import { ClassificationBadge, Escalated, Gauge, Reallocation, Reason, ReportBlock, Section } from "./CaseDetailPanel";
import { fmtNum } from "./format";

type SortKey = "cell_id" | "target_datetime" | "classification" | "naive_forecast" | "congestion_threshold" | "deficit" | "covered";

const COLUMNS: { key: SortKey | "fully_resolved" | "has_report"; label: string; num?: boolean }[] = [
  { key: "cell_id", label: "Cell", num: true },
  { key: "target_datetime", label: "Target time" },
  { key: "classification", label: "Classification" },
  { key: "naive_forecast", label: "Forecast", num: true },
  { key: "congestion_threshold", label: "Threshold", num: true },
  { key: "deficit", label: "Deficit", num: true },
  { key: "covered", label: "Covered", num: true },
  { key: "fully_resolved", label: "Fully resolved?" },
  { key: "has_report", label: "Report?" },
];

interface Props {
  hours: HourSummary[];
  onShowOnMap: (id: number, targetDatetime: string) => void;
  onReportGenerated: (id: number) => void;
}

/** Tabular dashboard over GET /cases: settings, case table, then the selected case's detail and report. */
export function DashboardView({ hours, onShowOnMap, onReportGenerated }: Props) {
  const [classification, setClassification] = useState<Classification | "">("");
  const [date, setDate] = useState(""); // "" = any date
  const [hourOfDay, setHourOfDay] = useState(""); // "" = every hour of the chosen date
  const [limit, setLimit] = useState(20);
  const [nonce, setNonce] = useState(0);
  const [rows, setRows] = useState<CaseSummary[] | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [sort, setSort] = useState<{ key: SortKey; dir: 1 | -1 } | null>(null);

  // Debounced so dragging the slider doesn't fire a request per step.
  useEffect(() => {
    const ctrl = new AbortController();
    setLoading(true);
    const t = setTimeout(() => {
      api
        .listCases(
          {
            limit,
            classification: classification || undefined,
            date: date && !hourOfDay ? date : undefined,
            target_datetime: date && hourOfDay ? `${date} ${hourOfDay}` : undefined,
          },
          ctrl.signal,
        )
        .then((data) => {
          setRows(data);
          setError(null);
          setSelectedId((id) => (id != null && data.some((c) => c.id === id) ? id : (data[0]?.id ?? null)));
        })
        .catch((err) => err.name !== "AbortError" && setError(err instanceof Error ? err.message : String(err)))
        .finally(() => !ctrl.signal.aborted && setLoading(false));
    }, 250);
    return () => {
      clearTimeout(t);
      ctrl.abort();
    };
  }, [classification, date, hourOfDay, limit, nonce]);

  const firstDay = hours[0]?.target_datetime.slice(0, 10);
  const lastDay = hours[hours.length - 1]?.target_datetime.slice(0, 10);
  const dayHours = useMemo(() => (date ? hours.filter((h) => h.target_datetime.startsWith(date)) : []), [hours, date]);

  const sorted = useMemo(() => {
    if (!rows || !sort) return rows ?? [];
    const { key, dir } = sort;
    return [...rows].sort((a, b) => {
      const av = a[key];
      const bv = b[key];
      if (av == null) return 1;
      if (bv == null) return -1;
      return (av < bv ? -1 : av > bv ? 1 : 0) * dir;
    });
  }, [rows, sort]);

  const toggleSort = (key: SortKey) =>
    setSort((s) => (s?.key !== key ? { key, dir: 1 } : s.dir === 1 ? { key, dir: -1 } : null));

  const reportGenerated = (id: number) => {
    setRows((r) => r?.map((c) => (c.id === id ? { ...c, has_report: true } : c)) ?? r);
    onReportGenerated(id);
  };

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="absolute inset-0 top-14 z-20 flex flex-col overflow-y-auto bg-[#060b14]/95 backdrop-blur-md md:flex-row md:overflow-hidden"
    >
      {/* Settings sidebar */}
      <aside className="shrink-0 border-b border-white/[0.06] bg-[#08111f]/80 p-5 md:w-64 md:border-b-0 md:border-r">
        <div className="flex items-center gap-2 font-mono text-[10px] uppercase tracking-[0.22em] text-slate-500">
          <SlidersHorizontal className="h-3.5 w-3.5" /> Settings
        </div>

        <label className="mt-5 block text-xs text-slate-400" htmlFor="dash-class">
          Filter by classification
        </label>
        <select
          id="dash-class"
          value={classification}
          onChange={(e) => setClassification(e.target.value as Classification | "")}
          className="mt-1.5 w-full rounded-md border border-white/10 bg-black/40 px-2.5 py-2 text-sm text-slate-100 outline-none focus:border-sky-400/50"
        >
          <option value="">All</option>
          <option value="ROUTINE">Routine</option>
          <option value="ANOMALOUS">Anomalous</option>
        </select>

        <label className="mt-5 block text-xs text-slate-400" htmlFor="dash-date">
          Date
        </label>
        <div className="mt-1.5 flex gap-1.5">
          <input
            id="dash-date"
            type="date"
            value={date}
            min={firstDay}
            max={lastDay}
            onChange={(e) => {
              setDate(e.target.value);
              setHourOfDay("");
            }}
            className="min-w-0 flex-1 rounded-md border border-white/10 bg-black/40 px-2.5 py-1.5 font-mono text-sm text-slate-100 outline-none [color-scheme:dark] focus:border-sky-400/50"
          />
          {date && (
            <button
              onClick={() => {
                setDate("");
                setHourOfDay("");
              }}
              className="rounded-md border border-white/10 px-2 text-xs text-slate-400 hover:bg-white/5 hover:text-slate-200"
              title="Any date"
            >
              Clear
            </button>
          )}
        </div>
        {date && dayHours.length === 0 && <div className="mt-1 text-[11px] text-slate-500">No cases on this date.</div>}

        <label className="mt-4 block text-xs text-slate-400" htmlFor="dash-hour">
          Time
        </label>
        <select
          id="dash-hour"
          value={hourOfDay}
          disabled={!date || dayHours.length === 0}
          onChange={(e) => setHourOfDay(e.target.value)}
          className="mt-1.5 w-full rounded-md border border-white/10 bg-black/40 px-2.5 py-2 font-mono text-sm text-slate-100 outline-none focus:border-sky-400/50 disabled:opacity-40"
        >
          <option value="">{date ? "All hours" : "Pick a date first"}</option>
          {dayHours.map((h) => {
            const t = h.target_datetime.slice(11);
            return (
              <option key={t} value={t}>
                {t.slice(0, 5)} · {h.total} cases
              </option>
            );
          })}
        </select>

        <div className="mt-5 flex items-baseline justify-between text-xs text-slate-400">
          <label htmlFor="dash-limit">Number of cases to show</label>
          <span className="font-mono text-slate-100">{limit}</span>
        </div>
        <input
          id="dash-limit"
          type="range"
          min={5}
          max={500}
          step={5}
          value={limit}
          onChange={(e) => setLimit(Number(e.target.value))}
          className="hud-range mt-2.5 w-full"
        />
        <div className="mt-1 flex justify-between font-mono text-[10px] text-slate-600">
          <span>5</span>
          <span>500</span>
        </div>

        <button
          onClick={() => setNonce((n) => n + 1)}
          className="mt-5 flex w-full items-center justify-center gap-2 rounded-md border border-white/10 px-3 py-2 text-sm text-slate-200 hover:bg-white/5"
        >
          <RefreshCw className={`h-3.5 w-3.5 ${loading ? "animate-spin" : ""}`} /> Refresh
        </button>
      </aside>

      {/* Main column */}
      <div className="min-w-0 flex-1 md:overflow-y-auto">
        <div className="mx-auto max-w-6xl px-4 py-6 md:px-8">
          <div className="font-mono text-[10px] uppercase tracking-[0.22em] text-slate-500">Dashboard</div>
          <h1 className="mt-1 text-2xl font-semibold text-slate-50">Network Congestion Diagnosis</h1>
          <p className="mt-1 text-sm text-slate-400">
            {rows ? `Showing ${rows.length} diagnosed case${rows.length === 1 ? "" : "s"}` : "Loading diagnosed cases…"}
            {classification && rows ? ` · ${classification.toLowerCase()} only` : ""}
            {date && rows ? ` · ${date}${hourOfDay ? ` ${hourOfDay.slice(0, 5)}` : ""}` : ""}
          </p>

          <div className="mt-5 overflow-hidden rounded-xl border border-white/[0.07] bg-[#08111f]/80">
            {error ? (
              <div className="px-4 py-10 text-center font-mono text-xs text-slate-400">
                {error}
                <div>
                  <button onClick={() => setNonce((n) => n + 1)} className="mt-3 rounded-md border border-white/10 px-3 py-1.5 text-slate-200 hover:bg-white/5">
                    Retry
                  </button>
                </div>
              </div>
            ) : !rows ? (
              <div className="flex items-center justify-center gap-2 px-4 py-10 font-mono text-xs text-slate-400">
                <Loader2 className="h-4 w-4 animate-spin" /> Loading cases
              </div>
            ) : rows.length === 0 ? (
              <div className="px-4 py-10 text-center text-sm text-slate-400">No cases match these settings.</div>
            ) : (
              <div className={`max-h-[380px] overflow-auto transition-opacity ${loading ? "opacity-60" : ""}`}>
                <table className="w-full min-w-[860px] border-collapse font-mono text-xs">
                  <thead className="sticky top-0 z-10 bg-[#0a1424]">
                    <tr className="text-left text-[10px] uppercase tracking-wider text-slate-500">
                      {COLUMNS.map((col) => {
                        const sortable = col.key !== "fully_resolved" && col.key !== "has_report";
                        const active = sort?.key === col.key;
                        return (
                          <th key={col.key} className={`border-b border-white/[0.06] px-3 py-2.5 font-medium ${col.num ? "text-right" : ""}`}>
                            {sortable ? (
                              <button
                                onClick={() => toggleSort(col.key as SortKey)}
                                className={`inline-flex items-center gap-1 uppercase tracking-wider hover:text-slate-300 ${active ? "text-slate-200" : ""}`}
                              >
                                {col.label}
                                {active && (sort!.dir === 1 ? <ArrowUp className="h-3 w-3" /> : <ArrowDown className="h-3 w-3" />)}
                              </button>
                            ) : (
                              col.label
                            )}
                          </th>
                        );
                      })}
                    </tr>
                  </thead>
                  <tbody>
                    {sorted.map((c) => (
                      <CaseRow key={c.id} c={c} selected={c.id === selectedId} onSelect={() => setSelectedId(c.id)} />
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>

          {selectedId != null && (
            <SelectedCase
              key={selectedId}
              caseId={selectedId}
              onShowOnMap={(id, t) => onShowOnMap(id, t)}
              onReportGenerated={reportGenerated}
            />
          )}
        </div>
      </div>
    </motion.div>
  );
}

function CaseRow({ c, selected, onSelect }: { c: CaseSummary; selected: boolean; onSelect: () => void }) {
  const accent = STATUS_COLOR[statusOf(c)];
  return (
    <tr
      onClick={onSelect}
      aria-selected={selected}
      className={`cursor-pointer border-b border-white/[0.03] transition-colors hover:bg-white/[0.04] ${selected ? "bg-sky-400/[0.08]" : ""}`}
    >
      <td className="px-3 py-2 text-right text-slate-100" style={{ boxShadow: selected ? `inset 2px 0 0 ${accent}` : undefined }}>
        {c.cell_id}
      </td>
      <td className="whitespace-nowrap px-3 py-2 text-slate-300">{c.target_datetime}</td>
      <td className="px-3 py-2">
        <ClassificationBadge c={c} />
      </td>
      <td className="px-3 py-2 text-right text-slate-200">{fmtNum(c.naive_forecast)}</td>
      <td className="px-3 py-2 text-right text-slate-400">{fmtNum(c.congestion_threshold)}</td>
      <td className="px-3 py-2 text-right text-slate-300">{fmtNum(c.deficit)}</td>
      <td className="px-3 py-2 text-right text-slate-300">{fmtNum(c.covered)}</td>
      <td className="px-3 py-2">
        {c.fully_resolved == null ? (
          <span className="text-slate-600">—</span>
        ) : c.fully_resolved ? (
          <span style={{ color: COLORS.resolved }}>Yes</span>
        ) : (
          <span className="text-slate-400">No</span>
        )}
      </td>
      <td className="px-3 py-2">{c.has_report ? <span className="text-sky-300">Yes</span> : <span className="text-slate-500">No</span>}</td>
    </tr>
  );
}

function SelectedCase({
  caseId,
  onShowOnMap,
  onReportGenerated,
}: {
  caseId: number;
  onShowOnMap: (id: number, targetDatetime: string) => void;
  onReportGenerated: (id: number) => void;
}) {
  const { state, refresh } = useCaseDetail(caseId);

  if (!state || state.status === "loading") {
    return (
      <div className="mt-8 flex items-center gap-2 font-mono text-xs text-slate-400">
        <Loader2 className="h-4 w-4 animate-spin" /> Loading case #{caseId}
      </div>
    );
  }
  if (state.status === "error") {
    return (
      <div className="mt-8 font-mono text-xs text-slate-400">
        Could not load case #{caseId}: {state.error}{" "}
        <button onClick={refresh} className="ml-2 text-sky-300 hover:underline">
          Retry
        </button>
      </div>
    );
  }

  const c = state.data;
  const status = statusOf(c);
  const color = STATUS_COLOR[status === "resolved" ? "routine" : status];

  return (
    <motion.section initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} className="mt-8">
      <div className="flex flex-wrap items-end justify-between gap-3 border-b border-white/[0.06] pb-3">
        <div>
          <div className="font-mono text-[10px] uppercase tracking-[0.22em] text-slate-500">Case detail · #{c.id}</div>
          <h2 className="mt-1 flex flex-wrap items-center gap-2 text-xl font-semibold text-slate-50">
            Cell {c.cell_id} <span className="font-mono text-sm font-normal text-slate-400">@ {c.target_datetime}</span>
            <ClassificationBadge c={c} />
          </h2>
        </div>
        <button
          onClick={() => onShowOnMap(c.id, c.target_datetime)}
          className="flex items-center gap-1.5 rounded-md border border-sky-300/20 px-2.5 py-1.5 text-xs text-sky-200 hover:bg-sky-400/10"
        >
          <LocateFixed className="h-3.5 w-3.5" /> Show on 3D map
        </button>
      </div>

      <div className="mt-5 grid gap-6 lg:grid-cols-2">
        <div className="space-y-6">
          <div className="grid grid-cols-2 gap-3">
            <Metric label="Forecasted load" value={fmtNum(c.naive_forecast)} color={color} />
            <Metric label="Congestion threshold" value={fmtNum(c.congestion_threshold)} />
          </div>
          <Gauge forecast={c.naive_forecast} threshold={c.congestion_threshold} color={color} />
          <Section title="Diagnosis">
            <Reason reason={c.reason} />
          </Section>
        </div>

        <div className="space-y-6">
          {c.classification === "ROUTINE" ? (
            <Section title="Capacity reallocation">
              <Reallocation c={c} />
            </Section>
          ) : (
            <Escalated />
          )}
          <Section title="Report">
            <ReportBlock
              c={c}
              onGenerated={() => {
                onReportGenerated(c.id);
                refresh();
              }}
            />
          </Section>
        </div>
      </div>
    </motion.section>
  );
}

function Metric({ label, value, color }: { label: string; value: string; color?: string }) {
  return (
    <div className="rounded-lg border border-white/[0.06] bg-white/[0.02] px-4 py-3">
      <div className="text-[11px] text-slate-500">{label}</div>
      <div className="mt-1 font-mono text-2xl font-semibold text-slate-100" style={{ color }}>
        {value}
      </div>
    </div>
  );
}
