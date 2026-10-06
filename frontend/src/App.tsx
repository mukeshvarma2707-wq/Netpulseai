import { useCallback, useEffect, useMemo, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { Scene } from "./scene/Scene";
import { HOME_VIEW, type CameraGoal } from "./scene/CameraRig";
import type { Tower } from "./scene/Towers";
import type { Beam } from "./scene/Beams";
import type { TileTint } from "./scene/BaseGrid";
import { useHourCases, useHours, useSourceDetails } from "./lib/useData";
import { cellToWorld, excessOf, statusOf } from "./lib/grid";
import type { CaseSummary } from "./lib/api";
import { TopBar, type View } from "./ui/TopBar";
import { StatsOverlay } from "./ui/StatsOverlay";
import { Controls, type Filter } from "./ui/Controls";
import { Hotspots } from "./ui/Hotspots";
import { Legend } from "./ui/Legend";
import { CaseDetailPanel } from "./ui/CaseDetailPanel";
import { ReportsView } from "./ui/ReportsView";
import { DashboardView } from "./ui/DashboardView";
import { ErrorScreen, LoadingScreen, EmptyScreen } from "./ui/StatusScreens";

export default function App() {
  const { state: hoursState, retry } = useHours();
  const [view, setView] = useState<View>("map");
  const [filter, setFilter] = useState<Filter>("ALL");
  const [hour, setHour] = useState<string | null>(null);
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [pendingSelect, setPendingSelect] = useState<number | null>(null);
  const [hoveredId, setHoveredId] = useState<number | null>(null);
  const [goal, setGoal] = useState<CameraGoal>({ ...HOME_VIEW, nonce: 0 });

  const hours = useMemo(() => (hoursState.status === "ready" ? hoursState.data : []), [hoursState]);
  const activeHour = hour ?? hours[0]?.target_datetime ?? null;
  const hourData = useHourCases(activeHour);
  const { cases: hourCases, patch } = hourData;
  const totalCases = useMemo(() => hours.reduce((n, h) => n + h.total, 0), [hours]);
  const byId = useMemo(() => new Map(hourCases.map((c) => [c.id, c])), [hourCases]);

  // Pillar height: sqrt-scaled excess over threshold, normalised to this
  // hour's 95th percentile so the busiest towers always stand out.
  const heightOf = useMemo(() => {
    const ex = hourCases.map(excessOf).sort((a, b) => a - b);
    const ref = Math.max(1, ex[Math.floor(ex.length * 0.95)] ?? 1);
    return (c: CaseSummary) => 0.4 + 7 * Math.sqrt(Math.min(excessOf(c) / ref, 2.2));
  }, [hourCases]);

  const towers: Tower[] = useMemo(
    () =>
      hourCases.map((c) => ({
        id: c.id,
        cellId: c.cell_id,
        status: statusOf(c),
        height: heightOf(c),
        visible: filter === "ALL" || c.classification === filter,
      })),
    [hourCases, heightOf, filter],
  );
  const towerById = useMemo(() => new Map(towers.map((t) => [t.id, t])), [towers]);

  const routine = useMemo(() => hourCases.filter((c) => c.classification === "ROUTINE"), [hourCases]);
  // Older backends don't return sources inline; fetch those cases one by one.
  const missingSourceIds = useMemo(() => routine.filter((c) => !c.sources).map((c) => c.id), [routine]);
  const fallback = useSourceDetails(missingSourceIds);
  const sourceProgress = missingSourceIds.length
    ? fallback
    : { loaded: routine.length, total: routine.length, failed: 0 };

  const beams: Beam[] = useMemo(() => {
    const out: Beam[] = [];
    for (const c of routine) {
      const list = c.sources ?? fallback.details.get(c.id)?.sources;
      list?.forEach((s) => out.push({ caseId: c.id, from: s.source_cell_id, to: c.cell_id, amount: s.amount_moved }));
    }
    return out;
    // fallback.loaded changes as fallback details stream in
  }, [routine, fallback.details, fallback.loaded]);

  const tints: TileTint = useMemo(() => {
    const m: TileTint = new Map();
    beams.forEach((b) => m.set(b.from, 2));
    hourCases.forEach((c) => m.set(c.cell_id, 1));
    return m;
  }, [beams, hourCases]);

  const selected = selectedId != null ? (towerById.get(selectedId) ?? null) : null;
  const hovered = hoveredId != null ? (towerById.get(hoveredId) ?? null) : null;

  const flyToCell = useCallback((cellId: number, height = 1) => {
    const [x, z] = cellToWorld(cellId);
    // Back off further for taller pillars, and aim slightly to the right of the
    // tower so it sits in the part of the view not covered by the side panel.
    const ox = 6 + height * 0.7;
    const oy = 8 + height * 1.4;
    const oz = 10 + height * 0.9;
    const shift = 0.12 * Math.hypot(ox, oy, oz);
    const rx = oz / Math.hypot(ox, oz);
    const rz = -ox / Math.hypot(ox, oz);
    const tx = x + rx * shift;
    const tz = z + rz * shift;
    setGoal((g) => ({
      position: [tx + ox, height * 0.5 + oy, tz + oz],
      target: [tx, height * 0.5, tz],
      nonce: g.nonce + 1,
    }));
  }, []);

  const select = useCallback(
    (id: number | null) => {
      setSelectedId(id);
      if (id == null) return;
      const c = byId.get(id);
      if (c) flyToCell(c.cell_id, heightOf(c));
    },
    [byId, flyToCell, heightOf],
  );

  const goHome = useCallback(() => setGoal((g) => ({ ...HOME_VIEW, nonce: g.nonce + 1 })), []);

  const closePanel = useCallback(() => {
    setSelectedId(null);
    goHome();
  }, [goHome]);

  const changeHour = useCallback((h: string) => {
    setHour(h);
    setSelectedId(null);
  }, []);

  // Open a case from anywhere (dashboard, incident log): load its hour, then
  // select it once that hour's cases have arrived.
  const openCase = useCallback((id: number, targetDatetime: string) => {
    setHour(targetDatetime);
    setFilter("ALL");
    setView("map");
    setPendingSelect(id);
  }, []);

  useEffect(() => {
    if (pendingSelect == null || !byId.has(pendingSelect)) return;
    select(pendingSelect);
    setPendingSelect(null);
  }, [pendingSelect, byId, select]);

  // A selection that is filtered out or belongs to another hour is dropped.
  useEffect(() => {
    if (selectedId != null && !towers.some((t) => t.id === selectedId && t.visible)) setSelectedId(null);
  }, [towers, selectedId]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && closePanel();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [closePanel]);

  if (hoursState.status === "loading") return <LoadingScreen />;
  if (hoursState.status === "error") return <ErrorScreen message={hoursState.error} onRetry={retry} />;
  if (hours.length === 0 || !activeHour) return <EmptyScreen onRetry={retry} />;

  return (
    <div className="relative h-full w-full overflow-hidden bg-[#060b14] text-slate-200">
      <div className="absolute inset-0">
        <Scene
          towers={towers}
          beams={beams}
          beamsVisible={filter !== "ANOMALOUS"}
          tints={tints}
          selected={selected}
          hovered={hovered}
          goal={goal}
          onHover={setHoveredId}
          onSelect={select}
        />
      </div>

      <TopBar view={view} onView={setView} loaded={totalCases} />

      <AnimatePresence>
        {view === "map" && (
          <motion.div
            key="overlay"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="pointer-events-none absolute inset-0 top-14"
          >
            <div className="pointer-events-none absolute left-4 top-4 flex w-[300px] max-w-[calc(100vw-32px)] flex-col gap-3">
              <StatsOverlay hour={hourData.loadedHour} cases={hourCases} sources={sourceProgress} beamCount={beams.length} />
              {hourData.error && (
                <div className="pointer-events-auto rounded-lg border border-slate-500/25 bg-[#08111f]/90 p-3 font-mono text-[11px] text-slate-300">
                  Couldn't load this hour: {hourData.error}
                  <button onClick={hourData.retry} className="ml-2 text-sky-300 hover:underline">
                    Retry
                  </button>
                </div>
              )}
              <Hotspots cases={hourCases} filter={filter} selectedId={selectedId} onSelect={select} />
            </div>
            <Controls
              filter={filter}
              onFilter={setFilter}
              hours={hours}
              hour={activeHour}
              onHour={changeHour}
              loading={hourData.loading}
              onHome={goHome}
              panelOpen={selectedId != null}
            />
            <Legend />
          </motion.div>
        )}
      </AnimatePresence>

      <AnimatePresence>
        {view === "map" && selectedId != null && (
          <CaseDetailPanel
            key={selectedId}
            caseId={selectedId}
            onClose={closePanel}
            onLocateCell={(cellId) => flyToCell(cellId)}
            onReportGenerated={(id) => patch(id, { has_report: true })}
          />
        )}
      </AnimatePresence>

      <AnimatePresence>
        {view === "log" && <ReportsView key="log" onOpenCase={openCase} />}
        {view === "dashboard" && (
          <DashboardView
            key="dashboard"
            hours={hours}
            onShowOnMap={openCase}
            onReportGenerated={(id) => patch(id, { has_report: true })}
          />
        )}
      </AnimatePresence>
    </div>
  );
}
