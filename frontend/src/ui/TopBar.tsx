import { Activity, LayoutDashboard, Map as MapIcon, ScrollText } from "lucide-react";
import { API_BASE } from "../lib/api";

export type View = "map" | "dashboard" | "log";

export function TopBar({ view, onView, loaded }: { view: View; onView: (v: View) => void; loaded: number }) {
  const tab = (v: View, label: string, Icon: typeof MapIcon) => (
    <button
      onClick={() => onView(v)}
      aria-label={label}
      className={`flex items-center gap-2 rounded-md px-3 py-1.5 text-xs font-medium tracking-wide transition-colors ${
        view === v ? "bg-sky-400/15 text-sky-200" : "text-slate-400 hover:bg-white/5 hover:text-slate-200"
      }`}
    >
      <Icon className="h-3.5 w-3.5" />
      <span className="hidden sm:inline">{label}</span>
    </button>
  );

  return (
    <header className="absolute inset-x-0 top-0 z-30 flex h-14 items-center justify-between gap-4 border-b border-white/5 bg-[#060b14]/80 px-4 backdrop-blur-md">
      <div className="flex min-w-0 items-center gap-3">
        <div className="grid h-8 w-8 place-items-center rounded-md border border-sky-400/30 bg-sky-400/10">
          <Activity className="h-4 w-4 text-sky-300" />
        </div>
        <div className="min-w-0 leading-tight">
          <div className="text-sm font-semibold tracking-wide text-slate-100">BalanceGrid</div>
          <div className="hidden truncate font-mono text-[10px] uppercase tracking-[0.2em] text-slate-500 sm:block">
            Network Operations · Milan 100×100
          </div>
        </div>
      </div>
      <nav className="flex items-center gap-1">
        {tab("map", "Command Center", MapIcon)}
        {tab("dashboard", "Dashboard", LayoutDashboard)}
        {tab("log", "Incident Log", ScrollText)}
      </nav>
      <div className="hidden items-center gap-2 font-mono text-[10px] text-slate-500 md:flex">
        <span className="relative flex h-2 w-2">
          <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-emerald-400 opacity-50" />
          <span className="relative inline-flex h-2 w-2 rounded-full bg-emerald-400" />
        </span>
        LINKED {API_BASE} · {loaded.toLocaleString()} CASES
      </div>
    </header>
  );
}
