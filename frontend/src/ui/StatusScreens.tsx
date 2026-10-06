import { Activity, DatabaseZap, PlugZap, RefreshCw } from "lucide-react";
import type { ReactNode } from "react";
import { API_BASE } from "../lib/api";

function Shell({ children }: { children: ReactNode }) {
  return (
    <div className="relative grid h-full w-full place-items-center overflow-hidden bg-[#060b14] px-4 text-slate-200">
      {/* faint static grid so even error states read as "the ops room" */}
      <div
        className="pointer-events-none absolute inset-0 opacity-[0.35]"
        style={{
          backgroundImage:
            "linear-gradient(#0f1d31 1px, transparent 1px), linear-gradient(90deg, #0f1d31 1px, transparent 1px)",
          backgroundSize: "28px 28px",
          maskImage: "radial-gradient(ellipse at center, black 30%, transparent 75%)",
        }}
      />
      <div className="relative w-full max-w-md">{children}</div>
    </div>
  );
}

export function LoadingScreen() {
  return (
    <Shell>
      <div className="flex flex-col items-center text-center">
        <div className="relative grid h-16 w-16 place-items-center">
          <span className="absolute inset-0 animate-ping rounded-full border border-sky-400/30" />
          <span className="absolute inset-2 rounded-full border border-sky-400/40" />
          <Activity className="h-6 w-6 text-sky-300" />
        </div>
        <div className="mt-6 font-mono text-xs uppercase tracking-[0.3em] text-slate-400">Linking to network</div>
        <div className="mt-2 font-mono text-[11px] text-slate-600">GET {API_BASE}/cases</div>
      </div>
    </Shell>
  );
}

export function ErrorScreen({ message, onRetry }: { message: string; onRetry: () => void }) {
  return (
    <Shell>
      <div className="rounded-xl border border-slate-500/25 bg-[#08111f]/90 p-6 shadow-2xl backdrop-blur">
        <div className="flex items-center gap-3">
          <div className="grid h-10 w-10 place-items-center rounded-lg border border-slate-400/20 bg-slate-400/10">
            <PlugZap className="h-5 w-5 text-slate-300" />
          </div>
          <div>
            <div className="font-mono text-[10px] uppercase tracking-[0.22em] text-slate-500">Link down</div>
            <div className="text-base font-semibold text-slate-100">Can't reach the BalanceGrid API</div>
          </div>
        </div>
        <p className="mt-4 font-mono text-xs leading-relaxed text-slate-400">{message}</p>
        <div className="mt-4 rounded-md border border-white/[0.06] bg-black/40 p-3 font-mono text-[11px] leading-relaxed text-slate-400">
          <div className="text-slate-500"># start the backend from the repo root</div>
          <div className="text-slate-200">uvicorn src.api.main:app --reload</div>
          <div className="mt-2 text-slate-500"># or point the UI elsewhere (frontend/.env)</div>
          <div className="text-slate-200">VITE_API_TARGET=http://127.0.0.1:8000</div>
        </div>
        <button
          onClick={onRetry}
          className="mt-5 flex w-full items-center justify-center gap-2 rounded-lg border border-sky-300/25 bg-sky-400/10 px-4 py-2 text-sm text-sky-100 hover:bg-sky-400/15"
        >
          <RefreshCw className="h-4 w-4" /> Retry connection
        </button>
      </div>
    </Shell>
  );
}

export function EmptyScreen({ onRetry }: { onRetry: () => void }) {
  return (
    <Shell>
      <div className="rounded-xl border border-white/[0.08] bg-[#08111f]/90 p-6 text-center shadow-2xl">
        <DatabaseZap className="mx-auto h-8 w-8 text-slate-400" />
        <div className="mt-3 text-base font-semibold text-slate-100">No diagnosed cases yet</div>
        <p className="mt-2 text-sm text-slate-400">The API is up, but the database is empty. Load the pipeline outputs:</p>
        <div className="mt-3 rounded-md bg-black/40 p-2.5 font-mono text-[11px] text-slate-200">python src/db/populate_db.py</div>
        <button onClick={onRetry} className="mt-5 rounded-lg border border-white/10 px-4 py-2 text-sm text-slate-200 hover:bg-white/5">
          Check again
        </button>
      </div>
    </Shell>
  );
}
