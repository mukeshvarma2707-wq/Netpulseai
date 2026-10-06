import type { ReactNode } from "react";

/** Glassy HUD surface used by every overlay. */
export function Panel({ children, className = "" }: { children: ReactNode; className?: string }) {
  return (
    <div
      className={`pointer-events-auto rounded-xl border border-white/[0.07] bg-[#08111f]/80 shadow-[0_8px_40px_rgba(0,0,0,0.45)] backdrop-blur-md ${className}`}
    >
      {children}
    </div>
  );
}
