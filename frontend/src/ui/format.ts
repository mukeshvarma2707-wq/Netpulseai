const hourFmt = new Intl.DateTimeFormat("en-GB", {
  weekday: "short",
  day: "2-digit",
  month: "short",
  year: "numeric",
  hour: "2-digit",
  minute: "2-digit",
});

/** "2013-11-02 01:00:00" -> "Sat, 02 Nov 2013, 01:00". Timestamps are shown as-is (no tz shift). */
export function formatHour(ts: string): string {
  const d = new Date(ts.replace(" ", "T"));
  return Number.isNaN(d.getTime()) ? ts : hourFmt.format(d);
}

export function fmtNum(n: number | null | undefined, digits = 1): string {
  return n == null ? "—" : n.toLocaleString(undefined, { minimumFractionDigits: digits, maximumFractionDigits: digits });
}
