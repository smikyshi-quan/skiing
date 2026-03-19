/**
 * Ski season grouping utility.
 *
 * A ski season runs October 1 through April 30.
 * Label format: "2025-26 Season" (starting year).
 * Runs outside Oct–Apr fall into "Off-season <year>".
 */

export interface SeasonGroup {
  label: string
  /** Sort key — higher = more recent */
  sortKey: number
  runs: Array<{ id: string; [key: string]: unknown }>
}

/**
 * Return the season label for a given ISO date string.
 *
 * Oct–Dec → season starts that calendar year
 * Jan–Apr → season started previous calendar year
 * May–Sep → off-season for that calendar year
 */
export function seasonLabel(isoDate: string): string {
  const d = new Date(isoDate)
  const month = d.getMonth() // 0-indexed: 0=Jan … 11=Dec
  const year = d.getFullYear()

  if (month >= 9) {
    // Oct (9), Nov (10), Dec (11) → season starts this year
    const endYear = (year + 1) % 100
    return `${year}-${String(endYear).padStart(2, '0')} Season`
  }
  if (month <= 3) {
    // Jan (0), Feb (1), Mar (2), Apr (3) → season started previous year
    const startYear = year - 1
    const endYear = year % 100
    return `${startYear}-${String(endYear).padStart(2, '0')} Season`
  }
  // May–Sep → off-season
  return `Off-season ${year}`
}

/**
 * Sort key for a season label: season year * 10 + (1 for season, 0 for off-season).
 * More recent seasons sort higher.
 */
export function seasonSortKey(isoDate: string): number {
  const d = new Date(isoDate)
  const month = d.getMonth()
  const year = d.getFullYear()

  if (month >= 9) return year * 10 + 1
  if (month <= 3) return (year - 1) * 10 + 1
  return year * 10
}

/**
 * Group an array of runs (with `created_at`) by ski season.
 * Returns groups sorted from most recent season to oldest.
 */
export function groupBySeason<T extends { created_at: string }>(runs: T[]): Array<{ label: string; runs: T[] }> {
  const groups = new Map<string, { sortKey: number; runs: T[] }>()

  for (const run of runs) {
    const label = seasonLabel(run.created_at)
    const existing = groups.get(label)
    if (existing) {
      existing.runs.push(run)
    } else {
      groups.set(label, { sortKey: seasonSortKey(run.created_at), runs: [run] })
    }
  }

  return [...groups.entries()]
    .sort((a, b) => b[1].sortKey - a[1].sortKey)
    .map(([label, { runs: groupRuns }]) => ({ label, runs: groupRuns }))
}
