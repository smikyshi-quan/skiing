/**
 * Rule-based practice guidance.
 *
 * Maps coaching tips (by keyword matching on title/explanation) to concrete
 * practice drills. When the same coaching category recurs across multiple
 * recent runs, it surfaces as a priority practice area.
 */

import type { CoachingTip } from './analysis-summary'

export interface PracticeDrill {
  id: string
  title: string
  description: string
  /** Which coaching category this addresses */
  category: 'balance' | 'edging' | 'rhythm' | 'movement' | 'general'
  /** How important this drill is — higher = more urgent */
  priority: number
}

interface Rule {
  /** Keywords to match against tip title + explanation (case-insensitive) */
  keywords: string[]
  drill: PracticeDrill
}

const RULES: Rule[] = [
  // ── Balance & Stance ──
  {
    keywords: ['knee', 'symmetr', 'asymmetr', 'flexion', 'uneven'],
    drill: {
      id: 'single-leg-balance',
      title: 'Single-leg balance holds',
      description: 'Stand on one leg for 30s each side before your next run. Focus on keeping hips level and knee tracking over the toe.',
      category: 'balance',
      priority: 0,
    },
  },
  {
    keywords: ['stance', 'width', 'narrow', 'wide', 'feet'],
    drill: {
      id: 'railroad-tracks',
      title: 'Railroad tracks drill',
      description: 'Ski a gentle slope keeping both skis hip-width apart like railroad tracks. Feel equal pressure through both feet.',
      category: 'balance',
      priority: 0,
    },
  },
  {
    keywords: ['lean', 'inclination', 'angulat', 'tilt', 'upright'],
    drill: {
      id: 'javelin-turns',
      title: 'Javelin turns',
      description: 'Hold poles together horizontally at chest height. Make medium turns while keeping the "javelin" pointing downhill — trains proper upper/lower body separation.',
      category: 'balance',
      priority: 0,
    },
  },

  // ── Edging & Grip ──
  {
    keywords: ['edge', 'carv', 'grip', 'skid', 'slip'],
    drill: {
      id: 'edge-lock-traverses',
      title: 'Edge-lock traverses',
      description: 'Traverse across the slope on your uphill edges only, then switch. Feel the ski bite and hold a clean line without skidding.',
      category: 'edging',
      priority: 0,
    },
  },
  {
    keywords: ['pressure', 'weight', 'load', 'transfer', 'com', 'center of mass'],
    drill: {
      id: 'thousand-steps',
      title: '1000 steps drill',
      description: 'Make short, quick steps from ski to ski while turning. Exaggerates weight transfer and builds commitment to the outside ski.',
      category: 'edging',
      priority: 0,
    },
  },

  // ── Turn Rhythm & Shape ──
  {
    keywords: ['rhythm', 'timing', 'tempo', 'consistent', 'duration', 'drift'],
    drill: {
      id: 'metronome-turns',
      title: 'Metronome turns',
      description: 'Count "one-two, one-two" out loud and initiate each turn on the beat. Start on gentle terrain and progressively steepen.',
      category: 'rhythm',
      priority: 0,
    },
  },
  {
    keywords: ['smooth', 'choppy', 'transition', 'flow', 'arc'],
    drill: {
      id: 'garland-turns',
      title: 'Garland turns',
      description: 'Make a series of half-turns (garlands) across the fall line, focusing on a smooth, round arc shape rather than a quick pivot.',
      category: 'rhythm',
      priority: 0,
    },
  },

  // ── Movement & Timing ──
  {
    keywords: ['quiet', 'upper body', 'torso', 'shoulder', 'head', 'sway', 'rotation'],
    drill: {
      id: 'pole-on-shoulders',
      title: 'Pole-across-shoulders drill',
      description: 'Place a pole behind your neck across both shoulders. Make turns while keeping the pole pointing downhill — if it rotates, your upper body is turning too much.',
      category: 'movement',
      priority: 0,
    },
  },
  {
    keywords: ['pole plant', 'hand', 'arm', 'reach'],
    drill: {
      id: 'active-pole-touch',
      title: 'Active pole touch',
      description: 'Focus on a deliberate, light pole plant at the start of each turn. The touch should be a timing cue, not a push — keep hands forward and visible.',
      category: 'movement',
      priority: 0,
    },
  },

  // ── General / fallback ──
  {
    keywords: ['confidence', 'pose', 'capture', 'video', 'quality'],
    drill: {
      id: 'camera-setup',
      title: 'Improve your capture setup',
      description: 'Use a tripod or fixed mount at the side of the course. Ensure the full run is in frame with good lighting — better video = better coaching.',
      category: 'general',
      priority: -1,
    },
  },
]

/**
 * Given a coaching tip, find matching practice drills.
 */
function matchDrills(tip: CoachingTip): PracticeDrill[] {
  const text = `${tip.title} ${tip.explanation}`.toLowerCase()
  const matches: PracticeDrill[] = []

  for (const rule of RULES) {
    if (rule.keywords.some((kw) => text.includes(kw))) {
      matches.push(rule.drill)
    }
  }

  return matches
}

/**
 * Analyse coaching tips from multiple recent runs to produce practice guidance.
 *
 * Tips that recur across runs get higher priority.
 * Returns deduplicated drills sorted by priority (recurring > single occurrence).
 */
export function buildPracticeGuidance(
  recentTipSets: CoachingTip[][],
): PracticeDrill[] {
  const drillCounts = new Map<string, { drill: PracticeDrill; count: number }>()

  for (const tips of recentTipSets) {
    // Track which drills are matched per run (avoid double-counting within one run)
    const seenInRun = new Set<string>()

    for (const tip of tips) {
      const drills = matchDrills(tip)
      for (const drill of drills) {
        if (seenInRun.has(drill.id)) continue
        seenInRun.add(drill.id)

        const existing = drillCounts.get(drill.id)
        if (existing) {
          existing.count++
        } else {
          drillCounts.set(drill.id, { drill, count: 1 })
        }
      }
    }
  }

  return [...drillCounts.values()]
    .sort((a, b) => {
      // Recurring drills first, then by base priority, then alphabetical
      if (b.count !== a.count) return b.count - a.count
      if (b.drill.priority !== a.drill.priority) return b.drill.priority - a.drill.priority
      return a.drill.title.localeCompare(b.drill.title)
    })
    .map(({ drill, count }) => ({
      ...drill,
      priority: count,
    }))
}

/**
 * Build a next-session coaching summary from recent tip patterns.
 * Returns a short headline and the top practice drills.
 */
export function buildNextSessionCard(
  recentTipSets: CoachingTip[][],
): { headline: string; drills: PracticeDrill[] } {
  const drills = buildPracticeGuidance(recentTipSets)

  if (!drills.length) {
    return {
      headline: 'Keep skiing and uploading — your practice plan builds as we see more of your technique.',
      drills: [],
    }
  }

  const recurring = drills.filter((d) => d.priority > 1)
  const topDrill = drills[0]

  let headline: string
  if (recurring.length >= 2) {
    headline = `Your recent runs keep flagging ${recurring[0].category} and ${recurring[1].category}. Focus your next session on these two areas.`
  } else if (recurring.length === 1) {
    headline = `${recurring[0].title} keeps coming up — make it your warm-up priority next time out.`
  } else {
    headline = `Try "${topDrill.title}" in your next session to work on ${topDrill.category}.`
  }

  return { headline, drills: drills.slice(0, 4) }
}
