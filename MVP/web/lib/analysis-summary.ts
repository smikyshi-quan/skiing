export interface TechniqueTurn {
  turn_idx: number
  side: string
  duration_s: number
  avg_knee_flexion_diff: number
  avg_stance_width_ratio: number
  avg_upper_body_quietness: number
  avg_lean_angle: number
  avg_edge_angle: number
  avg_com_shift_3d: number
  quality_score: number
  smoothness_score: number | null
}

export interface CoachingTip {
  title: string
  explanation: string
  evidence: string
  severity: 'action' | 'warn' | 'info'
}

export interface TechniqueRunSummary {
  quality?: {
    overall_pose_confidence_mean?: number
    warnings?: string[]
  }
  coaching_tips?: CoachingTip[]
  turns?: TechniqueTurn[]
}

export interface DashboardMetric {
  label: string
  value: string
  helper: string
  leftLabel: string
  rightLabel: string
  fill: number
}

export interface DashboardCategory {
  id: string
  title: string
  score: number
  status: string
  metrics: DashboardMetric[]
}

export interface TechniqueDashboard {
  overview: {
    overallScore: number
    smoothnessScore: number | null
    edgeAngle: number
    poseConfidence: number
    turnsDetected: number
    bestTurnScore: number
  }
  categories: DashboardCategory[]
  focusCards: CoachingTip[]
  turnHighlights: Array<{
    title: string
    score: number
    detail: string
  }>
  warnings: string[]
}

function toArray<T>(value: unknown): T[] {
  return Array.isArray(value) ? value : []
}

function clamp(value: number, min: number, max: number) {
  return Math.max(min, Math.min(max, value))
}

function mean(values: number[]) {
  if (!values.length) return 0
  return values.reduce((total, value) => total + value, 0) / values.length
}

function stddev(values: number[]) {
  if (values.length < 2) return 0
  const average = mean(values)
  const variance = mean(values.map((value) => (value - average) ** 2))
  return Math.sqrt(variance)
}

function round(value: number, digits = 1) {
  const factor = 10 ** digits
  return Math.round(value * factor) / factor
}

export function scoreLabel(score: number) {
  if (score >= 78) return 'Dialed'
  if (score >= 62) return 'Good'
  if (score >= 45) return 'Building'
  return 'Focus'
}

function smallerIsBetter(value: number, best: number, worst: number) {
  if (value <= best) return 100
  if (value >= worst) return 0
  return clamp(100 - ((value - best) / (worst - best)) * 100, 0, 100)
}

function closenessScore(value: number, target: number, spread: number) {
  return clamp(100 - (Math.abs(value - target) / spread) * 100, 0, 100)
}

function positiveScore(value: number, floor: number, ceiling: number) {
  if (value <= floor) return 0
  if (value >= ceiling) return 100
  return clamp(((value - floor) / (ceiling - floor)) * 100, 0, 100)
}

function railPercent(score: number) {
  return clamp(Math.round(score), 6, 100)
}

export function buildTechniqueDashboard(summary: TechniqueRunSummary): TechniqueDashboard {
  const turns = toArray<TechniqueTurn>(summary.turns)
  const qualityScores = turns.map((turn) => turn.quality_score).filter((value) => Number.isFinite(value))
  const smoothnessScores = turns
    .map((turn) => turn.smoothness_score)
    .filter((value): value is number => typeof value === 'number' && Number.isFinite(value))
  const edgeAngles = turns.map((turn) => turn.avg_edge_angle).filter((value) => Number.isFinite(value))
  const stanceWidths = turns.map((turn) => turn.avg_stance_width_ratio).filter((value) => Number.isFinite(value))
  const asymmetry = turns.map((turn) => Math.abs(turn.avg_knee_flexion_diff)).filter((value) => Number.isFinite(value))
  const leanAngles = turns.map((turn) => turn.avg_lean_angle).filter((value) => Number.isFinite(value))
  const quietness = turns.map((turn) => turn.avg_upper_body_quietness).filter((value) => Number.isFinite(value))
  const comShift = turns.map((turn) => turn.avg_com_shift_3d).filter((value) => Number.isFinite(value))
  const durations = turns.map((turn) => turn.duration_s).filter((value) => Number.isFinite(value))

  const overallScore = round(mean(qualityScores), 0)
  const smoothnessScore = smoothnessScores.length ? round(mean(smoothnessScores), 0) : null
  const edgeAngle = round(mean(edgeAngles))
  const stanceWidth = round(mean(stanceWidths), 2)
  const kneeAsymmetry = round(mean(asymmetry))
  const leanAngle = round(mean(leanAngles))
  const quietnessMean = mean(quietness)
  const poseConfidence = round((summary.quality?.overall_pose_confidence_mean ?? 0) * 100, 0)
  const comShiftMean = round(mean(comShift), 2)
  const durationDrift = round(stddev(durations), 2)
  const bestTurnScore = round(Math.max(...qualityScores, 0), 0)

  const balanceScore = round(
    mean([
      smallerIsBetter(kneeAsymmetry, 6, 28),
      closenessScore(stanceWidth, 1.45, 1.35),
      closenessScore(leanAngle, 24, 18),
    ]),
    0,
  )

  const edgingScore = round(
    mean([
      closenessScore(edgeAngle, 47, 24),
      closenessScore(comShiftMean, 0.28, 0.26),
      positiveScore(poseConfidence, 55, 92),
    ]),
    0,
  )

  const rhythmScore = round(
    mean([
      smoothnessScore ?? overallScore,
      smallerIsBetter(durationDrift, 0.1, 1.7),
      closenessScore(mean(durations), 1.7, 1.4),
    ]),
    0,
  )

  const movementScore = round(
    mean([
      smallerIsBetter(quietnessMean, 0.002, 0.02),
      overallScore,
      positiveScore(bestTurnScore, 35, 82),
    ]),
    0,
  )

  const categories: DashboardCategory[] = [
    {
      id: 'balance',
      title: 'Balance & Stance',
      score: balanceScore,
      status: scoreLabel(balanceScore),
      metrics: [
        {
          label: 'Knee symmetry',
          value: `${kneeAsymmetry.toFixed(1)}°`,
          helper: 'Keep the load balanced from left to right.',
          leftLabel: 'Uneven',
          rightLabel: 'Stacked',
          fill: railPercent(smallerIsBetter(kneeAsymmetry, 6, 28)),
        },
        {
          label: 'Stance width',
          value: `${stanceWidth.toFixed(2)}x`,
          helper: 'A slightly narrower platform helps clean transitions.',
          leftLabel: 'Narrow',
          rightLabel: 'Wide',
          fill: railPercent(closenessScore(stanceWidth, 1.45, 1.35)),
        },
      ],
    },
    {
      id: 'edging',
      title: 'Edging & Grip',
      score: edgingScore,
      status: scoreLabel(edgingScore),
      metrics: [
        {
          label: 'Average edge angle',
          value: `${edgeAngle.toFixed(1)}°`,
          helper: 'Higher edge angles usually mean stronger carve commitment.',
          leftLabel: 'Flat',
          rightLabel: 'High edge',
          fill: railPercent(closenessScore(edgeAngle, 47, 24)),
        },
        {
          label: 'Pressure shift',
          value: `${comShiftMean.toFixed(2)}m`,
          helper: 'Track how decisively the center of mass moves across turns.',
          leftLabel: 'Static',
          rightLabel: 'Committed',
          fill: railPercent(closenessScore(comShiftMean, 0.28, 0.26)),
        },
      ],
    },
    {
      id: 'rhythm',
      title: 'Turn Rhythm & Shape',
      score: rhythmScore,
      status: scoreLabel(rhythmScore),
      metrics: [
        {
          label: 'Smoothness',
          value: smoothnessScore != null ? `${smoothnessScore}/100` : 'Pending',
          helper: 'Higher smoothness means cleaner timing through the arc.',
          leftLabel: 'Choppy',
          rightLabel: 'Flowing',
          fill: railPercent(smoothnessScore ?? overallScore),
        },
        {
          label: 'Turn timing drift',
          value: `${durationDrift.toFixed(2)}s`,
          helper: 'Lower drift means a steadier rhythm from turn to turn.',
          leftLabel: 'Variable',
          rightLabel: 'Consistent',
          fill: railPercent(smallerIsBetter(durationDrift, 0.1, 1.7)),
        },
      ],
    },
    {
      id: 'movement',
      title: 'Movement & Timing',
      score: movementScore,
      status: scoreLabel(movementScore),
      metrics: [
        {
          label: 'Upper-body quietness',
          value: quietnessMean.toExponential(2),
          helper: 'Less head and torso sway keeps pressure where you need it.',
          leftLabel: 'Busy',
          rightLabel: 'Quiet',
          fill: railPercent(smallerIsBetter(quietnessMean, 0.002, 0.02)),
        },
        {
          label: 'Pose confidence',
          value: `${poseConfidence.toFixed(0)}%`,
          helper: 'A cleaner capture gives more reliable coaching feedback.',
          leftLabel: 'Patchy',
          rightLabel: 'Reliable',
          fill: railPercent(positiveScore(poseConfidence, 55, 92)),
        },
      ],
    },
  ]

  const focusCards = toArray<CoachingTip>(summary.coaching_tips).slice(0, 4)
  const turnHighlights = [...turns]
    .sort((left, right) => right.quality_score - left.quality_score)
    .slice(0, 4)
    .map((turn) => ({
      title: `Turn ${turn.turn_idx + 1} · ${turn.side}`,
      score: round(turn.quality_score, 0),
      detail: `${round(turn.avg_edge_angle)}° edge · ${round(turn.duration_s, 2)}s duration`,
    }))

  return {
    overview: {
      overallScore,
      smoothnessScore,
      edgeAngle,
      poseConfidence,
      turnsDetected: turns.length,
      bestTurnScore,
    },
    categories,
    focusCards,
    turnHighlights,
    warnings: toArray<string>(summary.quality?.warnings),
  }
}
