export interface TechniqueTurn {
  turn_idx: number
  side: string
  start_s?: number
  end_s?: number
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
  time_ranges?: [number, number][]
}

export interface TrackingSegment {
  idx: number
  start_s: number
  end_s: number
  n_confident_frames: number
  mean_confidence: number
  n_turns: number
  is_primary: boolean
}

export type RecapReliability = 'reliable' | 'limited' | 'insufficient'

export type KnownQualityWarningCode =
  | 'low_pose_confidence'
  | 'insufficient_skeleton_detection'
  | 'stance_not_measurable'
  | 'wedge_likely'
  | 'short_clip'
  | 'low_boundary_reliability'
  | 'tracking_loss'
  | 'follow_cam_degraded'

export type QualityWarningCode = KnownQualityWarningCode | (string & {})

export interface TechniqueQualityReport {
  score_reliability?: RecapReliability | string
  score_counts_for_progress?: boolean
  quality_warnings?: QualityWarningCode[]
  stance_measurable?: boolean
  stance_visibility_fraction?: number
  wedge_likely?: boolean
  overall_pose_confidence_mean?: number
  overall_pose_confidence_min?: number
  low_confidence_fraction?: number
  viewpoint_warning?: string | null
  jitter_score_mean?: number
  warnings?: string[]
  resolved_max_fps?: number | null
  resolved_max_dimension?: number | null
}

export interface NormalizedTechniqueQualityReport extends TechniqueQualityReport {
  score_reliability: RecapReliability
  score_counts_for_progress: boolean
  quality_warnings: QualityWarningCode[]
  stance_measurable: boolean
  wedge_likely: boolean
}

export interface TechniqueRunSummary {
  quality_report?: TechniqueQualityReport
  quality?: TechniqueQualityReport
  coaching_tips?: CoachingTip[]
  turns?: TechniqueTurn[]
  segments?: TrackingSegment[]
}

// ── Recap reliability ───────────────────────────────────────
export const INSUFFICIENT_RELIABILITY_WARNING =
  "This video's quality is too low for reliable skeleton detection, so the score may be misleading. Try a better angle, higher resolution, closer camera, steadier shot, and cleaner single-skier framing."

export interface ReliabilityMessage {
  title: string
  explanation: string
  nextStep: string
  hideScoreReason: string
}

export function clipQualityLabel(reliability: RecapReliability): string {
  switch (reliability) {
    case 'reliable':
      return 'Clear'
    case 'limited':
      return 'Usable'
    case 'insufficient':
      return 'Retry'
    default:
      return 'Clear'
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

// Runs analysed before the reliability contract existed carry no
// `score_reliability`. Treat those as 'reliable' so historical recaps and the
// bundled sample aren't retroactively flagged; the pipeline sets the tier
// explicitly for every run it produces now.
function normalizeReliability(value: unknown): RecapReliability {
  return value === 'reliable' || value === 'insufficient' || value === 'limited'
    ? value
    : 'reliable'
}

function normalizedQualitySource(summary: TechniqueRunSummary): TechniqueQualityReport {
  const source = isRecord(summary.quality_report)
    ? summary.quality_report
    : isRecord(summary.quality)
      ? summary.quality
      : {}
  return source as TechniqueQualityReport
}

export function getQualityReport(summary: TechniqueRunSummary): NormalizedTechniqueQualityReport {
  const source = normalizedQualitySource(summary)
  const scoreReliability = normalizeReliability(source.score_reliability)
  const qualityWarnings = toArray<QualityWarningCode>(source.quality_warnings)

  return {
    ...source,
    score_reliability: scoreReliability,
    score_counts_for_progress: typeof source.score_counts_for_progress === 'boolean'
      ? source.score_counts_for_progress
      : scoreReliability !== 'insufficient',
    quality_warnings: qualityWarnings,
    stance_measurable: typeof source.stance_measurable === 'boolean' ? source.stance_measurable : true,
    wedge_likely: source.wedge_likely === true,
  }
}

export function computeReliability(summary: TechniqueRunSummary): RecapReliability {
  return getQualityReport(summary).score_reliability
}

export function scoreCountsForProgress(summary: TechniqueRunSummary | null | undefined): boolean {
  if (!summary) return true
  return getQualityReport(summary).score_counts_for_progress !== false
}

export function combinedQualityWarnings(summary: TechniqueRunSummary): string[] {
  const qualityReport = getQualityReport(summary)
  return [
    ...toArray<string>(qualityReport.quality_warnings),
    ...toArray<string>(qualityReport.warnings),
  ]
}

export function buildReliabilityMessage(summary: TechniqueRunSummary): ReliabilityMessage {
  const reliability = computeReliability(summary)

  if (reliability === 'insufficient') {
    return {
      title: 'Score may be misleading',
      explanation: INSUFFICIENT_RELIABILITY_WARNING,
      nextStep: 'The score is still shown for transparency, but it is excluded from progress trends by default.',
      hideScoreReason: 'The score is visible, but this clip is not reliable enough to count toward progress.',
    }
  }

  if (reliability === 'limited') {
    return {
      title: 'Review with caution',
      explanation: 'Some measurements are less certain for this clip, so treat the score and feedback as directional rather than exact.',
      nextStep: 'This score counts for progress unless the quality report explicitly excludes it.',
      hideScoreReason: 'The score is still visible, but the capture quality makes it less certain than usual.',
    }
  }

  return {
    title: 'Reliable review',
    explanation: 'Tracking was stable enough for full scoring and detailed coaching.',
    nextStep: 'Use the recap and metrics below to decide what to work on next.',
    hideScoreReason: 'The score is safe to show for this run.',
  }
}

// ── Human-readable metric labels ────────────────────────────
export function humanMetricLabel(fill: number): string {
  if (fill >= 78) return 'Strong'
  if (fill >= 55) return 'Moderate'
  if (fill >= 30) return 'Needs work'
  return 'Limited'
}

// ── Score meaning context ───────────────────────────────────
export function scoreContext(score: number): string {
  if (score >= 78) return 'Expert-level mechanics — clean, efficient movement throughout.'
  if (score >= 62) return 'Solid fundamentals with room to refine specific areas.'
  if (score >= 45) return 'Core patterns are forming — the focus areas below will accelerate progress.'
  return 'Scores reflect technical execution; most recreational skiers land between 35–55. Focus on the top priorities below.'
}

// ── Category deduplication groups ───────────────────────────
const CATEGORY_GROUPS: Record<string, string[]> = {
  'Upper Body Control': ['upper_body_quietness', 'shoulder_tilt', 'torso', 'rotation', 'quiet', 'stabilise', 'upper body'],
  'Lower Body Alignment': ['knee', 'hip_knee_ankle', 'flexion', 'alignment', 'knees forward'],
  'Balance & Stance': ['stance', 'balance', 'lean', 'width', 'narrow', 'wide'],
  'Edge Engagement': ['edge', 'carv', 'angle', 'inclination'],
}

function tipGroupKey(tip: CoachingTip): string {
  const text = `${tip.title} ${tip.evidence}`.toLowerCase()
  for (const [group, keywords] of Object.entries(CATEGORY_GROUPS)) {
    if (keywords.some((kw) => text.includes(kw))) return group
  }
  return tip.title // unique fallback — no dedup
}

export function deduplicateTips(tips: CoachingTip[]): CoachingTip[] {
  const seen = new Map<string, CoachingTip>()
  const severityRank: Record<string, number> = { action: 0, warn: 1, info: 2 }
  for (const tip of tips) {
    const group = tipGroupKey(tip)
    const existing = seen.get(group)
    if (!existing || (severityRank[tip.severity] ?? 3) < (severityRank[existing.severity] ?? 3)) {
      seen.set(group, tip)
    }
  }
  return Array.from(seen.values())
}

// ── AI coaching types ─────────────────────────────────────
export interface AiCoachingPoint {
  title: string
  feedback: string
  category: 'balance' | 'edging' | 'rhythm' | 'movement'
  severity: 'action' | 'warn' | 'info'
  recommended_drill_id: string | null
}

export interface AiCoaching {
  judge_status?: 'available' | 'unavailable'
  judge_kind?: 'metrics_only_llm' | string
  judge_error?: string
  coach_summary: string
  coaching_points: AiCoachingPoint[]
  additional_observations: string[]
}

export function isJudgeUnavailable(aiCoaching: AiCoaching | null | undefined): boolean {
  return aiCoaching?.judge_status === 'unavailable'
}

// ── Model limitations ────────────────────────────────────
export interface ModelLimitation {
  title: string
  explanation: string
}

export function generateLimitations(summary: TechniqueRunSummary): ModelLimitation[] {
  const limitations: ModelLimitation[] = []
  const qualityReport = getQualityReport(summary)
  const confidence = qualityReport.overall_pose_confidence_mean ?? 1
  const lowFrac = qualityReport.low_confidence_fraction ?? 0
  const warnings = combinedQualityWarnings(summary)
  const segments = summary.segments ?? []

  if (qualityReport.score_reliability === 'insufficient') {
    limitations.push({
      title: 'Score may be misleading',
      explanation: INSUFFICIENT_RELIABILITY_WARNING,
    })
  }

  if (confidence < 0.70) {
    limitations.push({
      title: 'Tracking was unstable',
      explanation: 'Some body positions were hard to follow consistently, so a few movement readings may be less precise than usual.',
    })
  }

  if (lowFrac > 0.20) {
    limitations.push({
      title: 'Parts of the run were hard to read',
      explanation: 'The skier was only partially visible for parts of the clip, so those sections carry less weight in the recap.',
    })
  }

  if (segments.length > 1) {
    limitations.push({
      title: 'The clip included multiple sections',
      explanation: 'This upload looks like more than one continuous run or camera segment, so comparisons across the whole clip are less useful.',
    })
  }

  if (warnings.some((w) => /occlu/i.test(w))) {
    limitations.push({
      title: 'The skier moved out of view',
      explanation: 'When parts of the body disappear from frame, the recap becomes more directional than exact.',
    })
  }

  if (warnings.some((w) => /camera|angle|perspective/i.test(w))) {
    limitations.push({
      title: 'Camera angle limited the read',
      explanation: 'Side or behind angles give the cleanest feedback. This angle reduced how much we could confidently judge.',
    })
  }

  if (warnings.some((w) => /scene.?cut/i.test(w))) {
    limitations.push({
      title: 'Cuts interrupted the run',
      explanation: 'A single uninterrupted clip works best. Edits or transitions make the recap less consistent.',
    })
  }

  if (qualityReport.stance_measurable === false || warnings.includes('stance_not_measurable')) {
    limitations.push({
      title: 'Stance metrics are unreliable',
      explanation: 'The feet were not clearly visible for enough of the clip, so stance-width and ski-separation metrics are less dependable.',
    })
  }

  if (qualityReport.wedge_likely || warnings.includes('wedge_likely')) {
    limitations.push({
      title: 'Wedge skiing may reduce metric reliability',
      explanation: 'This video appears to include wedge or snowplow skiing, so carved-turn edge and boundary metrics may be less reliable.',
    })
  }

  if (warnings.includes('low_boundary_reliability')) {
    limitations.push({
      title: 'Turn boundaries need caution',
      explanation: 'Some turn boundaries were flagged as lower reliability, so per-turn comparisons may need manual review.',
    })
  }

  if (warnings.includes('follow_cam_degraded')) {
    limitations.push({
      title: 'Follow-cam angle reduced measurement accuracy',
      explanation: 'A following camera angle makes stance and lateral movement harder to measure from a single video.',
    })
  }

  // Always show this baseline caveat
  limitations.push({
    title: 'What this review cannot judge',
    explanation: 'This recap focuses on movement patterns. It cannot fully judge snow conditions, terrain, intent, or tactics from the clip alone.',
  })

  return limitations
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
    clipQualityLabel: string
    turnsDetected: number
    bestTurnScore: number
  }
  reliability: RecapReliability
  categories: DashboardCategory[]
  focusCards: CoachingTip[]
  allTips: CoachingTip[]
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
  const qualityReport = getQualityReport(summary)
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
  const poseConfidence = round((qualityReport.overall_pose_confidence_mean ?? 0) * 100, 0)
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
          value: humanMetricLabel(railPercent(smallerIsBetter(quietnessMean, 0.002, 0.02))),
          helper: 'Less head and torso sway keeps pressure where you need it.',
          leftLabel: 'Busy',
          rightLabel: 'Quiet',
          fill: railPercent(smallerIsBetter(quietnessMean, 0.002, 0.02)),
        },
        {
          label: 'Best turn quality',
          value: `${bestTurnScore}/100`,
          helper: 'Your strongest turn shows the movement pattern that is already repeatable.',
          leftLabel: 'Inconsistent',
          rightLabel: 'Repeatable',
          fill: railPercent(bestTurnScore),
        },
      ],
    },
  ]

  const allRawTips = toArray<CoachingTip>(summary.coaching_tips)
  const deduped = deduplicateTips(allRawTips)
  const focusCards = deduped.slice(0, 2)
  const allTips = deduped

  const reliability = computeReliability(summary)

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
      clipQualityLabel: clipQualityLabel(reliability),
      turnsDetected: turns.length,
      bestTurnScore,
    },
    reliability,
    categories,
    focusCards,
    allTips,
    turnHighlights,
    warnings: combinedQualityWarnings(summary),
  }
}
