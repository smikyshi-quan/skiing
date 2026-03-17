'use client'

import { useEffect, useState, useCallback } from 'react'
import { useParams } from 'next/navigation'
import Link from 'next/link'
import { Job, ArtifactWithUrl, JobStatus } from '@/lib/types'

interface JobResponse {
  job: Job
  artifacts: ArtifactWithUrl[]
}

const ACTIVE: Set<JobStatus> = new Set(['created', 'uploaded', 'queued', 'running'])

const STATUS_LABEL: Record<JobStatus, string> = {
  created:  'Job created',
  uploaded: 'Video uploaded',
  queued:   'Waiting in queue',
  running:  'Analysing run…',
  done:     'Analysis complete',
  error:    'Analysis failed',
}

const STATUS_DOT: Record<JobStatus, string> = {
  created:  'rgba(255,255,255,0.25)',
  uploaded: '#4F8EFF',
  queued:   '#F59E0B',
  running:  '#4F8EFF',
  done:     '#22D07A',
  error:    '#F87171',
}

const STATUS_TEXT: Record<JobStatus, string> = {
  created:  'rgba(255,255,255,0.5)',
  uploaded: '#93C5FD',
  queued:   '#FCD34D',
  running:  '#93C5FD',
  done:     '#6EE7B7',
  error:    '#FCA5A5',
}

const STATUS_BG: Record<JobStatus, string> = {
  created:  'rgba(255,255,255,0.07)',
  uploaded: 'rgba(79,142,255,0.12)',
  queued:   'rgba(245,158,11,0.12)',
  running:  'rgba(79,142,255,0.12)',
  done:     'rgba(34,208,122,0.12)',
  error:    'rgba(248,113,113,0.12)',
}

export default function JobDetailPage() {
  const { id } = useParams<{ id: string }>()
  const [data, setData] = useState<JobResponse | null>(null)
  const [fetchError, setFetchError] = useState<string | null>(null)

  const loadJob = useCallback(async () => {
    try {
      const res = await fetch(`/api/jobs/${id}`)
      if (!res.ok) throw new Error(`${res.status}`)
      const json: JobResponse = await res.json()
      setData(json)
      return json.job.status
    } catch (err) {
      setFetchError(err instanceof Error ? err.message : 'Failed to load job')
      return null
    }
  }, [id])

  useEffect(() => {
    let timer: ReturnType<typeof setTimeout>

    async function poll() {
      const status = await loadJob()
      if (status && ACTIVE.has(status as JobStatus)) {
        timer = setTimeout(poll, 4000)
      }
    }

    poll()
    return () => clearTimeout(timer)
  }, [loadJob])

  if (fetchError) {
    return (
      <div className="space-y-3">
        <div
          className="text-sm rounded-xl px-4 py-3"
          style={{ background: 'rgba(248,113,113,0.1)', color: '#FCA5A5', border: '1px solid rgba(248,113,113,0.25)' }}
        >
          {fetchError}
        </div>
        <Link href="/jobs" className="text-sm hover:underline" style={{ color: 'var(--accent)' }}>
          ← Back to runs
        </Link>
      </div>
    )
  }

  if (!data) {
    return (
      <div className="space-y-4 animate-pulse">
        <div className="h-5 w-32 rounded-lg" style={{ background: 'rgba(255,255,255,0.07)' }} />
        <div className="h-28 rounded-2xl" style={{ background: 'var(--bg-surface)', border: '1px solid var(--border-subtle)' }} />
      </div>
    )
  }

  const { job, artifacts } = data
  const progressNote = typeof job.config?.progress_note === 'string' ? job.config.progress_note : null
  const overlayArtifact = artifacts.find((a) => a.kind === 'video_overlay')
  const coolMomentPhotos = artifacts.filter((a) => a.kind === 'cool_moment_photo')
  const peakFrames = artifacts.filter(
    (a) => a.kind === 'peak_pressure_frame' || a.kind === 'peak_pressure_frame_enhanced'
  )
  const summaryArtifact = artifacts.find((a) => a.kind === 'summary_json')
  const metricsArtifact = artifacts.find((a) => a.kind === 'metrics_csv')
  const isActive = ACTIVE.has(job.status)

  return (
    <div className="space-y-7">
      {/* Breadcrumb */}
      <div className="flex items-center gap-2 text-sm" style={{ color: 'rgba(255,255,255,0.35)' }}>
        <Link href="/jobs" className="hover:text-white transition-colors">
          My Runs
        </Link>
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
          <path d="M9 18l6-6-6-6"/>
        </svg>
        <span className="font-mono text-white">{job.id.slice(0, 8)}</span>
      </div>

      {/* Status card */}
      <div className="card p-5">
        <div className="flex items-start justify-between gap-4">
          <div className="flex items-center gap-3">
            <div
              className="w-10 h-10 rounded-xl flex items-center justify-center shrink-0"
              style={{ background: STATUS_BG[job.status] }}
            >
              {isActive ? (
                <div
                  className="w-3 h-3 rounded-full animate-pulse"
                  style={{ background: STATUS_DOT[job.status] }}
                />
              ) : job.status === 'done' ? (
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke={STATUS_DOT[job.status]} strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                  <polyline points="20 6 9 17 4 12"/>
                </svg>
              ) : (
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke={STATUS_DOT[job.status]} strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                  <circle cx="12" cy="12" r="10"/><path d="M12 8v4M12 16h.01"/>
                </svg>
              )}
            </div>
            <div>
              <p className="font-semibold text-white">{STATUS_LABEL[job.status]}</p>
              <p className="text-xs mt-0.5" style={{ color: 'rgba(255,255,255,0.3)' }}>
                Updated {new Date(job.updated_at).toLocaleString()}
              </p>
            </div>
          </div>

          <span
            className="shrink-0 text-xs font-semibold px-3 py-1.5 rounded-full"
            style={{ background: STATUS_BG[job.status], color: STATUS_TEXT[job.status] }}
          >
            {STATUS_LABEL[job.status]}
          </span>
        </div>

        {/* Progress bar for active states */}
        {isActive && (
          <div className="mt-4 space-y-2">
            <div className="h-1 rounded-full overflow-hidden" style={{ background: 'rgba(255,255,255,0.07)' }}>
              <div
                className="h-full rounded-full animate-pulse"
                style={{ width: '65%', background: 'var(--accent)' }}
              />
            </div>
            {progressNote && (
              <p className="text-xs" style={{ color: '#93C5FD' }}>{progressNote}</p>
            )}
          </div>
        )}

        {/* Error message */}
        {job.error && (
          <div
            className="mt-4 text-sm rounded-xl px-4 py-3"
            style={{ background: 'rgba(248,113,113,0.1)', color: '#FCA5A5', border: '1px solid rgba(248,113,113,0.2)' }}
          >
            {job.error}
          </div>
        )}
      </div>

      {/* Overlay video — full width, prominent */}
      {overlayArtifact?.url && (
        <div>
          <div className="flex items-center justify-between mb-3">
            <h2 className="text-xl font-bold text-white">Analysis video</h2>
            <a
              href={overlayArtifact.url}
              target="_blank"
              rel="noopener noreferrer"
              className="text-xs flex items-center gap-1.5 px-3 py-1.5 rounded-lg transition-colors"
              style={{ color: 'var(--accent)', background: 'var(--accent-dim)' }}
            >
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M18 13v6a2 2 0 01-2 2H5a2 2 0 01-2-2V8a2 2 0 012-2h6M15 3h6v6M10 14L21 3"/>
              </svg>
              Open full screen
            </a>
          </div>
          <div className="rounded-2xl overflow-hidden" style={{ background: '#000', border: '1px solid var(--border-subtle)' }}>
            <video
              src={overlayArtifact.url}
              controls
              playsInline
              className="w-full aspect-video bg-black"
            />
          </div>
        </div>
      )}

      {/* Cool-moment photos */}
      {coolMomentPhotos.length > 0 && (
        <div>
          <h2 className="text-xl font-bold text-white mb-4">
            Key moments
            <span className="ml-2 text-sm font-normal" style={{ color: 'rgba(255,255,255,0.35)' }}>
              {coolMomentPhotos.length} frames
            </span>
          </h2>
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
            {coolMomentPhotos.map((photo) => (
              <a
                key={photo.id}
                href={photo.url}
                target="_blank"
                rel="noopener noreferrer"
                className="group rounded-2xl overflow-hidden block"
                style={{ border: '1px solid var(--border-subtle)', background: 'var(--bg-surface)' }}
              >
                <div className="relative">
                  <img
                    src={photo.url}
                    alt={`Turn ${(photo.meta.turn_idx ?? 0) + 1} ${photo.meta.side ?? ''}`}
                    className="w-full aspect-video object-cover group-hover:scale-[1.02] transition-transform duration-300"
                  />
                </div>
                <div className="px-3 py-2.5 text-xs font-medium" style={{ color: 'rgba(255,255,255,0.5)' }}>
                  Turn {(photo.meta.turn_idx ?? 0) + 1}
                  {photo.meta.side ? ` · ${photo.meta.side}` : ''}
                  {photo.meta.timestamp_s != null
                    ? ` · ${Number(photo.meta.timestamp_s).toFixed(1)}s`
                    : ''}
                </div>
              </a>
            ))}
          </div>
        </div>
      )}

      {/* Peak pressure frames */}
      {peakFrames.length > 0 && (
        <div>
          <h2 className="text-xl font-bold text-white mb-4">
            Peak pressure frames
            <span className="ml-2 text-sm font-normal" style={{ color: 'rgba(255,255,255,0.35)' }}>
              {peakFrames.length} frames
            </span>
          </h2>
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
            {peakFrames.map((frame) => (
              <a
                key={frame.id}
                href={frame.url}
                target="_blank"
                rel="noopener noreferrer"
                className="group rounded-2xl overflow-hidden block"
                style={{ border: '1px solid var(--border-subtle)', background: 'var(--bg-surface)' }}
              >
                <img
                  src={frame.url}
                  alt={`Turn ${(frame.meta.turn_idx ?? 0) + 1} ${frame.meta.side ?? ''}`}
                  className="w-full aspect-video object-cover group-hover:scale-[1.02] transition-transform duration-300"
                />
                <div className="px-3 py-2.5 text-xs font-medium" style={{ color: 'rgba(255,255,255,0.5)' }}>
                  Turn {(frame.meta.turn_idx ?? 0) + 1}
                  {frame.meta.side ? ` · ${frame.meta.side}` : ''}
                  {frame.meta.timestamp_s != null
                    ? ` · ${Number(frame.meta.timestamp_s).toFixed(1)}s`
                    : ''}
                  {frame.kind === 'peak_pressure_frame_enhanced'
                    ? <span style={{ color: 'var(--accent)' }}> · enhanced</span>
                    : ''}
                </div>
              </a>
            ))}
          </div>
        </div>
      )}

      {/* Downloads */}
      {(summaryArtifact || metricsArtifact) && (
        <div
          className="card p-5 flex flex-wrap gap-3"
        >
          <p className="w-full text-sm font-semibold text-white mb-1">Downloads</p>
          {summaryArtifact && (
            <a
              href={summaryArtifact.url}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-2 text-xs px-3.5 py-2 rounded-lg font-medium transition-colors"
              style={{ background: 'rgba(255,255,255,0.07)', color: 'rgba(255,255,255,0.7)' }}
            >
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4M7 10l5 5 5-5M12 15V3"/>
              </svg>
              Summary JSON
            </a>
          )}
          {metricsArtifact && (
            <a
              href={metricsArtifact.url}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-2 text-xs px-3.5 py-2 rounded-lg font-medium transition-colors"
              style={{ background: 'rgba(255,255,255,0.07)', color: 'rgba(255,255,255,0.7)' }}
            >
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4M7 10l5 5 5-5M12 15V3"/>
              </svg>
              Metrics CSV
            </a>
          )}
        </div>
      )}
    </div>
  )
}
