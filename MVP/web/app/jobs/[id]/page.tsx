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
  running:  'Analysing run...',
  done:     'Analysis complete',
  error:    'Analysis failed',
}

const STATUS_PILL: Record<JobStatus, string> = {
  created:  'bg-gray-100 text-gray-600',
  uploaded: 'bg-blue-50  text-blue-700',
  queued:   'bg-yellow-50 text-yellow-700',
  running:  'bg-blue-100 text-blue-800',
  done:     'bg-green-50 text-green-700',
  error:    'bg-red-50   text-red-700',
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
      <div>
        <p className="text-red-600 text-sm mb-2">{fetchError}</p>
        <Link href="/jobs" className="text-blue-600 text-sm hover:underline">
          Back to jobs
        </Link>
      </div>
    )
  }

  if (!data) {
    return <p className="text-sm text-gray-400 animate-pulse">Loading...</p>
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

  return (
    <div className="space-y-6">
      {/* Breadcrumb */}
      <div className="flex items-center gap-2 text-sm text-gray-400">
        <Link href="/jobs" className="hover:text-gray-700 transition-colors">
          Jobs
        </Link>
        <span>/</span>
        <span className="font-mono">{job.id.slice(0, 8)}</span>
      </div>

      {/* Status card */}
      <div className="bg-white border border-gray-200 rounded-xl p-5">
        <div className="flex items-start justify-between gap-4">
          <div>
            <p className="font-medium">{STATUS_LABEL[job.status]}</p>
            <p className="text-xs text-gray-400 mt-1">
              Updated {new Date(job.updated_at).toLocaleString()}
            </p>
          </div>
          <span
            className={`shrink-0 text-xs font-medium px-2.5 py-1 rounded-full ${
              STATUS_PILL[job.status]
            }`}
          >
            {job.status}
          </span>
        </div>

        {ACTIVE.has(job.status) && (
          <div className="mt-4">
            <div className="h-1 bg-gray-100 rounded-full overflow-hidden">
              <div className="h-full bg-blue-400 rounded-full animate-pulse w-3/4" />
            </div>
            {progressNote && (
              <p className="mt-2 text-xs text-blue-600">{progressNote}</p>
            )}
          </div>
        )}

        {job.error && (
          <p className="mt-3 text-sm text-red-600 bg-red-50 rounded-lg px-3 py-2.5">
            {job.error}
          </p>
        )}
      </div>

      {/* Overlay video */}
      {overlayArtifact?.url && (
        <div>
          <h2 className="text-lg font-semibold mb-3">
            Overlay video
          </h2>
          <div className="rounded-xl overflow-hidden border border-gray-200 bg-white">
            <video
              src={overlayArtifact.url}
              controls
              playsInline
              className="w-full aspect-video bg-black"
            />
          </div>
          <div className="mt-2">
            <a
              href={overlayArtifact.url}
              target="_blank"
              rel="noopener noreferrer"
              className="text-sm text-blue-600 hover:underline"
            >
              Open overlay video
            </a>
          </div>
        </div>
      )}

      {/* Cool-moment photos */}
      {coolMomentPhotos.length > 0 && (
        <div>
          <h2 className="text-lg font-semibold mb-3">
            Cool-moment photos ({coolMomentPhotos.length})
          </h2>
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
            {coolMomentPhotos.map((photo) => (
              <div
                key={photo.id}
                className="rounded-xl overflow-hidden border border-gray-200 bg-white"
              >
                <img
                  src={photo.url}
                  alt={`Turn ${(photo.meta.turn_idx ?? 0) + 1} ${photo.meta.side ?? ''}`}
                  className="w-full aspect-video object-cover"
                />
                <div className="px-3 py-2 text-xs text-gray-500">
                  Turn {(photo.meta.turn_idx ?? 0) + 1}
                  {photo.meta.side ? ` (${photo.meta.side})` : ''}
                  {photo.meta.timestamp_s != null
                    ? ` · ${Number(photo.meta.timestamp_s).toFixed(1)} s`
                    : ''}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Peak frames gallery (legacy) */}
      {peakFrames.length > 0 && (
        <div>
          <h2 className="text-lg font-semibold mb-3">
            Peak pressure frames ({peakFrames.length})
          </h2>
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
            {peakFrames.map((frame) => (
              <div
                key={frame.id}
                className="rounded-xl overflow-hidden border border-gray-200 bg-white"
              >
                <img
                  src={frame.url}
                  alt={`Turn ${(frame.meta.turn_idx ?? 0) + 1} ${frame.meta.side ?? ''}`}
                  className="w-full aspect-video object-cover"
                />
                <div className="px-3 py-2 text-xs text-gray-500">
                  Turn {(frame.meta.turn_idx ?? 0) + 1}
                  {frame.meta.side ? ` (${frame.meta.side})` : ''}
                  {frame.meta.timestamp_s != null
                    ? ` · ${Number(frame.meta.timestamp_s).toFixed(1)} s`
                    : ''}
                  {frame.kind === 'peak_pressure_frame_enhanced' ? ' · enhanced' : ''}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Downloads */}
      {(summaryArtifact || metricsArtifact) && (
        <div>
          {summaryArtifact && (
            <a
              href={summaryArtifact.url}
              target="_blank"
              rel="noopener noreferrer"
              className="text-sm text-blue-600 hover:underline"
            >
              Download analysis summary (JSON)
            </a>
          )}
          {metricsArtifact && (
            <div className={summaryArtifact ? 'mt-2' : ''}>
              <a
                href={metricsArtifact.url}
                target="_blank"
                rel="noopener noreferrer"
                className="text-sm text-blue-600 hover:underline"
              >
                Download metrics (CSV)
              </a>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
