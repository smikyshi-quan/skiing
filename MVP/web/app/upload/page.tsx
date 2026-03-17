'use client'

import { useState, useRef } from 'react'
import { useRouter } from 'next/navigation'
import { createClient } from '@/lib/supabase/client'

type Step = 'idle' | 'creating' | 'uploading' | 'finalizing' | 'done' | 'error'

const PROGRESS: Record<Step, number> = {
  idle: 0,
  creating: 15,
  uploading: 60,
  finalizing: 90,
  done: 100,
  error: 0,
}

const LABEL: Record<Step, string> = {
  idle: '',
  creating: 'Creating job…',
  uploading: 'Uploading video…',
  finalizing: 'Queuing analysis…',
  done: 'Queued! Opening run…',
  error: '',
}

export default function UploadPage() {
  const router = useRouter()
  const supabase = createClient()
  const fileRef = useRef<HTMLInputElement>(null)

  const [file, setFile] = useState<File | null>(null)
  const [step, setStep] = useState<Step>('idle')
  const [error, setError] = useState<string | null>(null)
  const [dragOver, setDragOver] = useState(false)

  async function handleUpload(e: React.FormEvent) {
    e.preventDefault()
    if (!file) return
    setError(null)

    try {
      setStep('creating')
      const createRes = await fetch('/api/jobs/create', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ filename: file.name }),
      })
      if (!createRes.ok) {
        const { error: msg } = await createRes.json()
        throw new Error(msg ?? 'Failed to create job')
      }
      const { jobId, path, token } = await createRes.json()

      setStep('uploading')
      const { error: uploadError } = await supabase.storage
        .from('videos')
        .uploadToSignedUrl(path, token, file)
      if (uploadError) throw uploadError

      setStep('finalizing')
      const markRes = await fetch('/api/jobs/mark-uploaded', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ jobId }),
      })
      if (!markRes.ok) {
        const { error: msg } = await markRes.json()
        throw new Error(msg ?? 'Failed to queue job')
      }

      setStep('done')
      setTimeout(() => router.push(`/jobs/${jobId}`), 800)
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : String(err))
      setStep('error')
    }
  }

  function handleFilePick(f: File | null) {
    if (!f) return
    if (f.size > 50 * 1024 * 1024) {
      setError(
        `File is ${(f.size / 1024 / 1024).toFixed(0)} MB — max 50 MB. ` +
        `Compress with iMovie (File → Share → File, lower quality) or Handbrake first.`
      )
      setFile(null)
      setStep('idle')
      if (fileRef.current) fileRef.current.value = ''
      return
    }
    setFile(f)
    setStep('idle')
    setError(null)
  }

  const busy = step !== 'idle' && step !== 'done' && step !== 'error'

  return (
    <div className="max-w-lg">
      {/* Page header */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-white tracking-tight">Analyse a run</h1>
        <p className="mt-2 text-sm" style={{ color: 'rgba(255,255,255,0.4)' }}>
          Upload your ski video and get AI-powered technique analysis
        </p>
      </div>

      <form onSubmit={handleUpload} className="space-y-5">
        {/* Drop zone */}
        <div
          onClick={() => !busy && fileRef.current?.click()}
          onDragOver={e => { e.preventDefault(); setDragOver(true) }}
          onDragLeave={() => setDragOver(false)}
          onDrop={e => {
            e.preventDefault()
            setDragOver(false)
            handleFilePick(e.dataTransfer.files?.[0] ?? null)
          }}
          className="relative rounded-2xl p-10 text-center cursor-pointer select-none transition-all"
          style={{
            border: `2px dashed ${dragOver ? 'var(--accent)' : file ? 'rgba(79,142,255,0.5)' : 'rgba(255,255,255,0.12)'}`,
            background: dragOver
              ? 'var(--accent-dim)'
              : file
              ? 'rgba(79,142,255,0.06)'
              : 'var(--bg-surface)',
          }}
        >
          {file ? (
            <div className="space-y-1">
              {/* Video icon */}
              <div className="flex justify-center mb-3">
                <div
                  className="w-12 h-12 rounded-xl flex items-center justify-center"
                  style={{ background: 'var(--accent-dim)' }}
                >
                  <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="var(--accent)" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M15 10l4.553-2.069A1 1 0 0121 8.87v6.26a1 1 0 01-1.447.894L15 14M3 8a2 2 0 012-2h8a2 2 0 012 2v8a2 2 0 01-2 2H5a2 2 0 01-2-2V8z"/>
                  </svg>
                </div>
              </div>
              <p className="text-sm font-semibold text-white truncate px-4">{file.name}</p>
              <p className="text-xs" style={{ color: 'rgba(255,255,255,0.35)' }}>
                {(file.size / 1024 / 1024).toFixed(1)} MB
              </p>
              <button
                type="button"
                onClick={e => { e.stopPropagation(); setFile(null); setStep('idle'); if (fileRef.current) fileRef.current.value = '' }}
                className="mt-3 text-xs px-3 py-1 rounded-lg"
                style={{ color: 'rgba(255,255,255,0.4)', background: 'rgba(255,255,255,0.07)' }}
              >
                Change file
              </button>
            </div>
          ) : (
            <div className="space-y-3">
              <div className="flex justify-center">
                <div
                  className="w-14 h-14 rounded-2xl flex items-center justify-center"
                  style={{ background: 'rgba(255,255,255,0.05)' }}
                >
                  <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="rgba(255,255,255,0.3)" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4M17 8l-5-5-5 5M12 3v12"/>
                  </svg>
                </div>
              </div>
              <div>
                <p className="text-sm font-medium text-white">Drop your video here</p>
                <p className="text-xs mt-1" style={{ color: 'rgba(255,255,255,0.3)' }}>
                  or click to browse — MP4, MOV, AVI · Max 50 MB
                </p>
              </div>
            </div>
          )}

          <input
            ref={fileRef}
            type="file"
            accept="video/*"
            className="hidden"
            onChange={e => handleFilePick(e.target.files?.[0] ?? null)}
          />
        </div>

        {/* Progress bar */}
        {(busy || step === 'done') && (
          <div className="space-y-2">
            <div className="h-1.5 rounded-full overflow-hidden" style={{ background: 'rgba(255,255,255,0.08)' }}>
              <div
                className="h-full rounded-full transition-all duration-500"
                style={{ width: `${PROGRESS[step]}%`, background: 'var(--accent)' }}
              />
            </div>
            <p className="text-xs" style={{ color: 'rgba(255,255,255,0.4)' }}>{LABEL[step]}</p>
          </div>
        )}

        {/* Error */}
        {error && (
          <div
            className="text-sm rounded-xl px-4 py-3"
            style={{
              background: 'rgba(239,68,68,0.1)',
              color: '#F87171',
              border: '1px solid rgba(239,68,68,0.25)',
            }}
          >
            {error}
          </div>
        )}

        {/* CTA */}
        <button
          type="submit"
          disabled={!file || busy || step === 'done'}
          className="btn-primary w-full text-base"
          style={{ padding: '0.875rem 1.25rem', fontSize: '0.9375rem' }}
        >
          {busy ? LABEL[step] : 'Start analysis'}
        </button>
      </form>
    </div>
  )
}
