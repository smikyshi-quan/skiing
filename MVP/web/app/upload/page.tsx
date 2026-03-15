'use client'

import { useState, useRef } from 'react'
import { useRouter } from 'next/navigation'
import { createClient } from '@/lib/supabase/client'

type Step = 'idle' | 'creating' | 'uploading' | 'finalizing' | 'done' | 'error'

export default function UploadPage() {
  const router = useRouter()
  const supabase = createClient()
  const fileRef = useRef<HTMLInputElement>(null)

  const [file, setFile] = useState<File | null>(null)
  const [step, setStep] = useState<Step>('idle')
  const [error, setError] = useState<string | null>(null)

  async function handleUpload(e: React.FormEvent) {
    e.preventDefault()
    if (!file) return
    setError(null)

    try {
      // 1. Create job row + get signed upload URL (service role on server)
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

      // 2. Upload video directly to Supabase Storage via signed URL
      setStep('uploading')
      const { error: uploadError } = await supabase.storage
        .from('videos')
        .uploadToSignedUrl(path, token, file)
      if (uploadError) throw uploadError

      // 3. Mark job queued so the worker picks it up
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
    creating: 'Creating job...',
    uploading: 'Uploading video...',
    finalizing: 'Queuing analysis...',
    done: 'Queued! Opening job...',
    error: '',
  }

  const busy = step !== 'idle' && step !== 'done' && step !== 'error'

  return (
    <div className="max-w-lg">
      <h1 className="text-2xl font-bold mb-6">Analyse a run</h1>

      <form onSubmit={handleUpload} className="space-y-5">
        <div
          onClick={() => fileRef.current?.click()}
          className="border-2 border-dashed border-gray-300 rounded-xl p-10 text-center cursor-pointer hover:border-blue-400 transition-colors select-none"
        >
          {file ? (
            <p className="text-sm font-medium text-gray-800 truncate">{file.name}</p>
          ) : (
            <p className="text-sm text-gray-400">Click to select a video file</p>
          )}
          <input
            ref={fileRef}
            type="file"
            accept="video/*"
            className="hidden"
            onChange={(e) => {
              setFile(e.target.files?.[0] ?? null)
              setStep('idle')
              setError(null)
            }}
          />
        </div>

        {(busy || step === 'done') && (
          <div className="space-y-1.5">
            <div className="h-1.5 bg-gray-200 rounded-full overflow-hidden">
              <div
                className="h-full bg-blue-500 rounded-full transition-all duration-500"
                style={{ width: `${PROGRESS[step]}%` }}
              />
            </div>
            <p className="text-xs text-gray-500">{LABEL[step]}</p>
          </div>
        )}

        {error && (
          <p className="text-sm text-red-600 bg-red-50 border border-red-200 rounded-md px-3 py-2">
            {error}
          </p>
        )}

        <button
          type="submit"
          disabled={!file || busy || step === 'done'}
          className="w-full bg-blue-600 text-white rounded-md px-4 py-2.5 text-sm font-medium hover:bg-blue-700 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
        >
          {busy ? LABEL[step] : 'Analyse'}
        </button>
      </form>
    </div>
  )
}
