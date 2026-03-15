export type JobStatus = 'created' | 'uploaded' | 'queued' | 'running' | 'done' | 'error'

export interface Job {
  id: string
  user_id: string
  status: JobStatus
  video_object_path: string | null
  result_prefix: string | null
  config: Record<string, unknown>
  error: string | null
  created_at: string
  updated_at: string
}

export interface Artifact {
  id: string
  job_id: string
  kind: string
  object_path: string
  meta: {
    turn_idx?: number
    side?: string
    timestamp_s?: number
    [key: string]: unknown
  }
  created_at: string
}

export interface ArtifactWithUrl extends Artifact {
  url: string
}
