import { call } from "./api";

export type TrainingState = "idle" | "running" | "cancelling" | "done" | "failed" | "cancelled" | "interrupted";
export type TrainingStage = "prepare" | "preprocess" | "pitch" | "features" | "train" | "index" | "export";
export type TrainingStatus = {
  state: TrainingState;
  voice_id: string | null;
  stage: TrainingStage | null;
  stage_index: number | null;
  stages: number;
  epoch: number | null;
  epochs: number | null;
  /** 0 to 100. */
  percent: number;
  /** Null while the backend cannot estimate yet. */
  eta_seconds: number | null;
  message: string | null;
  error: string | null;
  started: number | null;
  finished: number | null;
  /** Decided by the backend: this voice has a stopped run that Resume can continue. */
  can_resume: boolean;
  /** Whether that resume continues from a saved checkpoint; without one it keeps the prepared audio and trains from epoch 1. */
  has_checkpoint: boolean;
};
export type TrainingEnvironment = {
  ready: boolean;
  ok: boolean;
  checks: { id: string; ok: boolean; message: string }[];
  /** The backend has not checked yet and may not start a GPU check now (live mode or training is running). */
  unknown?: boolean;
  /** This app has no such part at all (the installed app): the one check carries the plain message, and there is
   * nothing to set up or check again. */
  unavailable?: boolean;
  /** The answer is what was known before: no check was run because live mode is on. */
  checked_while_live?: boolean;
  /** Likewise, because a training run is active. */
  checked_while_training?: boolean;
};
export type StartTraining = { voice_id: string; epochs?: number; batch_size?: number; resume?: boolean; minutes?: number };
/** What a run limited to some minutes would train on (GET /training/plan). */
export type TrainingPlan = {
  clip_count: number;
  audio_minutes: number;
  available_minutes: number;
  epochs: number;
  estimated_minutes: number;
};

export const trainingApi = {
  /** With a voice id: that voice's latest job. Without: the running job, else the most recently finished one. */
  status: (voiceId?: string) =>
    call<TrainingStatus>("GET", `/training/status${voiceId === undefined ? "" : `?voice_id=${encodeURIComponent(voiceId)}`}`),
  environment: (refresh = false) => call<TrainingEnvironment>("GET", `/training/environment${refresh ? "?refresh=true" : ""}`),
  start: (req: StartTraining) => call<TrainingStatus>("POST", "/training/start", req),
  plan: (voiceId: string, minutes: number) =>
    call<TrainingPlan>("GET", `/training/plan?voice_id=${encodeURIComponent(voiceId)}&minutes=${minutes}`),
  cancel: () => call<TrainingStatus>("POST", "/training/cancel"),
};

export const isActive = (s: TrainingStatus | null | undefined) => s?.state === "running" || s?.state === "cancelling";
