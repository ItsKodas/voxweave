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
  /** Decided by the backend: this voice has a stopped run with a checkpoint that can continue. */
  can_resume: boolean;
};
export type TrainingEnvironment = {
  ready: boolean;
  ok: boolean;
  checks: { id: string; ok: boolean; message: string }[];
};
export type StartTraining = { voice_id: string; epochs?: number; batch_size?: number; resume?: boolean };

export const trainingApi = {
  /** With a voice id: that voice's latest job. Without: the running job, else the most recently finished one. */
  status: (voiceId?: string) =>
    call<TrainingStatus>("GET", `/training/status${voiceId === undefined ? "" : `?voice_id=${encodeURIComponent(voiceId)}`}`),
  environment: (refresh = false) => call<TrainingEnvironment>("GET", `/training/environment${refresh ? "?refresh=true" : ""}`),
  start: (req: StartTraining) => call<TrainingStatus>("POST", "/training/start", req),
  cancel: () => call<TrainingStatus>("POST", "/training/cancel"),
};

export const isActive = (s: TrainingStatus | null | undefined) => s?.state === "running" || s?.state === "cancelling";
