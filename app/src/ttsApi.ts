import { call } from "./api";

export const MAX_TEXT = 5000;

export type TtsPreset = "fast" | "balanced" | "best";
export type TtsStatus = {
  /** "starting" is the worker launching (about 15 s, the first time about 35 s): waiting, not an error. */
  engine: "stopped" | "starting" | "ready" | "failed";
  state: "idle" | "preparing" | "speaking" | "stopping";
  voice_id: string | null;
  preset: string | null;
  /** The sentence being spoken, counting from 1, and how many there are. */
  piece: number;
  pieces: number;
  /** When idle: how the last request ended. Kept until the next request starts. */
  error: string | null;
  saved_path: string | null;
  started: number | null;
};
export type TtsEnvironment = {
  ready: boolean;
  ok: boolean;
  checks: { id: string; ok: boolean; message: string }[];
  /** Nothing has been checked yet and the backend may not run a check now. */
  unknown?: boolean;
  checked_while_live?: boolean;
  checked_while_training?: boolean;
};
export type TtsVoice = {
  id: string;
  name: string;
  /** A trained model exists, so the Best preset is available. */
  has_model: boolean;
  reference: { ok: boolean; seconds: number | null; message: string };
};
export type SpeakRequest = {
  voice_id: string;
  text: string;
  preset: TtsPreset;
  mic_device: number | null;
  monitor_device: number | null;
  save: boolean;
};

export const ttsApi = {
  status: () => call<TtsStatus>("GET", "/tts/status"),
  environment: (refresh = false) => call<TtsEnvironment>("GET", `/tts/environment${refresh ? "?refresh=true" : ""}`),
  voices: () => call<TtsVoice[]>("GET", "/tts/voices"),
  /** Refusals (409 something else is active, 400 anything else) arrive as an Error with the backend's plain message. */
  speak: (req: SpeakRequest) => call<TtsStatus>("POST", "/tts/speak", req),
  stop: () => call<TtsStatus>("POST", "/tts/stop"),
};

/** A request is in progress, or the worker is still launching for one. */
export const isBusy = (s: TtsStatus | null | undefined) => !!s && (s.state !== "idle" || s.engine === "starting");
