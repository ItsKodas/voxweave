import { call } from "./api";

/** What a calibration recording measured about the person speaking. */
export type VoiceProfile = {
  pitch_hz: number;
  pitch_low_hz: number;
  pitch_high_hz: number;
  level_db: number;
  noise_db: number;
  gate_db: number;
  input_gain: number;
  device: string;
  created: number;
};

export type CalibrationStatus = {
  recording: boolean;
  seconds: number;
  level: number;
  /** While recording: first stay quiet, then read the text aloud. */
  phase: "quiet" | "reading" | null;
  quiet_seconds: number;
  total_seconds: number;
  text: string;
  error: string | null;
  profile: VoiceProfile | null;
};

/** The pitch shift that moves the speaker onto a target voice; null with a reason when it cannot be worked out. */
export type PitchSuggestion = {
  pitch_shift: number | null;
  reason: string | null;
  speaker_hz?: number;
  target_hz?: number;
};

export const calibrationApi = {
  status: () => call<CalibrationStatus>("GET", "/calibration/status"),
  start: (inputDevice: number) => call<CalibrationStatus>("POST", "/calibration/start", { input_device: inputDevice }),
  finish: () => call<CalibrationStatus>("POST", "/calibration/finish"),
  cancel: () => call<CalibrationStatus>("POST", "/calibration/cancel"),
  forget: () => call<CalibrationStatus>("DELETE", "/calibration/profile"),
  suggestion: (voice: string) => call<PitchSuggestion>("GET", `/calibration/suggestion?voice=${encodeURIComponent(voice)}`),
};
