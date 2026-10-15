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
  /** Whether the reading itself was kept (deep calibration converts it); older calibrations kept only numbers. */
  has_sample?: boolean;
};

/** The speaker-recognition model deep calibration scores with; downloaded only when the person agrees. */
export type SpeakerModelInfo = { ready: boolean; size_mb: number; source: string; licence: string };

/** One setting deep calibration tried: how much it sounded like the voice, and how many words stayed clear. */
export type TunedSetting = {
  pitch_shift: number; likeness: number; similarity: number; clarity: number; quality: number;
  advanced?: Record<string, number | string | boolean>;
};
export type TuneResult = TunedSetting & {
  voice_id: string; quality: number; baseline: TunedSetting; tried: TunedSetting[]; created: number;
};
export type TuneStatus = {
  active: boolean; voice_id: string | null; step: number; steps: number; message: string | null;
  error: string | null; result: TuneResult | null;
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
  speakerModel: () => call<SpeakerModelInfo>("GET", "/calibration/speaker-model"),
  downloadSpeakerModel: () => call<SpeakerModelInfo>("POST", "/calibration/speaker-model/download"),
  tune: (body: { voice_id: string; quality: number; pitch_shift: number; likeness: number; advanced?: Record<string, unknown> }) =>
    call<TuneStatus>("POST", "/calibration/tune", body),
  tuneStatus: () => call<TuneStatus>("GET", "/calibration/tune/status"),
  tuneCancel: () => call<TuneStatus>("POST", "/calibration/tune/cancel"),
};
