import { call, failure, origin } from "./api";

/** Audio from the backend as a URL the page can play (revoke it when done). */
async function audioUrl(path: string, body?: unknown): Promise<string> {
  const res = await fetch(`http://${await origin()}${path}`, body === undefined ? undefined : {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  });
  if (!res.ok) throw await failure(res);
  return URL.createObjectURL(await res.blob());
}

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
  /** The level of the test phrase, which set the input boost. */
  test_level_db?: number | null;
};

export type CalibrationStatus = {
  recording: boolean;
  seconds: number;
  level: number;
  /** Stay quiet, then say the test phrase; "review" (nothing recording): play it back and set the boost; then read. */
  phase: "quiet" | "testing" | "review" | "reading" | null;
  /** Where the reading starts in `seconds`: 0 when it is a take of its own after the review. */
  read_from?: number;
  /** The mic check waiting for the speaker to set the boost. */
  review?: MicCheck | null;
  test_seconds?: number;
  test_phrase?: string;
  /** Done can be pressed once this much has been read; the take ends by itself at total_seconds. */
  min_read_seconds?: number;
  quiet_seconds: number;
  total_seconds: number;
  text: string;
  error: string | null;
  profile: VoiceProfile | null;
  /** Whether the reading itself was kept (deep calibration converts it); older calibrations kept only numbers. */
  has_sample?: boolean;
};

/** A mic check as the microphone picked it up, with the boost suggested for it. */
export type MicCheck = {
  noise_db: number; test_level_db: number; test_peak: number; suggested_gain: number;
  min_gain: number; max_gain: number; target_db: number;
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
  /** The calibration reading turned out in a voice at these settings, to play. */
  preview: (body: { voice_id: string; quality: number; pitch_shift: number; likeness: number; advanced?: Record<string, unknown> }) =>
    audioUrl("/calibration/preview", body),
  /** The last mic check, as the microphone picked it up, to play. */
  testAudioUrl: (gain = 1) => audioUrl(`/calibration/test-audio?gain=${gain}`),
  /** Ends the mic check, for review. */
  check: () => call<CalibrationStatus>("POST", "/calibration/check"),
  /** Starts the reading with the boost chosen at the review. */
  read: (gain: number) => call<CalibrationStatus>("POST", "/calibration/read", { gain }),
};
