import { call, failure, origin } from "./api";

export type FeminizeSettings = { pitch_semitones: number; formant_shift: number; breathiness: number };
export const DEFAULT_FEMINIZE: FeminizeSettings = { pitch_semitones: 5, formant_shift: 1.15, breathiness: 0.15 };
export const DEFAULT_MASCULINIZE: FeminizeSettings = { pitch_semitones: -5, formant_shift: 0.88, breathiness: 0 };
/** The voice live mode offers for the feminized preview of the speaker's own voice, with no model behind it. */
export const PREVIEW_VOICE_ID = "@feminize";

export type FeminizeJob = {
  active: boolean;
  finished: boolean;
  cancelled: boolean;
  done: number;
  total: number;
  source: string | null;
  voice_id: string | null;
  voice_name: string | null;
  error: string | null;
  skipped?: number;
};

const voicePath = (id: string) => `/library/voices/${encodeURIComponent(id)}`;

export const feminizeApi = {
  /** One of the voice's clips, feminized with these settings, as an object URL to play. */
  sampleUrl: async (voiceId: string, settings: FeminizeSettings): Promise<string> => {
    const res = await fetch(`http://${await origin()}${voicePath(voiceId)}/feminize/preview`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(settings),
    });
    if (!res.ok) throw await failure(res);
    return URL.createObjectURL(await res.blob());
  },
  start: (voiceId: string, settings: FeminizeSettings) =>
    call<FeminizeJob>("POST", `${voicePath(voiceId)}/feminize`, settings),
  status: () => call<FeminizeJob>("GET", "/library/feminize/status"),
  cancel: () => call<FeminizeJob>("POST", "/library/feminize/cancel"),
  settings: () => call<FeminizeSettings>("GET", "/library/feminize/settings"),
  saveSettings: (settings: FeminizeSettings) => call<FeminizeSettings>("PUT", "/library/feminize/settings", settings),
};
