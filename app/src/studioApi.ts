import { call, failure, origin } from "./api";

export type LibraryVoice = {
  id: string;
  name: string;
  created: string;
  minutes: number;
  clips: number;
  /** Trained engines that can be used live, for example ["rvc"]. */
  models: string[];
};
export type Clip = {
  index: number;
  text: string;
  seconds: number;
  source: "recorded" | "imported";
  prompt: number | null;
  origin: string | null;
};
/** `index` is null once every sentence has been recorded. */
export type PromptKind = "sentences" | "paragraphs";
/** `number` is the position within the kind, from 1; `index` is what a recording is started with. */
export type Prompt = {
  index: number | null; text: string | null; total: number; recorded: number; kind?: PromptKind; number?: number | null;
};
export type TakeResult = { accepted: boolean; reason: string | null; message: string | null; clip: Clip | null };
export type ImportStatus = {
  active: boolean;
  finished: boolean;
  total_files: number;
  done_files: number;
  current: string | null;
  clips_added: number;
  skipped: number;
  errors: string[];
  /** The voice of the most recent job (null before any), and its number in start order (0 before any). */
  voice_id: string | null;
  job: number;
};
export type StudioStatus = { recording: boolean; level: number; seconds: number; full: boolean; import: ImportStatus };

const voicePath = (id: string) => `/library/voices/${encodeURIComponent(id)}`;

export const studioApi = {
  voices: () => call<LibraryVoice[]>("GET", "/library/voices"),
  createVoice: (name: string) => call<LibraryVoice>("POST", "/library/voices", { name }),
  deleteVoice: (id: string) => call<{ deleted: boolean }>("DELETE", voicePath(id)),
  clips: (id: string) => call<Clip[]>("GET", `${voicePath(id)}/clips`),
  editClip: (id: string, index: number, text: string) => call<Clip>("PATCH", `${voicePath(id)}/clips/${index}`, { text }),
  deleteClip: (id: string, index: number) => call<{ deleted: boolean }>("DELETE", `${voicePath(id)}/clips/${index}`),
  /**
   * Fetches the clip (fetch always names the app's origin, which the backend requires for /library) and returns
   * an object URL for playback. The caller owns it and must release it with URL.revokeObjectURL.
   */
  clipAudioUrl: async (id: string, index: number): Promise<string> => {
    const res = await fetch(`http://${await origin()}${voicePath(id)}/clips/${index}/audio`);
    if (!res.ok) throw await failure(res);
    return URL.createObjectURL(await res.blob());
  },
  prompt: (id: string, after?: number, kind: PromptKind = "sentences") =>
    call<Prompt>("GET", `${voicePath(id)}/prompt?kind=${kind}${after === undefined ? "" : `&after=${after}`}`),
  recordStart: (voiceId: string, inputDevice: number, prompt: number) =>
    call<StudioStatus>("POST", "/library/record/start", { voice_id: voiceId, input_device: inputDevice, prompt }),
  recordStop: () => call<TakeResult>("POST", "/library/record/stop"),
  recordCancel: () => call<StudioStatus>("POST", "/library/record/cancel"),
  importFiles: (id: string, paths: string[]) => call<StudioStatus>("POST", `${voicePath(id)}/import`, { paths }),
  importCancel: () => call<StudioStatus>("POST", "/library/import/cancel"),
  status: () => call<StudioStatus>("GET", "/library/status"),
};
