import { invoke } from "@tauri-apps/api/core";

export type Device = { index: number; name: string; max_input: number; max_output: number };
export type Voice = { id: string; name: string; has_index: boolean };
export type Params = {
  chunk_ms: number;
  crossfade_ms: number;
  context_ms: number;
  f0_method: "dio" | "harvest";
  index_rate: number;
};
export type Check = { id: string; ok: boolean; message: string };
export type Mode = "converted" | "muted";
export type Stats = {
  running: boolean;
  mode: Mode | "off";
  error: string | null;
  infer_ms: number;
  latency_ms: number;
  in_level: number;
  out_level: number;
  underruns: number;
  slow_chunks: number;
  suggest_faster: boolean;
};
export type StartRequest = {
  input_device: number;
  mic_device: number;
  voice: string;
  monitor_device: number | null;
  monitor_gain: number;
  preset: string;
  overrides: Partial<Params> | null;
  pitch_shift: number;
};
export type SettingsPatch = { pitch_shift?: number; monitor_enabled?: boolean; monitor_gain?: number };

let port: number | null = null;

async function origin(): Promise<string> {
  if (port === null) port = await invoke<number>("backend_port");
  return `127.0.0.1:${port}`;
}

async function call<T>(method: string, path: string, body?: unknown): Promise<T> {
  const res = await fetch(`http://${await origin()}${path}`, {
    method,
    headers: body === undefined ? undefined : { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!res.ok) {
    const data = await res.json().catch(() => ({}));
    const detail = typeof data.detail === "string" ? data.detail : `Request failed (${res.status})`;
    throw new Error(detail);
  }
  return res.json() as Promise<T>;
}

export const api = {
  health: () => call<{ status: string }>("GET", "/health"),
  devices: () => call<{ devices: Device[]; virtual_mic: number | null }>("GET", "/devices"),
  voices: () => call<Voice[]>("GET", "/voices"),
  presets: () => call<Record<string, Params>>("GET", "/presets"),
  setupCheck: () => call<Check[]>("GET", "/setup-check"),
  start: (req: StartRequest) => call<Stats>("POST", "/live/start", req),
  stop: () => call<Stats>("POST", "/live/stop"),
  mode: (mode: Mode) => call<Stats>("POST", "/live/mode", { mode }),
  settings: (patch: SettingsPatch) => call<Stats>("PATCH", "/live/settings", patch),
  wsUrl: async () => `ws://${await origin()}/ws`,
};
