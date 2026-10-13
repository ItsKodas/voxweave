import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";

/** What the shell says about the installed app's first-run setup (src-tauri/src/setup.rs, SetupState). */
export type SetupState = {
  /** "dev" is the source checkout, which has no setup: its status is always "ready". */
  mode: "dev" | "installed";
  status: "ready" | "running" | "failed";
  step: string;
  /** 0 to 100. */
  percent: number;
  message: string;
  error: string | null;
  /** No setup has finished here before: this one downloads everything. False for the check after an update or a repair. */
  first_time: boolean;
  /** False when the app itself is incomplete (a file is missing): starting the setup again cannot help. */
  can_retry: boolean;
  /** Where the installed app keeps everything it writes; empty in the source checkout. */
  root: string;
  data_dir: string;
  setup_log: string;
};

export const setupApi = {
  state: () => invoke<SetupState>("setup_state"),
  /** Starts the setup again (Retry, or a repair). The shell ignores it while a setup is running. */
  start: () => invoke<SetupState>("start_setup"),
  /** Calls handler with every change of the setup. Resolves to the function that stops listening. */
  onProgress: (handler: (state: SetupState) => void) =>
    listen<SetupState>("setup-progress", (event) => handler(event.payload)),
};

let installed = false;

/** Whether this is the installed app, as far as the shell has said so. The panels that describe training and
 * text to speech use it; false until the shell has answered and in the source checkout. */
export const installedApp = () => installed;

export function rememberMode(state: SetupState | null) {
  installed = state?.mode === "installed";
}
