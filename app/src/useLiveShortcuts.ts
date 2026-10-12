import { listen } from "@tauri-apps/api/event";
import { MutableRefObject, useEffect, useRef } from "react";
import { api, Mode, Stats } from "./api";

/** Hotkey and tray actions. `onError` gets a message on failure and null after a success. */
export function useLiveShortcuts(
  stats: MutableRefObject<Stats>,
  onError: (message: string | null) => void,
) {
  const report = useRef(onError);
  report.current = onError;

  useEffect(() => {
    const act = (label: string, call: () => Promise<unknown>) =>
      call().then(
        () => report.current(null),
        (e: Error) => report.current(`Could not ${label}: ${e.message}`),
      );
    const setMode = (mode: Mode) => act(mode === "muted" ? "mute" : "unmute", () => api.mode(mode));

    const toggle = listen("live-toggle", () => {
      const s = stats.current;
      if (s.running) setMode(s.mode === "muted" ? "converted" : "muted");
    });
    const mode = listen<string>("live-mode", (event) => {
      if (event.payload === "off") act("stop", () => api.stop());
      else if (stats.current.running) setMode(event.payload as Mode);
    });
    return () => {
      toggle.then((off) => off());
      mode.then((off) => off());
    };
  }, [stats]);
}
