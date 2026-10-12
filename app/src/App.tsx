import { listen } from "@tauri-apps/api/event";
import { useEffect, useRef, useState } from "react";
import { api } from "./api";
import "./App.css";
import { LivePanel } from "./LivePanel";
import { SetupCheck } from "./SetupCheck";
import { useStats } from "./useStats";

function useBackendReady(): boolean {
  const [ready, setReady] = useState(false);
  useEffect(() => {
    let stop = false;
    const poll = async () => {
      while (!stop) {
        try {
          await api.health();
          setReady(true);
          return;
        } catch {
          await new Promise((r) => setTimeout(r, 300));
        }
      }
    };
    poll();
    return () => {
      stop = true;
    };
  }, []);
  return ready;
}

function Main() {
  const stats = useStats();
  const latest = useRef(stats);
  latest.current = stats;

  useEffect(() => {
    const toggle = listen("live-toggle", () => {
      const s = latest.current;
      if (s.running) api.mode(s.mode === "muted" ? "converted" : "muted").catch(() => undefined);
    });
    const mode = listen<string>("live-mode", (event) => {
      if (event.payload === "off") api.stop().catch(() => undefined);
      else if (latest.current.running) api.mode(event.payload as "converted" | "muted").catch(() => undefined);
    });
    return () => {
      toggle.then((off) => off());
      mode.then((off) => off());
    };
  }, []);

  return (
    <>
      <SetupCheck />
      <LivePanel stats={stats} />
    </>
  );
}

export default function App() {
  const ready = useBackendReady();
  return (
    <main>
      <h1>AI Voice</h1>
      {ready ? <Main /> : <p>Starting the voice engine…</p>}
      <footer>Ctrl+Alt+V mutes and unmutes while live.</footer>
    </main>
  );
}
