import { useEffect, useRef, useState } from "react";
import { api } from "./api";
import "./App.css";
import { LivePanel } from "./LivePanel";
import { SetupCheck } from "./SetupCheck";
import { useLiveShortcuts } from "./useLiveShortcuts";
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
  const [actionError, setActionError] = useState<string | null>(null);

  useLiveShortcuts(latest, setActionError);

  return (
    <>
      <SetupCheck />
      <LivePanel stats={stats} actionError={actionError} />
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
