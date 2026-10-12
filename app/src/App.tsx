import { useEffect, useRef, useState } from "react";
import { api } from "./api";
import "./App.css";
import { LivePanel } from "./LivePanel";
import { SetupCheck } from "./SetupCheck";
import { useLiveShortcuts } from "./useLiveShortcuts";
import { useStats } from "./useStats";

const START_TIMEOUT_MS = 15_000;

/** Polls /health until it answers. `failed` turns on after 15 s of silence; polling continues. */
function useBackendReady(): { ready: boolean; failed: boolean } {
  const [ready, setReady] = useState(false);
  const [failed, setFailed] = useState(false);
  useEffect(() => {
    let stop = false;
    const giveUp = setTimeout(() => setFailed(true), START_TIMEOUT_MS);
    const poll = async () => {
      while (!stop) {
        try {
          await api.health();
          clearTimeout(giveUp);
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
      clearTimeout(giveUp);
    };
  }, []);
  return { ready, failed: failed && !ready };
}

function StartFailure() {
  return (
    <div role="alert" className="error">
      <p>The voice engine failed to start. The app keeps trying. The usual causes are:</p>
      <ul>
        <li>The backend environment is not set up. See Setup in the README.</li>
        <li>The file data/settings.json is broken. Fix or delete it.</li>
        <li>The GPU runtime (DirectML) is missing.</li>
      </ul>
      <p>The details are in the log file data/backend.log.</p>
    </div>
  );
}

function Main() {
  const stats = useStats();
  const latest = useRef(stats);
  latest.current = stats;
  const [actionError, setActionError] = useState<string | null>(null);
  const [devicesVersion, setDevicesVersion] = useState(0);

  useLiveShortcuts(latest, setActionError);

  return (
    <>
      <SetupCheck refreshKey={devicesVersion} />
      <LivePanel stats={stats} actionError={actionError} onDevicesRefreshed={() => setDevicesVersion((v) => v + 1)} />
    </>
  );
}

export default function App() {
  const { ready, failed } = useBackendReady();
  return (
    <main>
      <h1>AI Voice</h1>
      {ready ? <Main /> : failed ? <StartFailure /> : <p>Starting the voice engine…</p>}
      <footer>Ctrl+Alt+V mutes and unmutes while live.</footer>
    </main>
  );
}
