import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "./api";
import "./App.css";
import { FirstRun, useSetup } from "./FirstRun";
import { LivePanel } from "./LivePanel";
import { SetupCheck } from "./SetupCheck";
import { SetupState } from "./setupApi";
import { SpeakPanel } from "./SpeakPanel";
import { TitleBar } from "./TitleBar";
import { isActive, TrainingStatus } from "./trainingApi";
import { isBusy } from "./ttsApi";
import { useLiveShortcuts } from "./useLiveShortcuts";
import { useStats } from "./useStats";
import { useTtsStatus } from "./useTtsStatus";
import { useTrainingWatch } from "./useTrainingWatch";
import { VoicesPanel } from "./VoicesPanel";

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

/** The installed app has no README, checkout or script to point to: it names its own log and offers a repair. */
function InstalledStartFailure({ setup, onRepair }: { setup: SetupState; onRepair: () => void }) {
  return (
    <div role="alert" className="error">
      <p>The voice engine failed to start. The app keeps trying.</p>
      <p>
        If it stays like this, repair the installation: it checks the engine's files and downloads again what
        is missing or damaged. Your voices and settings are kept.
      </p>
      <p><button className="secondary" onClick={onRepair}>Repair the installation</button></p>
      <p>The details are in the log file {setup.data_dir}\backend.log</p>
    </div>
  );
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
  const [liveMonitor, setLiveMonitor] = useState<number | null>(null);
  const [liveMic, setLiveMic] = useState<number | null>(null);
  const [voicesVersion, setVoicesVersion] = useState(0);
  const [tab, setTab] = useState<"live" | "voices" | "speak">("live");
  const reloadLiveVoices = useCallback(() => setVoicesVersion((v) => v + 1), []);
  // Followed here, not in a tab: a model that finishes while the Live tab is shown must still appear there.
  const [training, setTraining] = useState<TrainingStatus | null>(null);
  const trainingStarted = useTrainingWatch(reloadLiveVoices, 1000, setTraining);

  // Followed here, not in the Speak tab: Live and Voices must know while speech is playing and their routes would refuse.
  const tts = useTtsStatus(tab === "speak");
  const speechActive = isBusy(tts.status);

  useLiveShortcuts(latest, setActionError);

  const showLive = () => {
    if (tab !== "live") reloadLiveVoices(); // whatever was trained or deleted meanwhile shows without a restart
    setTab("live");
  };

  const status = stats.running ? (
    <span className="pill pill-live">Live · {(stats.latency_ms / 1000).toFixed(1)} s behind</span>
  ) : training !== null && isActive(training) ? (
    <span className="pill busy">Training · {Math.round(training.percent ?? 0)}%</span>
  ) : null;

  return (
    <>
      <TitleBar status={status}>
        <div role="tablist" className="tabs">
          <button role="tab" aria-selected={tab === "live"} onClick={showLive}>Live</button>
          <button role="tab" aria-selected={tab === "voices"} onClick={() => setTab("voices")}>Voices</button>
          <button role="tab" aria-selected={tab === "speak"} onClick={() => setTab("speak")}>Speak</button>
        </div>
      </TitleBar>
      <div className="content">
      {/* All three stay mounted so switching tabs never resets a form or interrupts a session. */}
      <div role="tabpanel" hidden={tab !== "live"}>
        <SetupCheck refreshKey={devicesVersion} />
        <LivePanel stats={stats} actionError={actionError} voicesVersion={voicesVersion} speechActive={speechActive} onMonitorChosen={setLiveMonitor} onMicChosen={setLiveMic} onDevicesRefreshed={() => setDevicesVersion((v) => v + 1)} />
      </div>
      <div role="tabpanel" hidden={tab !== "voices"}>
        <VoicesPanel active={tab === "voices"} speechActive={speechActive} devicesVersion={devicesVersion} onModelReady={reloadLiveVoices} onTrainingStarted={trainingStarted} />
      </div>
      <div role="tabpanel" hidden={tab !== "speak"}>
        <SpeakPanel tts={tts} active={tab === "speak"} liveRunning={stats.running} devicesVersion={devicesVersion} preferredMonitor={liveMonitor} preferredMic={liveMic} />
      </div>
      </div>
    </>
  );
}

/** The window before the voice engine answers: the title bar, and the message below it. */
function Shell({ children }: { children: React.ReactNode }) {
  return (
    <>
      <TitleBar />
      <div className="content">{children}</div>
    </>
  );
}

function Engine({ setup, onRepair }: { setup: SetupState | null; onRepair: () => void }) {
  const { ready, failed } = useBackendReady();
  if (ready) return <Main />;
  if (!failed) return <Shell><p className="starting">Starting the voice engine…</p></Shell>;
  return (
    <Shell>
      {setup?.mode === "installed" ? <InstalledStartFailure setup={setup} onRepair={onRepair} /> : <StartFailure />}
    </Shell>
  );
}

export default function App() {
  const { setup, retry } = useSetup();
  // Only the installed app has a setup. In the source checkout, and until the shell has answered, the app
  // starts as it always did.
  const settingUp = setup !== null && setup.mode === "installed" && setup.status !== "ready";
  return (
    <div className="app">
      {settingUp ? <Shell><FirstRun setup={setup} onRetry={retry} /></Shell> : <Engine setup={setup} onRepair={retry} />}
    </div>
  );
}
