import { useCallback, useEffect, useRef, useState } from "react";
import { LibraryVoice } from "./studioApi";
import { TrainingEnvironment, TrainingStage, TrainingStatus, isActive, trainingApi } from "./trainingApi";
import { useTrainingStatus } from "./useTrainingStatus";

export const MIN_TRAIN_MINUTES = 5;
const EPOCH_CHOICES = [100, 200, 300];
const BATCH_CHOICES = [4, 8, 16];
const DEFAULT_EPOCHS = 200;
const DEFAULT_BATCH = 8;
const SECONDS_PER_EPOCH_PER_MINUTE = 0.95;

const STAGE_NAMES: Record<TrainingStage, string> = {
  prepare: "Preparing",
  preprocess: "Slicing audio",
  pitch: "Finding pitch",
  features: "Extracting features",
  train: "Training",
  index: "Building index",
  export: "Exporting",
};

/** Rough wall-clock time for a run, rounded up to the next 5 minutes. */
export function estimate(epochs: number, minutesOfAudio: number): string {
  const minutes = (epochs * minutesOfAudio * SECONDS_PER_EPOCH_PER_MINUTE) / 60;
  return `About ${Math.max(5, Math.ceil(minutes / 5) * 5)} minutes`;
}

export function timeLeft(seconds: number | null): string {
  if (seconds === null) return "Estimating time…";
  if (seconds < 60) return "Less than a minute left";
  const minutes = Math.round(seconds / 60);
  if (minutes < 60) return `About ${minutes} ${minutes === 1 ? "minute" : "minutes"} left`;
  return `About ${Math.floor(minutes / 60)} h ${minutes % 60} min left`;
}

const percentOf = (s: TrainingStatus) => Math.round(s.percent);

function Bar({ status, label }: { status: TrainingStatus; label: string }) {
  return (
    <div className="train-bar">
      <progress max={100} value={status.percent} aria-label={label} />
      <span>{percentOf(status)}%</span>
    </div>
  );
}

export function TrainPanel({ voice, active, onModelReady }: {
  voice: LibraryVoice;
  /** Whether the Voices tab is visible: nothing is asked of the backend while it is not. */
  active: boolean;
  onModelReady: () => void;
}) {
  const { status, elsewhere, refresh } = useTrainingStatus(voice.id, active);
  const [env, setEnv] = useState<TrainingEnvironment | null>(null);
  const [envError, setEnvError] = useState<string | null>(null);
  const [checking, setChecking] = useState(false);
  const [epochs, setEpochs] = useState(DEFAULT_EPOCHS);
  const [batch, setBatch] = useState(DEFAULT_BATCH);
  const [actionError, setActionError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [stopping, setStopping] = useState(false);

  const enough = voice.minutes >= MIN_TRAIN_MINUTES;
  const ownActive = isActive(status);
  const otherActive = isActive(elsewhere);
  const hasModel = voice.models.length > 0;

  // Tell the Live tab when a job this panel watched running has finished (this voice's or another's).
  const notify = useRef(onModelReady);
  notify.current = onModelReady;
  const wasOwn = useRef(false);
  const wasOther = useRef(false);
  useEffect(() => {
    if ((wasOwn.current && status?.state === "done") || (wasOther.current && elsewhere?.state === "done")) {
      notify.current();
    }
    wasOwn.current = ownActive;
    wasOther.current = otherActive;
  }, [status, elsewhere, ownActive, otherActive]);

  useEffect(() => {
    if (status !== null && !ownActive) setStopping(false);
  }, [status, ownActive]);

  // The environment check takes seconds, so it runs once, when it can matter.
  const asked = useRef(false);
  const checkEnvironment = useCallback(async (fresh: boolean) => {
    setChecking(true);
    setEnvError(null);
    try {
      setEnv(await trainingApi.environment(fresh));
    } catch (e) {
      setEnvError((e as Error).message);
    } finally {
      setChecking(false);
    }
  }, []);
  useEffect(() => {
    if (!active || !enough || status === null || ownActive || otherActive || asked.current) return;
    asked.current = true;
    void checkEnvironment(false);
  }, [active, enough, status, ownActive, otherActive, checkEnvironment]);

  const act = async (action: () => Promise<unknown>) => {
    setBusy(true);
    setActionError(null);
    try {
      await action();
      return true;
    } catch (e) {
      setActionError((e as Error).message);
      return false;
    } finally {
      setBusy(false);
      refresh();
    }
  };

  const start = (resume: boolean, confirmText: string | null) => {
    if (confirmText !== null && !window.confirm(confirmText)) return;
    void act(() => trainingApi.start(resume ? { voice_id: voice.id, resume: true } : { voice_id: voice.id, epochs, batch_size: batch, resume: false }));
  };

  const cancel = async () => {
    if (!window.confirm("Stop training? You can resume it later.")) return;
    if (await act(() => trainingApi.cancel())) setStopping(true);
  };

  const alerts = [actionError, status?.state === "failed" ? (status.error ?? "Training failed") : null, envError]
    .filter((m): m is string => m !== null);
  const alert = alerts.length > 0 && (
    <div role="alert" className="error">
      {alerts.map((m) => <p key={m}>{m}</p>)}
    </div>
  );

  if (status === null) return <div className="train">{alert}</div>;

  if (ownActive) {
    const stage = status.stage ? STAGE_NAMES[status.stage] : "Preparing";
    const showEpoch = status.stage === "train" && status.epoch !== null && status.epochs !== null;
    return (
      <div className="train">
        {alert}
        <p className="train-stage">{stage}</p>
        {showEpoch && <p>Epoch {status.epoch} of {status.epochs}</p>}
        <Bar status={status} label="Training progress" />
        {stopping || status.state === "cancelling" ? (
          <p>Stopping…</p>
        ) : (
          <>
            <p>{timeLeft(status.eta_seconds)}</p>
            <button className="secondary" disabled={busy} onClick={cancel}>Cancel</button>
          </>
        )}
        <p className="hint">Live mode and recording are unavailable until training finishes.</p>
      </div>
    );
  }

  if (otherActive && elsewhere) {
    return (
      <div className="train">
        {alert}
        <p>Another voice is training</p>
        <Bar status={elsewhere} label="Other voice training progress" />
        <p className="hint">Live mode and recording are unavailable until training finishes.</p>
      </div>
    );
  }

  const stopped = status.state === "failed" || status.state === "cancelled" || status.state === "interrupted";
  const canStart = enough && env?.ready === true;
  const failing = env?.checks.filter((c) => !c.ok) ?? [];
  const startLabel = stopped ? "Start over" : hasModel ? "Train again" : "Start training";
  const startConfirm = stopped
    ? "Start over? The progress so far will be discarded."
    : hasModel
      ? "Train again? The current model is replaced only when the new training finishes."
      : null;
  const usable = Math.floor(voice.minutes * 10 + 1e-9) / 10;

  return (
    <div className="train">
      {alert}
      {hasModel && <p className="ok">Trained model: ready</p>}
      {status.state === "done" && <p className="ok">Training finished. This voice is now available on the Live tab.</p>}
      {status.state === "cancelled" && <p>Training was stopped</p>}
      {status.state === "interrupted" && <p>Training was interrupted when the app closed</p>}
      {status.state === "failed" && (
        <p className="hint">The details are in data/voices/{voice.id}/training/worker.log</p>
      )}

      {!enough && <p className="hint">Record or import at least {MIN_TRAIN_MINUTES} minutes to train. You have {usable.toFixed(1)}.</p>}
      {enough && env === null && !envError && <p className="hint">Checking this computer…</p>}
      {enough && env !== null && !env.ready && (
        <div className="setup">
          <p>Training is not set up on this computer. See Training setup in the README.</p>
          <ul>{failing.map((c) => <li key={c.id}>{c.message}</li>)}</ul>
        </div>
      )}
      {enough && (envError !== null || (env !== null && !env.ready)) && (
        <button className="secondary" disabled={checking} onClick={() => checkEnvironment(true)}>Check again</button>
      )}

      {canStart && (
        <>
          <label>
            Epochs
            <select value={epochs} onChange={(e) => setEpochs(Number(e.target.value))}>
              {EPOCH_CHOICES.map((n) => <option key={n} value={n}>{n}</option>)}
            </select>
          </label>
          <details>
            <summary>Advanced</summary>
            <label>
              Batch size
              <select value={batch} onChange={(e) => setBatch(Number(e.target.value))}>
                {BATCH_CHOICES.map((n) => <option key={n} value={n}>{n}</option>)}
              </select>
            </label>
          </details>
          <p>Rough time: <strong>{estimate(epochs, voice.minutes)}</strong></p>
          <div className="actions">
            {stopped && status.can_resume && (
              <button disabled={busy} onClick={() => start(true, null)}>Resume</button>
            )}
            <button className={stopped ? "secondary" : undefined} disabled={busy} onClick={() => start(false, startConfirm)}>
              {startLabel}
            </button>
          </div>
        </>
      )}
    </div>
  );
}
