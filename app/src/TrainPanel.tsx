import { useCallback, useEffect, useRef, useState } from "react";
import { installedApp } from "./setupApi";
import { LibraryVoice } from "./studioApi";
import { TrainingEnvironment, TrainingStage, TrainingStatus, isActive, trainingApi, TrainingPlan } from "./trainingApi";
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
/** The earlier epochs a run also keeps a version of the voice at (as backend/aivoice/training.py snapshot_epochs). */
export function snapshotEpochs(epochs: number): number[] {
  if (epochs < 60) return [];
  const marks = new Set([0.5, 0.75].map((f) => Math.round((epochs * f) / 5) * 5));
  return [...marks].filter((e) => e > 0 && e < epochs).sort((a, b) => a - b);
}

/** Time limits offered for a quick run, in minutes. */
const TIME_LIMITS = [10, 20, 30, 60, 120];

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

/**
 * The last environment report, kept while the app is open: the check takes seconds, and choosing another voice
 * mounts a new panel. "Check again" replaces it. A report the backend could not make (unknown) is never kept.
 */
let lastEnvironment: TrainingEnvironment | null = null;

/** For tests: start without a remembered report. */
export const forgetEnvironment = () => {
  lastEnvironment = null;
};

function Bar({ status, label }: { status: TrainingStatus; label: string }) {
  return (
    <div className="train-bar">
      <progress max={100} value={status.percent} aria-label={label} />
      <span>{percentOf(status)}%</span>
    </div>
  );
}

export function TrainPanel({ voice, active, onModelReady, onStarted = () => {} }: {
  voice: LibraryVoice;
  /** Whether the Voices tab is visible: nothing is asked of the backend while it is not. */
  active: boolean;
  onModelReady: () => void;
  /** A training run was started or resumed here (the app then follows it on every tab). */
  onStarted?: () => void;
}) {
  const { status, elsewhere, unreachable, refresh } = useTrainingStatus(voice.id, active);
  const [env, setEnv] = useState<TrainingEnvironment | null>(lastEnvironment);
  const [envError, setEnvError] = useState<string | null>(null);
  const [checking, setChecking] = useState(false);
  const [epochs, setEpochs] = useState(DEFAULT_EPOCHS);
  // 0 trains on every clip for the chosen epochs; otherwise the run is planned to fit this many minutes.
  const [limit, setLimit] = useState(0);
  const [plan, setPlan] = useState<TrainingPlan | null>(null);
  const [planError, setPlanError] = useState<string | null>(null);
  useEffect(() => {
    setPlan(null);
    setPlanError(null);
    if (limit === 0) return;
    let stale = false;
    trainingApi.plan(voice.id, limit).then(
      (p) => !stale && setPlan(p),
      (e: Error) => !stale && setPlanError(e.message),
    );
    return () => {
      stale = true;
    };
  }, [limit, voice.id, voice.clips]);
  const [batch, setBatch] = useState(DEFAULT_BATCH);
  const [actionError, setActionError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [stopping, setStopping] = useState(false);
  const [sawFinish, setSawFinish] = useState(false);

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
    if (wasOwn.current && status?.state === "done") setSawFinish(true);
    if ((wasOwn.current && status?.state === "done") || (wasOther.current && elsewhere?.state === "done")) {
      notify.current();
    }
    wasOwn.current = ownActive;
    wasOther.current = otherActive;
  }, [status, elsewhere, ownActive, otherActive]);

  useEffect(() => {
    if (status !== null && !ownActive) setStopping(false);
  }, [status, ownActive]);

  // The environment check takes seconds, so it runs once, when it can matter, and its answer is remembered.
  const asked = useRef(lastEnvironment !== null);
  const checkEnvironment = useCallback(async (fresh: boolean) => {
    setChecking(true);
    setEnvError(null);
    try {
      const report = await trainingApi.environment(fresh);
      if (!report.unknown) lastEnvironment = report;
      setEnv(report);
    } catch (e) {
      setEnvError((e as Error).message);
    } finally {
      setChecking(false);
    }
  }, []);
  useEffect(() => {
    // The installed app cannot train at all, which is worth saying before anyone records five minutes for it;
    // there the question costs nothing. In the source checkout it starts a GPU check, so it waits for enough audio.
    if (!active || (!enough && !installedApp()) || status === null || ownActive || otherActive || asked.current) return;
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

  const start = async (resume: boolean, confirmText: string | null) => {
    if (confirmText !== null && !window.confirm(confirmText)) return;
    const started = await act(() => trainingApi.start(resume ? { voice_id: voice.id, resume: true }
      : limit > 0 ? { voice_id: voice.id, minutes: limit, batch_size: batch, resume: false }
      : { voice_id: voice.id, epochs, batch_size: batch, resume: false }));
    if (started) onStarted();
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

  if (status === null) {
    return (
      <div className="train">
        {alert}
        {unreachable && (
          <p role="alert" className="error">
            Could not reach the voice engine. Retrying… <button onClick={refresh}>Retry</button>
          </p>
        )}
      </div>
    );
  }

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
  const unchecked = env?.unknown === true; // the backend may not probe the GPU right now: not a broken setup
  const unavailable = env?.unavailable === true; // the installed app: nothing here can be set up or checked again
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
      {(hasModel || (status.state === "done" && !sawFinish)) && <p className="ok">Trained model: ready</p>}
      {status.state === "done" && sawFinish && <p className="ok">Training finished. This voice is now available on the Live tab.</p>}
      {status.state === "cancelled" && <p>Training was stopped</p>}
      {status.state === "interrupted" && <p>Training was interrupted when the app closed</p>}
      {status.state === "failed" && (
        <p className="hint">The details are in data/voices/{voice.id}/training/worker.log</p>
      )}

      {unavailable && failing.map((c) => <p key={c.id} className="hint">{c.message}</p>)}
      {!enough && !unavailable && <p className="hint">Record or import at least {MIN_TRAIN_MINUTES} minutes to train. You have {usable.toFixed(1)}.</p>}
      {enough && env === null && !envError && <p className="hint">Checking this computer…</p>}
      {enough && unchecked && (
        <p className="hint">
          {env?.checked_while_live ? "Training can be checked once live mode is stopped." : "Training cannot be checked right now."}
        </p>
      )}
      {enough && env !== null && !env.ready && !unchecked && !unavailable && (
        <div className="setup">
          <p>Training is not set up on this computer. See Training setup in the README.</p>
          <ul>{failing.map((c) => <li key={c.id}>{c.message}</li>)}</ul>
        </div>
      )}
      {enough && !unavailable && (envError !== null || (env !== null && !env.ready)) && (
        <button className="secondary" disabled={checking} onClick={() => checkEnvironment(true)}>Check again</button>
      )}

      {canStart && (
        <>
          <label>
            Train for
            <select value={limit} onChange={(e) => setLimit(Number(e.target.value))}>
              <option value={0}>All clips, a set number of epochs</option>
              {TIME_LIMITS.map((m) => <option key={m} value={m}>{m < 60 ? `${m} minutes` : `${m / 60} hour${m > 60 ? "s" : ""}`}, best clips first</option>)}
            </select>
          </label>
          {limit === 0 ? (
            <label>
              Epochs
              <select value={epochs} onChange={(e) => setEpochs(Number(e.target.value))}>
                {EPOCH_CHOICES.map((n) => <option key={n} value={n}>{n}</option>)}
              </select>
            </label>
          ) : planError ? (
            <p role="alert" className="error">{planError}</p>
          ) : plan === null ? (
            <p className="hint">Picking the best clips...</p>
          ) : (
            <p className="hint">
              Trains on the best {plan.audio_minutes} of {plan.available_minutes} minutes ({plan.clip_count} clips) for{" "}
              {plan.epochs} epochs: about {Math.round(plan.estimated_minutes)} minutes. Clips are chosen for clear,
              undistorted speech well above the background noise.
            </p>
          )}
          <details>
            <summary>Advanced</summary>
            <label>
              Batch size
              <select value={batch} onChange={(e) => setBatch(Number(e.target.value))}>
                {BATCH_CHOICES.map((n) => <option key={n} value={n}>{n}</option>)}
              </select>
            </label>
          </details>
          {limit === 0 && <p>Rough time: <strong>{estimate(epochs, voice.minutes)}</strong></p>}
          {(() => {
            const kept = snapshotEpochs(limit === 0 ? epochs : (plan?.epochs ?? 0));
            return kept.length > 0 && (limit === 0 || plan !== null) ? (
              <p className="hint">
                Also keeps the voice as it was at epochs {kept.join(" and ")}, listed on the Live tab as "{voice.name} (epoch{" "}
                {kept[0]})" and so on, so you can compare them by ear and keep the one that sounds best.
              </p>
            ) : null;
          })()}
          <div className="actions">
            {stopped && status.can_resume && (
              <button disabled={busy} onClick={() => start(true, null)}>Resume</button>
            )}
            <button className={stopped ? "secondary" : undefined} disabled={busy} onClick={() => start(false, startConfirm)}>
              {startLabel}
            </button>
          </div>
          {stopped && status.can_resume && !status.has_checkpoint && (
            <p className="hint">Resume keeps the prepared audio and starts training from the first epoch.</p>
          )}
        </>
      )}
    </div>
  );
}
