import { useEffect, useRef, useState } from "react";
import { calibrationApi, CalibrationStatus, VoiceProfile } from "./calibrationApi";

const POLL_MS = 250;

/** How loud the microphone is right now, by the peak of the latest audio: red, green, yellow, red. */
export function levelZone(level: number): { label: string; zone: "quiet" | "good" | "loud" | "clip" } {
  if (level < 0.03) return { label: "Too quiet", zone: "quiet" };
  if (level < 0.5) return { label: "Good", zone: "good" };
  if (level < 0.9) return { label: "Loud", zone: "loud" };
  return { label: "Too loud", zone: "clip" };
}

function LevelBar({ level }: { level: number }) {
  const { label, zone } = levelZone(level);
  return (
    <div className="level-bar">
      <div className="level-track" aria-hidden="true">
        <div className={`level-fill ${zone}`} style={{ width: `${Math.min(100, Math.max(2, level * 100))}%` }} />
      </div>
      <span className={`level-label ${zone}`} aria-live="polite">{label}</span>
    </div>
  );
}

/** Plays a backend audio URL, revoking it when it ends. */
function play(url: string, onEnd: () => void): HTMLAudioElement {
  const audio = new Audio(url);
  const done = () => {
    URL.revokeObjectURL(url);
    onEnd();
  };
  audio.onended = done;
  audio.onerror = done;
  void audio.play().catch(done);
  return audio;
}

/**
 * "My voice": one short recording of the person speaking (a few seconds of quiet, then reading aloud), from which
 * live mode sets its pitch shift, a quiet gate and the input level. `input` is the microphone chosen on the Live tab.
 */
export function CalibrationPanel({ input, disabled, onProfile }: {
  input: number | null;
  /** Live mode is running or speech is playing: the microphone and the devices are in use. */
  disabled: boolean;
  /** Called with the profile whenever it is loaded, made or removed. */
  onProfile: (profile: VoiceProfile | null) => void;
}) {
  const [status, setStatus] = useState<CalibrationStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [playing, setPlaying] = useState(false);
  const finishing = useRef(false);
  const onProfileRef = useRef(onProfile);
  onProfileRef.current = onProfile;

  const take = (next: CalibrationStatus) => {
    setStatus(next);
    onProfileRef.current(next.profile);
  };

  useEffect(() => {
    let gone = false;
    calibrationApi.status().then((s) => !gone && take(s), () => undefined);
    return () => {
      gone = true;
    };
  }, []);

  const recording = status?.recording === true;

  // While recording: follow the take, and end it by itself when the time is up.
  useEffect(() => {
    if (!recording) return;
    let gone = false;
    const timer = setInterval(async () => {
      if (finishing.current) return;
      try {
        const now = await calibrationApi.status();
        if (gone) return;
        if (now.recording && now.seconds >= now.total_seconds) {
          finishing.current = true;
          try {
            take(await calibrationApi.finish());
            setError(null);
          } catch (e) {
            if (!gone) {
              setError((e as Error).message);
              calibrationApi.status().then((s) => !gone && take(s), () => undefined);
            }
          } finally {
            finishing.current = false;
          }
        } else {
          setStatus(now);
        }
      } catch {
        // the next tick asks again
      }
    }, POLL_MS);
    return () => {
      gone = true;
      clearInterval(timer);
    };
  }, [recording]);

  // Leaving the tab's component mid-take must not leave the microphone open.
  const recordingRef = useRef(recording);
  recordingRef.current = recording;
  useEffect(() => () => {
    if (recordingRef.current) calibrationApi.cancel().catch(() => undefined);
  }, []);

  const run = async (action: () => Promise<CalibrationStatus>) => {
    setBusy(true);
    setError(null);
    try {
      take(await action());
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const finishNow = () => {
    if (finishing.current) return;
    finishing.current = true;
    run(calibrationApi.finish).finally(() => {
      finishing.current = false;
    });
  };
  const readFrom = (status?.quiet_seconds ?? 3) + (status?.test_seconds ?? 4);
  const canFinish = !!status && status.seconds >= readFrom + (status.min_read_seconds ?? 8);
  const playCheck = async () => {
    setError(null);
    try {
      const url = await calibrationApi.testAudioUrl();
      setPlaying(true);
      play(url, () => setPlaying(false));
    } catch (e) {
      setError((e as Error).message);
    }
  };

  const profile = status?.profile ?? null;
  const left = status ? Math.max(0, Math.ceil(status.total_seconds - status.seconds)) : 0;

  return (
    <section className="calibration" aria-label="My voice">
      <h3>My voice</h3>
      {error && <p role="alert" className="error">{error}</p>}
      {recording && status ? (
        <>
          {status.phase === "quiet" ? (
            <p className="sentence" aria-live="polite">Stay quiet for a moment. The microphone is measuring the room.</p>
          ) : status.phase === "testing" ? (
            <>
              <p className="hint" aria-live="polite">Now say this in your normal voice. It sets your microphone level.</p>
              <p className="sentence">&ldquo;{status.test_phrase ?? "Testing, one, two, three"}&rdquo;</p>
            </>
          ) : (
            <>
              <p className="hint" aria-live="polite">Now read the whole paragraph at your normal pace, then press Done.</p>
              <p className="sentence">{status.text}</p>
            </>
          )}
          <LevelBar level={status.level} />
          <div className="actions">
            {status.phase === "reading" && (
              <button disabled={busy || !canFinish} onClick={finishNow}>Done</button>
            )}
            <button className="secondary" disabled={busy} onClick={() => run(calibrationApi.cancel)}>Cancel</button>
          </div>
          {status.phase === "reading" && <p className="hint">It stops by itself after {left} more seconds.</p>}
        </>
      ) : (
        <>
          {profile ? (
            <p className="hint">
              Calibrated on {profile.device}: your usual pitch is {Math.round(profile.pitch_hz)} Hz, the room is at{" "}
              {Math.round(profile.noise_db)} dB, and your speech at {Math.round(profile.level_db)} dB, boosted{" "}
              {profile.input_gain.toFixed(1)}x for the model. Calibrate again if you change microphone or room.
            </p>
          ) : (
            <p className="hint">
              Not calibrated yet. A recording of about 20 seconds sets the pitch for each voice automatically, keeps
              room noise from being turned into voice, and evens out your level.
            </p>
          )}
          <button className="secondary" disabled={busy || disabled || input === null}
            onClick={() => input !== null && run(() => calibrationApi.start(input))}>
            {profile ? "Calibrate again" : "Calibrate my voice"}
          </button>
          {profile && (
            <button className="secondary" disabled={busy || playing} onClick={playCheck}>
              {playing ? "Playing..." : "Play my mic check"}
            </button>
          )}
          {profile && (
            <button className="secondary" disabled={busy} onClick={() => run(calibrationApi.forget)}>Forget calibration</button>
          )}
        </>
      )}
    </section>
  );
}
