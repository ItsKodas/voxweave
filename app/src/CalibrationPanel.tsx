import { useEffect, useRef, useState } from "react";
import { calibrationApi, CalibrationStatus, VoiceProfile } from "./calibrationApi";

const POLL_MS = 250;

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
              <p className="hint" aria-live="polite">Now read this aloud in your normal voice. {left} seconds left.</p>
              <p className="sentence">{status.text}</p>
            </>
          )}
          <label className="meter">
            Level
            <progress max={1} value={Math.min(1, status.level)} />
          </label>
          <button className="secondary" disabled={busy} onClick={() => run(calibrationApi.cancel)}>Cancel</button>
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
            <button className="secondary" disabled={busy} onClick={() => run(calibrationApi.forget)}>Forget calibration</button>
          )}
        </>
      )}
    </section>
  );
}
