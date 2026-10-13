import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api, Device, DeviceList } from "./api";
import { isBusy, MAX_TEXT, TtsEnvironment, TtsPreset, ttsApi, TtsVoice } from "./ttsApi";
import { TtsView } from "./useTtsStatus";

const PRESETS: { id: TtsPreset; label: string }[] = [
  { id: "fast", label: "Fast" },
  { id: "balanced", label: "Balanced" },
  { id: "best", label: "Best" },
];

/** A virtual mic or cable is never a place to listen: that would send the speech to other apps by another route. */
const isVirtual = (d: Device) => /voicemeeter|cable/i.test(d.name);

/** Characters the way the backend counts them (an emoji is one, not two UTF-16 units). */
const countChars = (s: string) => [...s].length;

export function SpeakPanel({ tts, active, liveRunning }: {
  tts: TtsView;
  /** Whether the Speak tab is visible: nothing is asked of the backend until it first is. */
  active: boolean;
  /** Live mode is running: a speak request would be refused, so Speak says so instead. */
  liveRunning: boolean;
}) {
  const { status, unreachable, refresh, apply } = tts;
  const [list, setList] = useState<DeviceList | null>(null);
  const [voices, setVoices] = useState<TtsVoice[] | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [env, setEnv] = useState<TtsEnvironment | null>(null);
  const [envError, setEnvError] = useState<string | null>(null);
  const [checking, setChecking] = useState(false);
  const [voiceId, setVoiceId] = useState("");
  const [preset, setPreset] = useState<TtsPreset>("balanced");
  const [text, setText] = useState("");
  const [monitor, setMonitor] = useState<number | null>(null);
  const [toMic, setToMic] = useState(true);
  const [save, setSave] = useState(false);
  const [actionError, setActionError] = useState<string | null>(null);
  const [pending, setPending] = useState(false);
  const [stopping, setStopping] = useState(false);
  const envAsked = useRef(false);

  const outputs = useMemo(
    () => (list?.devices ?? []).filter((d) => d.max_output > 0 && !isVirtual(d) && d.index !== list?.virtual_mic),
    [list],
  );

  const load = useCallback(async () => {
    try {
      const [devs, vs] = await Promise.all([list ? Promise.resolve(list) : api.devices(), ttsApi.voices()]);
      if (!list) {
        setList(devs);
        const real = devs.devices.filter((d) => d.max_output > 0 && !isVirtual(d) && d.index !== devs.virtual_mic);
        setMonitor((real.find((d) => d.index === devs.default_output) ?? real[0])?.index ?? null);
      }
      setVoices(vs);
      setVoiceId((current) => {
        if (vs.some((v) => v.id === current && v.reference.ok)) return current;
        return vs.find((v) => v.reference.ok)?.id ?? "";
      });
      setLoadError(null);
    } catch (e) {
      setLoadError((e as Error).message);
    }
  }, [list]);

  const checkEnvironment = useCallback(async (refreshIt: boolean) => {
    setChecking(true);
    try {
      setEnv(await ttsApi.environment(refreshIt));
      setEnvError(null);
    } catch (e) {
      setEnvError((e as Error).message);
    } finally {
      setChecking(false);
    }
  }, []);

  // Voices (a new recording makes a voice usable) reload each time the tab is shown; devices and the check only once.
  useEffect(() => {
    if (!active) return;
    void load();
    if (!envAsked.current) {
      envAsked.current = true;
      void checkEnvironment(false);
    }
  }, [active]); // eslint-disable-line react-hooks/exhaustive-deps

  const voice = voices?.find((v) => v.id === voiceId) ?? null;
  const bestAllowed = voice?.has_model === true;
  const shownPreset: TtsPreset = preset === "best" && !bestAllowed ? "balanced" : preset;

  const busy = isBusy(status);
  const chars = countChars(text);
  const tooLong = chars > MAX_TEXT;
  const hasVirtualMic = list?.virtual_mic != null;
  const micOn = hasVirtualMic && toMic;
  const hasDestination = micOn || monitor !== null;
  const unchecked = env?.unknown === true; // the backend may not probe the GPU right now: not a broken setup
  const envBroken = env !== null && !env.ready && !unchecked;
  const failing = env?.checks.filter((c) => !c.ok) ?? [];
  const canSpeak =
    status !== null && !busy && !pending && !liveRunning && voice !== null && text.trim() !== "" && !tooLong &&
    hasDestination;

  const speak = async () => {
    if (!canSpeak || voice === null) return;
    setPending(true);
    setActionError(null);
    try {
      apply(
        await ttsApi.speak({
          voice_id: voice.id,
          text,
          preset: shownPreset,
          mic_device: micOn ? (list?.virtual_mic ?? null) : null,
          monitor_device: monitor,
          save,
        }),
      );
    } catch (e) {
      setActionError((e as Error).message);
      refresh(); // the refusal may have come after something started
    } finally {
      setPending(false);
    }
  };

  const stop = async () => {
    if (stopping || status?.state === "stopping") return;
    setStopping(true);
    setActionError(null);
    try {
      apply(await ttsApi.stop());
    } catch (e) {
      setActionError((e as Error).message);
    } finally {
      setStopping(false);
    }
  };

  const message = actionError ?? (!pending && status?.state === "idle" ? status.error : null);
  const speaking = status?.state === "speaking";
  const progress = (() => {
    if (!status) return "";
    if (status.engine === "starting")
      return "Getting ready. Starting the voice engine. The first time takes about 20 seconds.";
    if (status.state === "speaking")
      return status.piece > 0 ? `Speaking sentence ${status.piece} of ${status.pieces}` : "Speaking";
    if (status.state === "preparing") return "Preparing the speech…";
    if (status.state === "stopping") return "Stopping…";
    return status.saved_path && !pending ? `Saved to ${status.saved_path}` : "";
  })();

  return (
    <section
      className="speak"
      onKeyDown={(e) => {
        if (e.key === "Escape" && speaking) {
          e.preventDefault();
          void stop();
        }
      }}
    >
      {message && <p role="alert" className="error">{message}</p>}
      {unreachable && (
        <p role="alert" className="error">
          Could not reach the voice engine. Retrying… <button onClick={refresh}>Retry</button>
        </p>
      )}
      {loadError && (
        <p role="alert" className="error">
          {loadError} <button onClick={() => void load()}>Retry</button>
        </p>
      )}
      {envError && <p role="alert" className="error">{envError}</p>}
      {voices !== null && voices.length === 0 && (
        <p className="hint">A voice needs one recorded sentence to be spoken. Record one on the Voices tab.</p>
      )}
      {envBroken && (
        <div className="setup">
          <p>Text to speech is not set up on this computer. See Text to speech setup in the README.</p>
          <ul>{failing.map((c) => <li key={c.id}>{c.message}</li>)}</ul>
        </div>
      )}
      {unchecked && (
        <p className="hint">
          {env?.checked_while_live
            ? "Text to speech can be checked once live mode is stopped."
            : "Text to speech cannot be checked right now."}
        </p>
      )}
      {(envError !== null || envBroken) && (
        <button className="secondary" disabled={checking} onClick={() => void checkEnvironment(true)}>Check again</button>
      )}

      {voices !== null && voices.length > 0 && !envBroken && (
        <>
          <label>
            Voice
            <select value={voiceId} onChange={(e) => setVoiceId(e.target.value)} disabled={busy}>
              {voices.map((v) => (
                <option key={v.id} value={v.id} disabled={!v.reference.ok}>
                  {v.reference.ok ? v.name : `${v.name} (${v.reference.message})`}
                </option>
              ))}
            </select>
          </label>
          <label>
            Text to speak
            <textarea
              rows={6}
              value={text}
              aria-describedby="speak-count"
              onChange={(e) => setText(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && e.ctrlKey) {
                  e.preventDefault();
                  void speak();
                }
              }}
            />
          </label>
          <p id="speak-count" className={tooLong ? "count over" : "count"}>{chars} of {MAX_TEXT} characters</p>
          <label>
            Speed / quality
            <select value={shownPreset} onChange={(e) => setPreset(e.target.value as TtsPreset)} disabled={busy}>
              {PRESETS.map((p) => (
                <option key={p.id} value={p.id} disabled={p.id === "best" && !bestAllowed}>{p.label}</option>
              ))}
            </select>
          </label>
          {!bestAllowed && voice !== null && (
            <p className="hint">Best needs a trained model. Train this voice on the Voices tab.</p>
          )}
          <label>
            Listen on
            <select
              value={monitor ?? ""}
              onChange={(e) => setMonitor(e.target.value === "" ? null : Number(e.target.value))}
              disabled={busy}
            >
              <option value="">Off</option>
              {outputs.map((d) => <option key={d.index} value={d.index}>{d.name}</option>)}
            </select>
          </label>
          {hasVirtualMic && (
            <label className="row">
              <input type="checkbox" checked={toMic} disabled={busy} onChange={(e) => setToMic(e.target.checked)} />
              Send to virtual mic
            </label>
          )}
          {!hasDestination && <p className="hint">Choose a listen device or the virtual mic to speak.</p>}
          <label className="row">
            <input type="checkbox" checked={save} disabled={busy} onChange={(e) => setSave(e.target.checked)} />
            Keep as a WAV file
          </label>

          {liveRunning && <p className="hint">Stop live mode on the Live tab to speak.</p>}
          <div className="actions">
            {busy ? (
              <button onClick={() => void stop()} disabled={stopping || status?.state === "stopping"}>Stop</button>
            ) : (
              <button onClick={() => void speak()} disabled={!canSpeak}>Speak</button>
            )}
          </div>
        </>
      )}
      <div role="status" className="progress-line">{progress}</div>
    </section>
  );
}
