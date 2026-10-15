import { useEffect, useMemo, useRef, useState } from "react";
import { LiveInsight } from "./LiveInsight";
import { api, Device, DeviceList, Params, Stats, Voice } from "./api";
import { CalibrationPanel } from "./CalibrationPanel";
import { calibrationApi, VoiceProfile } from "./calibrationApi";
import { installedApp } from "./setupApi";
import { heardAs, micOptionLabel } from "./virtualMic";
import { PREVIEW_VOICE_ID } from "./feminizeApi";


type Selection = { input: number | null; mic: number | null; monitor: number | null };

const isVirtual = (d: Device) => /voicemeeter|cable/i.test(d.name);

/**
 * Chooses devices for a fresh device list. `keep` holds the current choices, which survive when
 * the device still exists. The virtual mic is only ever the detected one or None: a real output
 * (speakers) is never picked for it, and the listen device is never the virtual mic.
 */
function chooseDevices(list: DeviceList, keep?: Selection & { hadVirtualMic: boolean }): Selection {
  const ins = list.devices.filter((d) => d.max_input > 0);
  const outs = list.devices.filter((d) => d.max_output > 0);
  const has = (group: Device[], index: number | null | undefined): index is number =>
    index != null && group.some((d) => d.index === index);

  const input = has(ins, keep?.input) ? keep.input
    : has(ins, list.default_input) ? list.default_input
    : ins[0]?.index ?? null;

  let mic = list.virtual_mic;
  if (keep) {
    if (has(outs, keep.mic)) mic = keep.mic;
    else if (keep.mic === null && keep.hadVirtualMic) mic = null; // None was the user's own choice
  }

  const listenable = outs.filter((d) => d.index !== mic);
  const monitor = has(listenable, keep?.monitor) ? keep.monitor
    : has(listenable, list.default_output) ? list.default_output
    : (listenable.find((d) => !isVirtual(d)) ?? listenable[0])?.index ?? null;

  return { input, mic, monitor };
}

/** VB-Audio's own page for VB-CABLE, the virtual cable that gives the app a microphone other apps can choose. */
const CABLE_PAGE = "https://vb-audio.com/Cable/";

/**
 * The steps of the quality slider, fastest first: the same as QUALITY_LEVELS in backend/aivoice/presets.py.
 * Longer chunks with more context give the model and the pitch tracker more to go on, at the cost of delay.
 */
export const QUALITY_LEVELS = [
  { chunk_ms: 120, context_ms: 400, crossfade_ms: 30 },
  { chunk_ms: 250, context_ms: 800, crossfade_ms: 50 },
  { chunk_ms: 500, context_ms: 1500, crossfade_ms: 80 },
  { chunk_ms: 750, context_ms: 2500, crossfade_ms: 100 },
  { chunk_ms: 1000, context_ms: 3000, crossfade_ms: 120 },
  { chunk_ms: 1500, context_ms: 4000, crossfade_ms: 150 },
  { chunk_ms: 2000, context_ms: 5000, crossfade_ms: 200 },
];
const QUALITY_KEY = "quality-level";
/** The level a new install starts at, and what Reset to recommended goes back to. */
const DEFAULT_QUALITY = 1;
const isQuality = (n: unknown): n is number => Number.isInteger(n) && (n as number) >= 0 && (n as number) < QUALITY_LEVELS.length;
function storedQuality(): number {
  try {
    const n = Number(localStorage.getItem(QUALITY_KEY));
    return isQuality(n) && localStorage.getItem(QUALITY_KEY) !== null ? n : DEFAULT_QUALITY;
  } catch {
    return DEFAULT_QUALITY;
  }
}
/** Roughly how far behind your voice the converted voice is at a level: a chunk is collected, then converted. */
const delaySeconds = (level: number) => {
  const q = QUALITY_LEVELS[level];
  return (2 * q.chunk_ms + q.crossfade_ms) / 1000;
};

/** The likeness chosen for each voice is remembered on this computer; it never needs to reach the backend's disk. */
const LIKENESS_KEY = (voiceId: string) => `likeness:${voiceId}`;
function storedLikeness(voiceId: string): number | null {
  try {
    const raw = localStorage.getItem(LIKENESS_KEY(voiceId));
    const value = raw === null ? NaN : Number(raw);
    return Number.isFinite(value) && value >= 0 && value <= 1 ? value : null;
  } catch {
    return null;
  }
}
function storeLikeness(voiceId: string, value: number | null) {
  try {
    if (value === null) localStorage.removeItem(LIKENESS_KEY(voiceId));
    else localStorage.setItem(LIKENESS_KEY(voiceId), String(value));
  } catch {
    // remembered for this session only
  }
}

/** Settings saved for a voice with "Save for this voice": put back whenever that voice is chosen. */
type VoiceSettings = { pitch: number; quality: number; likeness: number };
const SETTINGS_KEY = (voiceId: string) => `voice-settings:${voiceId}`;
function savedSettings(voiceId: string): VoiceSettings | null {
  try {
    const raw = localStorage.getItem(SETTINGS_KEY(voiceId));
    const s = raw === null ? null : JSON.parse(raw);
    if (s === null || typeof s !== "object") return null;
    const ok = Number.isInteger(s.pitch) && s.pitch >= -24 && s.pitch <= 24 && isQuality(s.quality)
      && typeof s.likeness === "number" && s.likeness >= 0 && s.likeness <= 1;
    return ok ? { pitch: s.pitch, quality: s.quality, likeness: s.likeness } : null;
  } catch {
    return null;
  }
}
function storeSettings(voiceId: string, settings: VoiceSettings | null) {
  try {
    if (settings === null) localStorage.removeItem(SETTINGS_KEY(voiceId));
    else localStorage.setItem(SETTINGS_KEY(voiceId), JSON.stringify(settings));
  } catch {
    // remembered for this session only
  }
}

function Meter({ label, level }: { label: string; level: number }) {
  return (
    <label className="meter">
      {label}
      <progress max={1} value={Math.min(1, level)} />
    </label>
  );
}

type Props = {
  stats: Stats;
  actionError?: string | null;
  /** Called after "Refresh devices" succeeded, so setup checks that depend on devices can re-run. */
  onDevicesRefreshed?: () => void;
  /** Changes when the voice library changed (a training run finished): the voice list reloads, nothing else. */
  voicesVersion?: number;
  /** Called with the listen device chosen here, so the Speak tab can start on the same one. */
  onMonitorChosen?: (index: number | null) => void;
  /** Called with the virtual mic chosen here, so the Speak tab can start on the same one. */
  onMicChosen?: (index: number | null) => void;
  /** Speech is being made or played: starting live mode and refreshing devices would be refused. */
  speechActive?: boolean;
};

export function LivePanel({ stats, actionError = null, onDevicesRefreshed, onMonitorChosen, onMicChosen, voicesVersion = 0, speechActive = false }: Props) {
  const [devices, setDevices] = useState<Device[]>([]);
  const [virtualMic, setVirtualMic] = useState<number | null>(null);
  const [voices, setVoices] = useState<Voice[]>([]);
  const [presets, setPresets] = useState<Record<string, Params>>({});
  const [input, setInput] = useState<number | null>(null);
  const [mic, setMic] = useState<number | null>(null);
  const [monitorOn, setMonitorOn] = useState(false);
  const [monitor, setMonitor] = useState<number | null>(null);
  const [monitorGain, setMonitorGain] = useState(1);
  const [voice, setVoice] = useState("");
  const voiceRef = useRef(voice); // for work that finishes after the voice may have been changed
  voiceRef.current = voice;
  const [preset] = useState("balanced");
  const [quality, setQuality] = useState(storedQuality);
  const [overrides, setOverrides] = useState<Partial<Params>>({});
  const [pitch, setPitch] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [loaded, setLoaded] = useState(false);
  const [copied, setCopied] = useState(false);
  const [profile, setProfile] = useState<VoiceProfile | null>(null);
  const [useCalibration, setUseCalibration] = useState(true);
  const [pitchNote, setPitchNote] = useState<string | null>(null);
  const [saved, setSaved] = useState<VoiceSettings | null>(null);
  const [resets, setResets] = useState(0); // counts Reset to recommended, which applies the voice's defaults again
  // Deep calibration: tries settings on the calibration reading and keeps the most alike (aivoice.deeptune).
  const [tuning, setTuning] = useState<string | null>(null); // the progress line while it runs
  const [tuneNote, setTuneNote] = useState<string | null>(null);

  // The quality level sets the chunk timing; the Advanced fields can still override each value by hand.
  useEffect(() => {
    setOverrides((o) => ({ ...o, ...QUALITY_LEVELS[quality] }));
    try {
      localStorage.setItem(QUALITY_KEY, String(quality));
    } catch {
      // remembered for this session only
    }
  }, [quality]);

  // Likeness is the index rate, kept per voice: how hard speech is pulled towards sounds of the training voice.
  // Settings saved for the voice come first: its pitch, quality and likeness, ahead of the calibration's pitch.
  useEffect(() => {
    if (voice === "") return;
    const settings = savedSettings(voice);
    setSaved(settings);
    if (settings !== null) {
      setQuality(settings.quality);
      setPitch(settings.pitch);
      setPitchNote(null);
      setOverrides((current) => ({ ...current, index_rate: settings.likeness }));
      return;
    }
    const remembered = storedLikeness(voice);
    setOverrides((current) => {
      const { index_rate: _dropped, ...rest } = current;
      return remembered === null ? rest : { ...rest, index_rate: remembered };
    });
  }, [voice, resets]);

  // With a calibration, choosing a voice sets the pitch that moves the speaker onto that voice. Only then: a
  // pitch moved by hand stays until the voice or the calibration changes.
  const calibrated = profile?.created ?? null;
  useEffect(() => {
    if (voice === PREVIEW_VOICE_ID) {
      // The preview takes its pitch from the feminize settings; the slider only adds to that.
      setPitch(0);
      setPitchNote("The Change gender settings on a voice's page set this preview's pitch. The slider adds to it.");
      return;
    }
    if (voice !== "" && savedSettings(voice) !== null) return;
    if (calibrated === null || voice === "") {
      setPitchNote(null);
      return;
    }
    let stale = false;
    calibrationApi.suggestion(voice).then(
      (s) => {
        if (stale) return;
        if (s.pitch_shift === null) {
          setPitchNote(s.reason);
          return;
        }
        setPitch(s.pitch_shift);
        setPitchNote(`Set from your calibration: you speak at about ${Math.round(s.speaker_hz ?? 0)} Hz, this voice at about ${Math.round(s.target_hz ?? 0)} Hz.`);
        if (stats.running) api.settings({ pitch_shift: s.pitch_shift }).catch(() => undefined);
      },
      () => undefined,
    );
    return () => {
      stale = true;
    };
  }, [calibrated, voice, resets]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (loaded) onMonitorChosen?.(monitor);
  }, [loaded, monitor]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => {
    if (loaded) onMicChosen?.(mic);
  }, [loaded, mic]); // eslint-disable-line react-hooks/exhaustive-deps

  const inputs = useMemo(() => devices.filter((d) => d.max_input > 0), [devices]);
  const outputs = useMemo(() => devices.filter((d) => d.max_output > 0), [devices]);

  const [loadFailed, setLoadFailed] = useState(false);

  const applyDevices = (list: DeviceList, chosen: Selection) => {
    setDevices(list.devices);
    setVirtualMic(list.virtual_mic);
    setInput(chosen.input);
    setMic(chosen.mic);
    setMonitor(chosen.monitor);
  };

  /** The backend still offers the feminized preview voice; the screen no longer shows it (the feature is shelved). */
  const shown = (list: Voice[]) => list.filter((v) => v.id !== PREVIEW_VOICE_ID);

  const load = () => {
    setError(null);
    setLoadFailed(false);
    Promise.all([api.devices(), api.voices(), api.presets()])
      .then(([d, v, p]) => {
        applyDevices(d, chooseDevices(d));
        setVoices(shown(v));
        setPresets(p);
        setVoice(shown(v)[0]?.id ?? "");
        setLoaded(true);
      })
      .catch((e: Error) => {
        setError(e.message);
        setLoadFailed(true);
      });
  };

  useEffect(load, []); // eslint-disable-line react-hooks/exhaustive-deps

  // A new trained voice appears without a restart. Only the list reloads, so the form and a running session stay.
  const seenVoicesVersion = useRef(voicesVersion);
  useEffect(() => {
    if (seenVoicesVersion.current === voicesVersion) return;
    seenVoicesVersion.current = voicesVersion;
    let stale = false;
    api.voices().then(
      (list) => {
        if (stale) return;
        const visible = shown(list);
        setVoices(visible);
        setVoice((current) => (visible.some((v) => v.id === current) ? current : (visible[0]?.id ?? "")));
      },
      () => {}, // the list on screen is still valid; the next change reloads it again
    );
    return () => {
      stale = true;
    };
  }, [voicesVersion]);

  const reportFailure = (e: unknown) => setError((e as Error).message);

  const run = async (action: () => Promise<unknown>) => {
    setBusy(true);
    setError(null);
    try {
      await action();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const refreshDevices = () =>
    run(async () => {
      const list = await api.refreshDevices();
      applyDevices(list, chooseDevices(list, { input, mic, monitor, hadVirtualMic: virtualMic !== null }));
      onDevicesRefreshed?.();
    });

  /** After installing a virtual cable: rescan, and move the virtual mic to the cable that has just appeared. */
  const findCable = () =>
    run(async () => {
      const list = await api.refreshDevices();
      const cable = list.devices.find((d) => d.max_output > 0 && /cable input/i.test(d.name));
      const chosen = chooseDevices(list, { input, mic, monitor, hadVirtualMic: virtualMic !== null });
      applyDevices(list, cable ? { ...chosen, mic: cable.index, monitor: chosen.monitor === cable.index ? null : chosen.monitor } : chosen);
      onDevicesRefreshed?.();
      if (!cable) setError("No virtual cable was found yet. Finish the installer, restart Windows if it asked, and try again.");
    });

  const start = () =>
    run(() =>
      api.start({
        input_device: input as number,
        mic_device: mic,
        voice,
        monitor_device: monitorOn ? monitor : null,
        monitor_gain: monitorGain,
        preset,
        overrides: changedOverrides(),
        pitch_shift: pitch,
        ...(profile !== null ? { use_calibration: useCalibration } : {}),
      }),
    );

  const base = presets[preset];
  const value = <K extends keyof Params>(key: K): Params[K] | "" =>
    (overrides[key] ?? base?.[key] ?? "") as Params[K] | "";
  const likeness = Number(value("index_rate") || 0);
  const voiceName = voices.find((v) => v.id === voice)?.name ?? "this voice";
  const savedChanged = saved !== null
    && (saved.pitch !== pitch || saved.quality !== quality || Math.abs(saved.likeness - likeness) > 1e-6);
  const saveForVoice = () => {
    const settings = { pitch, quality, likeness };
    storeSettings(voice, settings);
    setSaved(settings);
  };
  const findBestSettings = async () => {
    const name = voiceName;
    const tunedVoice = voice;
    setTuneNote(null);
    setError(null);
    try {
      const cal = await calibrationApi.status();
      if (!cal.has_sample) {
        setTuneNote("Calibrate your voice again first: Find best settings converts the reading from your calibration, and calibrations made before this version kept only their measurements.");
        return;
      }
      const model = await calibrationApi.speakerModel();
      if (!model.ready) {
        const ok = window.confirm(
          `Find best settings needs a speaker-recognition model (${model.size_mb} MB) from Hugging Face: the ` +
          `WeSpeaker ResNet34 model, licensed CC BY 4.0. It is downloaded once and checked before it is used.\n\n` +
          "Download it now?",
        );
        if (!ok) return;
        setTuning("Downloading the speaker model...");
        await calibrationApi.downloadSpeakerModel();
      }
      setTuning("Getting ready...");
      let status = await calibrationApi.tune({ voice_id: tunedVoice, quality, pitch_shift: pitch, likeness });
      while (status.active) {
        if (status.steps > 0 && status.step > 0) setTuning(`${status.message} (${status.step} of ${status.steps})`);
        else if (status.message) setTuning(`${status.message}...`);
        await new Promise((r) => setTimeout(r, 600));
        status = await calibrationApi.tuneStatus();
      }
      if (status.error) throw new Error(status.error);
      const result = status.result;
      if (!result) {
        setTuneNote(status.message ?? "Stopped.");
        return;
      }
      const settings = { pitch: result.pitch_shift, quality, likeness: result.likeness };
      storeSettings(tunedVoice, settings);
      if (tunedVoice === voiceRef.current) {
        setPitch(result.pitch_shift);
        setPitchNote(null);
        setOverrides((o) => ({ ...o, index_rate: result.likeness }));
        setSaved(settings);
      }
      const pct = (x: number) => Math.round(x * 100);
      const same = result.pitch_shift === result.baseline.pitch_shift && result.likeness === result.baseline.likeness;
      setTuneNote(
        (same
          ? `Your settings were already the best found for ${name}: pitch ${result.pitch_shift > 0 ? "+" : ""}${result.pitch_shift}, likeness ${pct(result.likeness)}%. `
          : `Best for ${name}: pitch ${result.pitch_shift > 0 ? "+" : ""}${result.pitch_shift}, likeness ${pct(result.likeness)}%. `) +
          `It sounds ${pct(result.similarity)}% like ${name} (${pct(result.baseline.similarity)}% with your settings before), ` +
          `with ${pct(result.clarity)}% of the words clear. Saved for this voice; how robotic it sounds is still for your ears.`,
      );
    } catch (e) {
      setTuneNote(null);
      setError((e as Error).message);
    } finally {
      setTuning(null);
    }
  };

  const resetToRecommended = () => {
    storeSettings(voice, null);
    storeLikeness(voice, null);
    setSaved(null);
    setQuality(DEFAULT_QUALITY);
    if (calibrated === null) setPitch(0);
    setResets((n) => n + 1); // the likeness and the calibration's pitch are worked out again
  };
  /** Number fields show the typed override, with the preset value as placeholder. */
  const typed = (key: keyof Params) => overrides[key] ?? "";
  const placeholder = (key: keyof Params) => (base ? String(base[key]) : "");
  const override = <K extends keyof Params>(key: K, v: Params[K] | "") =>
    setOverrides((prev) => {
      const next = { ...prev };
      if (v === "") delete next[key];
      else next[key] = v as Params[K];
      return next;
    });
  const numberChange =
    (key: "chunk_ms" | "crossfade_ms" | "context_ms" | "index_rate") =>
    (e: React.ChangeEvent<HTMLInputElement>) =>
      override(key, e.target.value === "" ? "" : Number(e.target.value));
  /** Only values that differ from the chosen preset are sent as overrides. */
  const changedOverrides = (): Partial<Params> | null => {
    const diff = Object.fromEntries(
      Object.entries(overrides).filter(([k, v]) => !base || base[k as keyof Params] !== v),
    ) as Partial<Params>;
    return Object.keys(diff).length ? diff : null;
  };

  const hasCable = outputs.some((d) => /cable input/i.test(d.name));
  const copyCableLink = () => {
    // The webview does not open outside links itself, so the address is put on the clipboard instead.
    navigator.clipboard?.writeText(CABLE_PAGE).then(() => setCopied(true), () => undefined);
  };
  const micHeard = mic === null ? null : heardAs(outputs.find((d) => d.index === mic)?.name ?? "");
  const hasDestination = mic !== null || (monitorOn && monitor !== null);
  const canStart = !busy && !speechActive && input !== null && hasDestination && voice !== "";
  const shownError = error ?? actionError ?? stats.error;

  return (
    <section className="live">
      {shownError && <p role="alert" className="error">{shownError}</p>}
      {loadFailed && <button onClick={load}>Retry</button>}
      {speechActive && !stats.running && <p className="hint">Speech is playing. Stop it on the Speak tab to use live mode.</p>}
      {loaded && voices.length === 0 && <p className="hint">{installedApp()
        ? "No voice models installed. Bring in a trained voice as a voice pack: drop its .voxweave file on the Voices tab."
        : "No voice models installed. See the README to add one."}</p>}

      <label>
        Microphone
        <select value={input ?? ""} onChange={(e) => setInput(Number(e.target.value))} disabled={stats.running}>
          {inputs.map((d) => <option key={d.index} value={d.index}>{d.name}</option>)}
        </select>
      </label>
      <label>
        Virtual microphone (what other apps hear)
        <select
          value={mic ?? ""}
          onChange={(e) => setMic(e.target.value === "" ? null : Number(e.target.value))}
          disabled={stats.running}
        >
          <option value="">None (listen only)</option>
          {outputs.map((d) => <option key={d.index} value={d.index}>{micOptionLabel(d.name)}</option>)}
        </select>
      </label>
      {loaded && virtualMic === null && (
        <p className="hint">
          No virtual mic was found, so other apps will not hear anything until VoiceMeeter is installed and
          running. You can still listen to the converted voice yourself.
        </p>
      )}
      {loaded && micHeard && (
        <p className="hint">
          In other apps (Discord, a game, a recorder), choose <strong>{micHeard.microphone}</strong> as the microphone.{" "}
          {micHeard.note} Windows lists the end this app plays into under playback devices, which is why that end is
          called "Input". You can rename the microphone (for example to "Voxweave Mic") in Windows Sound settings, under
          Recording.
        </p>
      )}
      {loaded && mic !== null && !micHeard && (
        <p className="hint">This is an ordinary playback device, not a virtual microphone: other apps cannot choose it as a microphone.</p>
      )}
      {loaded && (hasCable ? null : (
        <details className="setup-mic">
          <summary>Set up a dedicated Voxweave microphone</summary>
          <p>
            Windows only lets a signed driver add a microphone, so this uses VB-CABLE, a free virtual cable from the
            makers of VoiceMeeter. It adds one pair of devices that pass sound straight through.
          </p>
          <ol>
            <li>
              Open <code>{CABLE_PAGE}</code> in your browser and download the VB-CABLE driver pack.{" "}
              <button className="secondary" onClick={copyCableLink}>{copied ? "Copied" : "Copy link"}</button>
            </li>
            <li>Unzip it, right-click <code>VBCABLE_Setup_x64.exe</code>, choose Run as administrator, and press Install Driver. Restart Windows if it asks.</li>
            <li>
              Come back here and press{" "}
              <button className="secondary" onClick={findCable} disabled={busy || stats.running || speechActive}>
                I have installed it
              </button>
              . "CABLE Input" is then chosen as the virtual mic above.
            </li>
            <li>In other apps, choose <strong>CABLE Output</strong> as the microphone. You can rename it to "Voxweave Mic" in Windows Sound settings.</li>
          </ol>
        </details>
      ))}
      {loaded && !hasDestination && !stats.running && (
        <p className="hint">Choose a virtual mic or tick Listen to output to start.</p>
      )}
      <label className="row">
        <input
          type="checkbox"
          checked={monitorOn}
          disabled={stats.running && !monitorOn}
          onChange={(e) => {
            const enabled = e.target.checked;
            if (!stats.running) setMonitorOn(enabled);
            else
              run(async () => {
                await api.settings({ monitor_enabled: enabled });
                setMonitorOn(enabled);
              });
          }}
        />
        Listen to output
      </label>
      {monitorOn && (
        <>
          <label>
            Listen on
            <select value={monitor ?? ""} onChange={(e) => setMonitor(Number(e.target.value))} disabled={stats.running}>
              {outputs.map((d) => <option key={d.index} value={d.index}>{d.name}</option>)}
            </select>
          </label>
          <label>
            Listen volume
            <input
              type="range" min={0} max={2} step={0.05} value={monitorGain}
              onChange={(e) => {
                const gain = Number(e.target.value);
                setMonitorGain(gain);
                if (stats.running) api.settings({ monitor_gain: gain }).catch(reportFailure);
              }}
            />
          </label>
        </>
      )}
      <button className="secondary" onClick={refreshDevices} disabled={busy || stats.running || speechActive}>
        Refresh devices
      </button>
      <label>
        Voice
        <select value={voice} onChange={(e) => setVoice(e.target.value)} disabled={stats.running}>
          {voices.map((v) => <option key={v.id} value={v.id}>{v.name}</option>)}
        </select>
      </label>
      <label>
        Quality: {quality + 1} of {QUALITY_LEVELS.length}, about {delaySeconds(quality).toFixed(1)} s behind your voice
        <input type="range" min={0} max={QUALITY_LEVELS.length - 1} step={1} value={quality} disabled={stats.running}
          onChange={(e) => setQuality(Number(e.target.value))} />
      </label>
      <p className="hint">Further right is smoother and steadier, and further behind. Set it before pressing Start.</p>
      <label>
        Pitch shift ({pitch > 0 ? "+" : ""}{pitch} semitones)
        <input
          type="range" min={-24} max={24} step={1} value={pitch}
          onChange={(e) => {
            const semitones = Number(e.target.value);
            setPitch(semitones);
            setPitchNote(null);
            if (stats.running) api.settings({ pitch_shift: semitones }).catch(reportFailure);
          }}
        />
      </label>
      {pitchNote && <p className="hint">{pitchNote}</p>}
      {voice !== PREVIEW_VOICE_ID && (
        <>
          <label>
            Likeness: {Math.round(Number(value("index_rate") || 0) * 100)}%
            <input type="range" min={0} max={1} step={0.05} value={Number(value("index_rate") || 0)} disabled={stats.running}
              onChange={(e) => {
                const rate = Number(e.target.value);
                setOverrides((o) => ({ ...o, index_rate: rate }));
                storeLikeness(voice, rate);
              }} />
          </label>
          <p className="hint">
            Higher sounds more like the voice but breaks up more when you speak differently from its recordings;
            lower is steadier. Remembered for each voice. Set it before pressing Start.
          </p>
        </>
      )}

      {voice !== "" && voice !== PREVIEW_VOICE_ID && (
        <>
          <div className="row">
            <button className="secondary" onClick={saveForVoice}>Save for this voice</button>
            <button className="secondary" disabled={stats.running || tuning !== null} onClick={resetToRecommended}>Reset to recommended</button>
            {profile !== null && (
              <button className="secondary" disabled={stats.running || tuning !== null} onClick={findBestSettings}>Find best settings</button>
            )}
          </div>
          {tuning !== null && (
            <p className="hint" aria-live="polite">
              {tuning}{" "}
              <button className="secondary" onClick={() => calibrationApi.tuneCancel().catch(reportFailure)}>Stop</button>
            </p>
          )}
          {tuneNote && <p className="hint" aria-live="polite">{tuneNote}</p>}
          <p className="hint">
            {saved === null
              ? "Save keeps this pitch, quality and likeness for this voice and puts them back whenever you choose it. Reset goes back to the pitch from your calibration, quality 2 and the usual likeness."
              : savedChanged
                ? `Changed since you saved. Press Save to keep these settings for ${voiceName}.`
                : `Using your saved settings for ${voiceName}.`}
          </p>
        </>
      )}

      <CalibrationPanel input={input} disabled={stats.running || speechActive} onProfile={setProfile} />
      {profile !== null && (
        <label className="row">
          <input type="checkbox" checked={useCalibration} disabled={stats.running}
            onChange={(e) => setUseCalibration(e.target.checked)} />
          Use my calibration (silence when I am quiet, even level)
        </label>
      )}

      <details>
        <summary>Advanced</summary>
        <label>
          Chunk (ms)
          <input type="number" value={typed("chunk_ms")} placeholder={placeholder("chunk_ms")} disabled={stats.running}
            onChange={numberChange("chunk_ms")} />
        </label>
        <label>
          Crossfade (ms)
          <input type="number" value={typed("crossfade_ms")} placeholder={placeholder("crossfade_ms")} disabled={stats.running}
            onChange={numberChange("crossfade_ms")} />
        </label>
        <label>
          Context (ms)
          <input type="number" value={typed("context_ms")} placeholder={placeholder("context_ms")} disabled={stats.running}
            onChange={numberChange("context_ms")} />
        </label>
        <label>
          Pitch detector
          <select value={value("f0_method")} disabled={stats.running}
            onChange={(e) => override("f0_method", e.target.value as Params["f0_method"])}>
            <option value="dio">Fast (dio)</option>
            <option value="harvest">Accurate (harvest)</option>
          </select>
        </label>
        <label>
          Index blend
          <input type="number" min={0} max={1} step={0.05} value={typed("index_rate")} placeholder={placeholder("index_rate")} disabled={stats.running}
            onChange={numberChange("index_rate")} />
        </label>
      </details>

      <div className="actions">
        {stats.running ? (
          <>
            <button onClick={() => run(() => api.stop())} disabled={busy}>Stop</button>
            {stats.mode === "muted" ? (
              <button onClick={() => run(() => api.mode("converted"))} disabled={busy}>Unmute</button>
            ) : (
              <button onClick={() => run(() => api.mode("muted"))} disabled={busy}>Mute</button>
            )}
          </>
        ) : (
          <button onClick={start} disabled={!canStart}>Start</button>
        )}
      </div>

      {stats.running && (
        <div className="readout">
          <p>
            Latency {Math.round(stats.latency_ms)} ms · Inference {Math.round(stats.infer_ms)} ms · Dropouts{" "}
            {stats.underruns}
          </p>
          <Meter label="Input" level={stats.in_level} />
          <Meter label="Output" level={stats.out_level} />
          {stats.suggest_faster && <p className="hint">The GPU is not keeping up. Move Quality further left.</p>}
          {stats.clipping && (
            <p className="hint" role="status">
              Your microphone is clipping: it is set too loud, and the distorted sound comes out harsh. Turn its level
              down in the Windows sound settings, then calibrate again.
            </p>
          )}
        </div>
      )}
      <LiveInsight running={stats.running} voiceName={voiceName} />
    </section>
  );
}
