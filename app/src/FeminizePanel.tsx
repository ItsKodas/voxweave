import { useEffect, useRef, useState } from "react";
import { DEFAULT_FEMINIZE, DEFAULT_MASCULINIZE, FeminizeJob, FeminizeSettings, feminizeApi } from "./feminizeApi";
import { LibraryVoice } from "./studioApi";

const POLL_MS = 500;
const signedPercent = (x: number) => {
  const p = Math.round((x - 1) * 100);
  return `${p > 0 ? "+" : ""}${p}%`;
};

type Mode = "feminine" | "masculine";
const modeOf = (s: FeminizeSettings): Mode =>
  s.pitch_semitones < 0 || (s.pitch_semitones === 0 && s.formant_shift < 1) ? "masculine" : "feminine";

/** What each direction offers: the slider ranges, the defaults, and the words used for the result. */
const MODES = {
  feminine: {
    defaults: DEFAULT_FEMINIZE, pitch: [0, 12], resonance: [1, 1.35], adjective: "feminized",
    explain: "Raises the pitch and the resonances of this voice, the way a shorter vocal tract would, and can add a little breath.",
  },
  masculine: {
    defaults: DEFAULT_MASCULINIZE, pitch: [-12, 0], resonance: [0.8, 1], adjective: "masculinized",
    explain: "Lowers the pitch and the resonances of this voice, the way a longer vocal tract would.",
  },
} as const;

/**
 * Change a voice's gender: a version with a different pitch and different resonances, made by signal processing.
 * Listen to a sample, use the settings for live mode's preview, or make a new voice from the changed clips and train it.
 */
export function FeminizePanel({ voice, onCreated }: { voice: LibraryVoice; onCreated: (voiceId: string) => void }) {
  const [settings, setSettings] = useState<FeminizeSettings>(DEFAULT_FEMINIZE);
  const [busy, setBusy] = useState<"sample" | "save" | "start" | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [saved, setSaved] = useState(false);
  const [job, setJob] = useState<FeminizeJob | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const urlRef = useRef<string | null>(null);
  const onCreatedRef = useRef(onCreated);
  onCreatedRef.current = onCreated;
  const mode = modeOf(settings);
  const m = MODES[mode];

  // Start from the settings saved for the live preview, so the sliders mean the same thing everywhere.
  useEffect(() => {
    let gone = false;
    feminizeApi.settings().then((s) => !gone && setSettings(s), () => undefined);
    feminizeApi.status().then((s) => !gone && s.active && setJob(s), () => undefined);
    return () => {
      gone = true;
      audioRef.current?.pause();
      if (urlRef.current) URL.revokeObjectURL(urlRef.current);
    };
  }, []);

  const following = job?.active === true;
  useEffect(() => {
    if (!following) return;
    let gone = false;
    const timer = setInterval(async () => {
      try {
        const now = await feminizeApi.status();
        if (gone) return;
        setJob(now);
        if (!now.active && now.finished && now.voice_id && !now.error && !now.cancelled) onCreatedRef.current(now.voice_id);
      } catch {
        // asked again on the next tick
      }
    }, POLL_MS);
    return () => {
      gone = true;
      clearInterval(timer);
    };
  }, [following]);

  const change = (key: keyof FeminizeSettings, value: number) => {
    setSettings((s) => ({ ...s, [key]: value }));
    setSaved(false);
  };

  const switchMode = (next: Mode) => {
    if (next === mode) return;
    setSettings({ ...MODES[next].defaults });
    setSaved(false);
  };

  const run = async (kind: "sample" | "save" | "start", action: () => Promise<void>) => {
    setBusy(kind);
    setError(null);
    try {
      await action();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(null);
    }
  };

  const playSample = () =>
    run("sample", async () => {
      const url = await feminizeApi.sampleUrl(voice.id, settings);
      audioRef.current?.pause();
      if (urlRef.current) URL.revokeObjectURL(urlRef.current);
      urlRef.current = url;
      const audio = new Audio(url);
      audioRef.current = audio;
      await audio.play().catch(() => undefined);
    });

  const otherVoiceJob = job?.active && job.source !== voice.id;

  return (
    <div className="feminize">
      {error && <p role="alert" className="error">{error}</p>}
      <div role="radiogroup" aria-label="Direction" className="kind">
        {(["feminine", "masculine"] as const).map((k) => (
          <label key={k}>
            <input type="radio" name={`gender-${voice.id}`} checked={mode === k} disabled={busy !== null}
              onChange={() => switchMode(k)} />
            {k === "feminine" ? "More feminine" : "More masculine"}
          </label>
        ))}
      </div>
      <p className="hint">
        {m.explain} Listen to a sample first. Then either use it live straight away (rougher, no training), or make a
        new voice from the changed clips and train it (smoother, about three hours).
      </p>
      <label>
        Pitch: {settings.pitch_semitones > 0 ? "+" : ""}{settings.pitch_semitones} semitones
        <input type="range" min={m.pitch[0]} max={m.pitch[1]} step={0.5} value={settings.pitch_semitones}
          onChange={(e) => change("pitch_semitones", Number(e.target.value))} />
      </label>
      <label>
        Resonance: {signedPercent(settings.formant_shift)}
        <input type="range" min={m.resonance[0]} max={m.resonance[1]} step={0.01} value={settings.formant_shift}
          onChange={(e) => change("formant_shift", Number(e.target.value))} />
      </label>
      <label>
        Breathiness: {Math.round(settings.breathiness * 100)}%
        <input type="range" min={0} max={0.6} step={0.05} value={settings.breathiness}
          onChange={(e) => change("breathiness", Number(e.target.value))} />
      </label>
      <div className="buttons">
        <button className="secondary" disabled={busy !== null || voice.clips === 0} onClick={playSample}>
          {busy === "sample" ? "Making the sample..." : "Hear a sample"}
        </button>
        <button className="secondary" disabled={busy !== null}
          onClick={() => run("save", async () => {
            setSettings(await feminizeApi.saveSettings(settings));
            setSaved(true);
          })}>
          Use for the live preview
        </button>
        <button disabled={busy !== null || following || voice.clips === 0}
          onClick={() => run("start", async () => setJob(await feminizeApi.start(voice.id, settings)))}>
          {mode === "feminine" ? "Make a feminized voice" : "Make a masculinized voice"}
        </button>
      </div>
      {saved && (
        <p className="hint" aria-live="polite">
          Saved. On the Live tab, choose the voice "My voice, {m.adjective} (preview, no training)".
        </p>
      )}
      {job && !otherVoiceJob && (
        job.active ? (
          <div aria-live="polite">
            <p className="hint">Making "{job.voice_name}": clip {job.done} of {job.total}.</p>
            <progress max={job.total || 1} value={job.done} />
            <button className="secondary" onClick={() => run("start", async () => setJob(await feminizeApi.cancel()))}>Stop</button>
          </div>
        ) : job.error ? (
          <p role="alert" className="error">{job.error}</p>
        ) : job.cancelled ? (
          <p className="hint">Stopped after {job.done} of {job.total} clips. "{job.voice_name}" keeps the clips made so far.</p>
        ) : job.finished ? (
          <p className="hint" aria-live="polite">
            "{job.voice_name}" is ready with {job.done - (job.skipped ?? 0)} clips. Open it in the list above and press
            Train. Then pick it on the Live tab.
          </p>
        ) : null
      )}
      {otherVoiceJob && <p className="hint">A changed voice is being made from another voice. Wait for it to finish.</p>}
    </div>
  );
}
