import { useEffect, useMemo, useState } from "react";
import { api, Device, Params, Stats, Voice } from "./api";

const PRESET_LABELS: Record<string, string> = {
  low_latency: "Low latency",
  balanced: "Balanced",
  high_quality: "High quality",
};

function Meter({ label, level }: { label: string; level: number }) {
  return (
    <label className="meter">
      {label}
      <progress max={1} value={Math.min(1, level)} />
    </label>
  );
}

export function LivePanel({ stats }: { stats: Stats }) {
  const [devices, setDevices] = useState<Device[]>([]);
  const [voices, setVoices] = useState<Voice[]>([]);
  const [presets, setPresets] = useState<Record<string, Params>>({});
  const [input, setInput] = useState<number | null>(null);
  const [mic, setMic] = useState<number | null>(null);
  const [monitorOn, setMonitorOn] = useState(false);
  const [monitor, setMonitor] = useState<number | null>(null);
  const [monitorGain, setMonitorGain] = useState(1);
  const [voice, setVoice] = useState("");
  const [preset, setPreset] = useState("balanced");
  const [overrides, setOverrides] = useState<Partial<Params>>({});
  const [pitch, setPitch] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [loaded, setLoaded] = useState(false);

  const inputs = useMemo(() => devices.filter((d) => d.max_input > 0), [devices]);
  const outputs = useMemo(() => devices.filter((d) => d.max_output > 0), [devices]);

  useEffect(() => {
    Promise.all([api.devices(), api.voices(), api.presets()])
      .then(([d, v, p]) => {
        setDevices(d.devices);
        setVoices(v);
        setPresets(p);
        const ins = d.devices.filter((x) => x.max_input > 0);
        const outs = d.devices.filter((x) => x.max_output > 0);
        setInput(ins[0]?.index ?? null);
        setMic(d.virtual_mic ?? outs[0]?.index ?? null);
        setMonitor(outs.find((x) => x.index !== d.virtual_mic)?.index ?? null);
        setVoice(v[0]?.id ?? "");
        setLoaded(true);
      })
      .catch((e: Error) => setError(e.message));
  }, []);

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

  const start = () =>
    run(() =>
      api.start({
        input_device: input as number,
        mic_device: mic as number,
        voice,
        monitor_device: monitorOn ? monitor : null,
        monitor_gain: monitorGain,
        preset,
        overrides: Object.keys(overrides).length ? overrides : null,
        pitch_shift: pitch,
      }),
    );

  const base = presets[preset];
  const value = <K extends keyof Params>(key: K): Params[K] | "" =>
    (overrides[key] ?? base?.[key] ?? "") as Params[K] | "";
  const override = <K extends keyof Params>(key: K, v: Params[K]) =>
    setOverrides((prev) => {
      const next = { ...prev };
      if (base && base[key] === v) delete next[key];
      else next[key] = v;
      return next;
    });

  const canStart = !busy && input !== null && mic !== null && voice !== "";
  const shownError = error ?? stats.error;

  return (
    <section className="live">
      {shownError && <p role="alert" className="error">{shownError}</p>}
      {loaded && voices.length === 0 && <p className="hint">No voice models installed. See the README to add one.</p>}

      <label>
        Microphone
        <select value={input ?? ""} onChange={(e) => setInput(Number(e.target.value))} disabled={stats.running}>
          {inputs.map((d) => <option key={d.index} value={d.index}>{d.name}</option>)}
        </select>
      </label>
      <label>
        Virtual mic
        <select value={mic ?? ""} onChange={(e) => setMic(Number(e.target.value))} disabled={stats.running}>
          {outputs.map((d) => <option key={d.index} value={d.index}>{d.name}</option>)}
        </select>
      </label>
      <label className="row">
        <input
          type="checkbox"
          checked={monitorOn}
          disabled={stats.running && !monitorOn}
          onChange={(e) => {
            setMonitorOn(e.target.checked);
            if (stats.running) run(() => api.settings({ monitor_enabled: e.target.checked }));
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
                if (stats.running) api.settings({ monitor_gain: gain }).catch(() => undefined);
              }}
            />
          </label>
        </>
      )}
      <label>
        Voice
        <select value={voice} onChange={(e) => setVoice(e.target.value)} disabled={stats.running}>
          {voices.map((v) => <option key={v.id} value={v.id}>{v.name}</option>)}
        </select>
      </label>
      <label>
        Speed / quality
        <select
          value={preset}
          disabled={stats.running}
          onChange={(e) => {
            setPreset(e.target.value);
            setOverrides({});
          }}
        >
          {Object.keys(presets).map((p) => <option key={p} value={p}>{PRESET_LABELS[p] ?? p}</option>)}
        </select>
      </label>
      <label>
        Pitch shift ({pitch > 0 ? "+" : ""}{pitch} semitones)
        <input
          type="range" min={-24} max={24} step={1} value={pitch}
          onChange={(e) => {
            const semitones = Number(e.target.value);
            setPitch(semitones);
            if (stats.running) api.settings({ pitch_shift: semitones }).catch(() => undefined);
          }}
        />
      </label>

      <details>
        <summary>Advanced</summary>
        <label>
          Chunk (ms)
          <input type="number" value={value("chunk_ms")} disabled={stats.running}
            onChange={(e) => override("chunk_ms", Number(e.target.value))} />
        </label>
        <label>
          Crossfade (ms)
          <input type="number" value={value("crossfade_ms")} disabled={stats.running}
            onChange={(e) => override("crossfade_ms", Number(e.target.value))} />
        </label>
        <label>
          Context (ms)
          <input type="number" value={value("context_ms")} disabled={stats.running}
            onChange={(e) => override("context_ms", Number(e.target.value))} />
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
          <input type="number" min={0} max={1} step={0.05} value={value("index_rate")} disabled={stats.running}
            onChange={(e) => override("index_rate", Number(e.target.value))} />
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
          {stats.suggest_faster && <p className="hint">The GPU is not keeping up. Try a faster preset.</p>}
        </div>
      )}
    </section>
  );
}
