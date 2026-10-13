import { useEffect, useMemo, useRef, useState } from "react";
import { api, Device, DeviceList, Params, Stats, Voice } from "./api";

const PRESET_LABELS: Record<string, string> = {
  low_latency: "Low latency",
  balanced: "Balanced",
  high_quality: "High quality",
};

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
  const [preset, setPreset] = useState("balanced");
  const [overrides, setOverrides] = useState<Partial<Params>>({});
  const [pitch, setPitch] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [loaded, setLoaded] = useState(false);
  const [copied, setCopied] = useState(false);

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

  const load = () => {
    setError(null);
    setLoadFailed(false);
    Promise.all([api.devices(), api.voices(), api.presets()])
      .then(([d, v, p]) => {
        applyDevices(d, chooseDevices(d));
        setVoices(v);
        setPresets(p);
        setVoice(v[0]?.id ?? "");
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
        setVoices(list);
        setVoice((current) => (list.some((v) => v.id === current) ? current : (list[0]?.id ?? "")));
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
      }),
    );

  const base = presets[preset];
  const value = <K extends keyof Params>(key: K): Params[K] | "" =>
    (overrides[key] ?? base?.[key] ?? "") as Params[K] | "";
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
  const hasDestination = mic !== null || (monitorOn && monitor !== null);
  const canStart = !busy && !speechActive && input !== null && hasDestination && voice !== "";
  const shownError = error ?? actionError ?? stats.error;

  return (
    <section className="live">
      {shownError && <p role="alert" className="error">{shownError}</p>}
      {loadFailed && <button onClick={load}>Retry</button>}
      {speechActive && !stats.running && <p className="hint">Speech is playing. Stop it on the Speak tab to use live mode.</p>}
      {loaded && voices.length === 0 && <p className="hint">No voice models installed. See the README to add one.</p>}

      <label>
        Microphone
        <select value={input ?? ""} onChange={(e) => setInput(Number(e.target.value))} disabled={stats.running}>
          {inputs.map((d) => <option key={d.index} value={d.index}>{d.name}</option>)}
        </select>
      </label>
      <label>
        Virtual mic
        <select
          value={mic ?? ""}
          onChange={(e) => setMic(e.target.value === "" ? null : Number(e.target.value))}
          disabled={stats.running}
        >
          <option value="">None (listen only)</option>
          {outputs.map((d) => <option key={d.index} value={d.index}>{d.name}</option>)}
        </select>
      </label>
      {loaded && virtualMic === null && (
        <p className="hint">
          No virtual mic was found, so other apps will not hear anything until VoiceMeeter is installed and
          running. You can still listen to the converted voice yourself.
        </p>
      )}
      {loaded && (hasCable ? (
        <p className="hint">
          Other apps hear this app on the microphone called <strong>CABLE Output</strong>: choose it as the microphone
          in Discord or wherever you talk. You can rename it (for example to "AI Voice Mic") in Windows Sound settings,
          under Recording.
        </p>
      ) : (
        <details className="setup-mic">
          <summary>Set up a dedicated AI Voice microphone</summary>
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
            <li>In other apps, choose <strong>CABLE Output</strong> as the microphone. You can rename it to "AI Voice Mic" in Windows Sound settings.</li>
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
            if (stats.running) api.settings({ pitch_shift: semitones }).catch(reportFailure);
          }}
        />
      </label>

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
          {stats.suggest_faster && <p className="hint">The GPU is not keeping up. Try a faster preset.</p>}
        </div>
      )}
    </section>
  );
}
