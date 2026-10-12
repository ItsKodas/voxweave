import { useCallback, useEffect, useRef, useState } from "react";
import { api, Device } from "./api";
import { Clip, Prompt, studioApi } from "./studioApi";
import { useStudioStatus } from "./useStudioStatus";

export function RecordSession({ voiceId, active, onSaved }: { voiceId: string; active: boolean; onSaved: () => void }) {
  const [inputs, setInputs] = useState<Device[]>([]);
  const [input, setInput] = useState<number | null>(null);
  const [prompt, setPrompt] = useState<Prompt | null>(null);
  const [recording, setRecording] = useState(false);
  const [saved, setSaved] = useState<{ clip: Clip; url: string | null } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  // The microphone is live while recording, so teardown paths read these refs rather than stale state.
  const recordingRef = useRef(false);
  recordingRef.current = recording;
  const activeRef = useRef(active);
  activeRef.current = active;
  const mountedRef = useRef(true);
  const status = useStudioStatus(active && recording);

  const fail = (e: unknown) => setError((e as Error).message);

  const loadPrompt = useCallback(
    async (after?: number) => {
      setPrompt(await studioApi.prompt(voiceId, after));
      setSaved(null);
    },
    [voiceId],
  );

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
    };
  }, []);

  useEffect(() => {
    let gone = false;
    (async () => {
      try {
        const list = await api.devices();
        if (gone) return;
        const ins = list.devices.filter((d) => d.max_input > 0);
        setInputs(ins);
        setInput(ins.some((d) => d.index === list.default_input) ? list.default_input : (ins[0]?.index ?? null));
        await loadPrompt();
      } catch (e) {
        if (!gone) fail(e);
      }
    })();
    return () => {
      gone = true;
      if (recordingRef.current) studioApi.recordCancel().catch(() => undefined);
    };
  }, [loadPrompt]);

  // Leaving the tab must not leave the microphone running.
  useEffect(() => {
    if (!active && recordingRef.current) {
      setRecording(false);
      studioApi.recordCancel().catch(fail);
    }
  }, [active]);

  const run = async (action: () => Promise<void>) => {
    setBusy(true);
    setError(null);
    try {
      await action();
    } catch (e) {
      fail(e);
    } finally {
      setBusy(false);
    }
  };

  const record = () =>
    run(async () => {
      if (input === null || prompt?.index == null) return;
      await studioApi.recordStart(voiceId, input, prompt.index);
      if (!mountedRef.current || !activeRef.current) {
        // The user left while the request was in flight: do not leave the take running.
        await studioApi.recordCancel();
        return;
      }
      setSaved(null);
      setRecording(true);
    });

  const stop = () =>
    run(async () => {
      let result;
      try {
        result = await studioApi.recordStop();
      } catch (e) {
        // Unknown backend state: make it certainly idle, then show the original error.
        setRecording(false);
        await studioApi.recordCancel().catch(() => undefined);
        throw e;
      }
      setRecording(false);
      if (!result.accepted || !result.clip) {
        setError(result.message ?? "That take could not be used.");
        return;
      }
      const clip = result.clip;
      setSaved({ clip, url: null });
      onSaved();
      const url = await studioApi.clipAudioUrl(voiceId, clip.index);
      setSaved({ clip, url });
    });

  const cancel = () =>
    run(async () => {
      setRecording(false);
      await studioApi.recordCancel();
    });

  if (prompt && prompt.index === null) {
    return <p className="hint">Every sentence is recorded. Import more audio below, or redo clips from the list.</p>;
  }

  return (
    <section className="record">
      {error && <p role="alert" className="error">{error}</p>}
      <label>
        Microphone
        <select value={input ?? ""} disabled={recording} onChange={(e) => setInput(Number(e.target.value))}>
          {inputs.map((d) => <option key={d.index} value={d.index}>{d.name}</option>)}
        </select>
      </label>
      {inputs.length === 0 && <p className="hint">No microphone was found. Plug one in and use Refresh devices on the Live tab.</p>}

      {prompt && prompt.index !== null && (
        <>
          <p className="progress-line">
            Sentence {prompt.index + 1} of {prompt.total}. {prompt.recorded} recorded.
          </p>
          <p className="sentence">{prompt.text}</p>
        </>
      )}

      {recording ? (
        <>
          <div className="meter">
            <progress aria-label="Input level" max={1} value={Math.min(1, status?.level ?? 0)} />
          </div>
          <p>{(status?.seconds ?? 0).toFixed(1)} s{status?.full ? ". The take is at its maximum length." : ""}</p>
          <div className="actions">
            <button onClick={stop} disabled={busy}>Stop</button>
            <button className="secondary" onClick={cancel} disabled={busy}>Cancel</button>
          </div>
        </>
      ) : saved ? (
        <>
          <p className="ok">Saved. {saved.clip.seconds.toFixed(1)} s.</p>
          {saved.url && <audio aria-label="Play back the take" controls src={saved.url} />}
          <div className="actions">
            <button onClick={() => run(() => loadPrompt())} disabled={busy}>Next sentence</button>
            <button className="secondary" onClick={record} disabled={busy}>Redo</button>
          </div>
        </>
      ) : (
        <div className="actions">
          <button onClick={record} disabled={busy || input === null || prompt === null}>Record</button>
          <button className="secondary" disabled={busy || prompt === null}
            onClick={() => run(() => loadPrompt(prompt?.index ?? undefined))}>Skip</button>
        </div>
      )}
    </section>
  );
}
