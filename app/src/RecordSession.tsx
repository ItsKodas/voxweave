import { useCallback, useEffect, useRef, useState } from "react";
import { api, Device, DeviceList } from "./api";
import { Clip, Prompt, PromptKind, studioApi } from "./studioApi";
import { PageReader } from "./PageReader";
import { useStudioStatus } from "./useStudioStatus";

/** Takes with more than 20 s of speech are rejected, so the warning comes while there is time to finish. */
const LONG_TAKE_SECONDS = 15;
const LONG_TAKE_WARNING = "Takes over 20 seconds are rejected. Finish the sentence and press Stop.";

/** `devicesVersion` changes whenever Refresh devices on the Live tab has reloaded the device lists. */
export function RecordSession({ voiceId, active, devicesVersion = 0, onSaved }: {
  voiceId: string;
  active: boolean;
  devicesVersion?: number;
  onSaved: () => void;
}) {
  const [inputs, setInputs] = useState<Device[]>([]);
  const [input, setInput] = useState<number | null>(null);
  const [prompt, setPrompt] = useState<Prompt | null>(null);
  // Paragraphs are two or three sentences in one take: more speech per press of Record.
  const [kind, setKind] = useState<PromptKind>("sentences");
  // A whole page read in one go, cut into clips afterwards (PageReader).
  const [pageMode, setPageMode] = useState(false);
  const kindRef = useRef(kind);
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
  const inputsRef = useRef(inputs);
  inputsRef.current = inputs;
  const inputRef = useRef(input);
  inputRef.current = input;
  const status = useStudioStatus(active && recording);

  const fail = (e: unknown) => setError((e as Error).message);

  const loadPrompt = useCallback(
    async (after?: number) => {
      setPrompt(await studioApi.prompt(voiceId, after, kindRef.current));
      setSaved(null);
    },
    [voiceId],
  );

  // Release a take's object URL when it is replaced or the component unmounts.
  const takeUrl = saved?.url ?? null;
  useEffect(() => {
    if (takeUrl === null) return;
    return () => URL.revokeObjectURL(takeUrl);
  }, [takeUrl]);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
    };
  }, []);

  // Shows the input devices of a fresh list. With `keep`, the chosen microphone stays chosen if it is still there.
  // It is found by name, because the system may number devices differently after a refresh.
  const showDevices = (list: DeviceList, keep: boolean) => {
    const ins = list.devices.filter((d) => d.max_input > 0);
    const chosen = keep ? inputsRef.current.find((d) => d.index === inputRef.current) : undefined;
    const same = chosen && (ins.find((d) => d.index === chosen.index && d.name === chosen.name) ?? ins.find((d) => d.name === chosen.name));
    setInputs(ins);
    setInput(same ? same.index : ins.some((d) => d.index === list.default_input) ? list.default_input : (ins[0]?.index ?? null));
  };

  useEffect(() => {
    let gone = false;
    (async () => {
      try {
        const list = await api.devices();
        if (gone) return;
        showDevices(list, false);
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

  // Refresh devices on the Live tab reloads the microphone list here too. Never during a take: the list is
  // fetched, or fetched again, once the take is over.
  const shownVersion = useRef<number | null>(devicesVersion);
  useEffect(() => {
    if (recording || shownVersion.current === devicesVersion) return;
    shownVersion.current = devicesVersion;
    let gone = false;
    let settled = false;
    api
      .devices()
      .then((list) => {
        if (gone || recordingRef.current) return;
        settled = true;
        showDevices(list, true);
      })
      .catch((e) => {
        if (gone) return;
        settled = true;
        fail(e);
      });
    return () => {
      gone = true;
      if (!settled) shownVersion.current = null; // overtaken by a take or a newer refresh: ask again next time
    };
  }, [devicesVersion, recording]);

  // Leaving the tab must not leave the microphone running. If the cancel fails the take may still be running,
  // so the controls stay and the cancel is tried again when the tab is active again.
  const retryCancel = useRef(false);
  useEffect(() => {
    if (!recordingRef.current || (active && !retryCancel.current)) return;
    studioApi
      .recordCancel()
      .then(() => {
        retryCancel.current = false;
        if (mountedRef.current) setRecording(false);
      })
      .catch((e) => {
        retryCancel.current = true;
        if (mountedRef.current) fail(e);
      });
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
        // Unknown backend state: make it certainly idle, then show the original error. If even that
        // fails, keep the controls so the user can try again.
        await studioApi.recordCancel().then(() => setRecording(false), () => undefined);
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
      if (!mountedRef.current) {
        URL.revokeObjectURL(url);
        return;
      }
      setSaved({ clip, url });
    });

  const cancel = () =>
    run(async () => {
      await studioApi.recordCancel();
      setRecording(false);
    });

  const paragraphs = kind === "paragraphs";
  const chooser = (
    <div role="radiogroup" aria-label="What to read" className="kind">
      {(["sentences", "paragraphs"] as const).map((k) => (
        <label key={k}>
          <input type="radio" name="prompt-kind" checked={!pageMode && kind === k} disabled={recording || busy}
            onChange={() => {
              kindRef.current = k;
              setKind(k);
              setPageMode(false);
              run(() => loadPrompt());
            }} />
          {k === "sentences" ? "Sentences" : "Paragraphs"}
        </label>
      ))}
      <label>
        <input type="radio" name="prompt-kind" checked={pageMode} disabled={recording || busy}
          onChange={() => setPageMode(true)} />
        A full page
      </label>
    </div>
  );

  const microphone = (
    <>
      <label>
        Microphone
        <select value={input ?? ""} disabled={recording} onChange={(e) => setInput(Number(e.target.value))}>
          {inputs.map((d) => <option key={d.index} value={d.index}>{d.name}</option>)}
        </select>
      </label>
      {inputs.length === 0 && <p className="hint">No microphone was found. Plug one in and use Refresh devices on the Live tab.</p>}
    </>
  );

  if (pageMode) {
    return (
      <section className="record">
        {error && <p role="alert" className="error">{error}</p>}
        {microphone}
        {chooser}
        <PageReader voiceId={voiceId} input={input} active={active} onSaved={onSaved} />
      </section>
    );
  }

  if (prompt && prompt.index === null) {
    return (
      <section className="record">
        {chooser}
        <p className="hint">
          Every {paragraphs ? "paragraph" : "sentence"} is recorded. {paragraphs ? "Switch to sentences" : "Switch to paragraphs"},
          import more audio below, or redo clips from the list.
        </p>
      </section>
    );
  }

  return (
    <section className="record">
      {error && <p role="alert" className="error">{error}</p>}
      {microphone}

      {chooser}
      {paragraphs && <p className="hint">Paragraphs take 10 to 15 seconds to read. Keep each take under 20 seconds.</p>}
      {prompt && prompt.index !== null && (
        <>
          <p className="progress-line">
            {paragraphs ? "Paragraph" : "Sentence"} {prompt.number ?? prompt.index + 1} of {prompt.total}. {prompt.recorded} recorded.
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
          {(status?.seconds ?? 0) > LONG_TAKE_SECONDS && <p className="hint">{LONG_TAKE_WARNING}</p>}
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
            <button onClick={() => run(() => loadPrompt())} disabled={busy}>{paragraphs ? "Next paragraph" : "Next sentence"}</button>
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
