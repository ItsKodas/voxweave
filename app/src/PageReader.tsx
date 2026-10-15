import { useEffect, useRef, useState } from "react";
import { Page, studioApi } from "./studioApi";
import { useStudioStatus } from "./useStudioStatus";

/** 75.2 -> "1:15". */
const clock = (seconds: number) => {
  const s = Math.max(0, Math.floor(seconds));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
};

type Phase = "ready" | "recording" | "importing" | "done";

/**
 * Reading a whole page in one go: the page opens full screen, the reading is recorded as one long take, and the
 * importer slices it into clips (transcribed, sorted into speech and sounds) like a dropped audio file.
 */
export function PageReader({ voiceId, input, active, onSaved }: {
  voiceId: string;
  input: number | null;
  active: boolean;
  onSaved: () => void;
}) {
  const [pages, setPages] = useState<Page[]>([]);
  const [open, setOpen] = useState<Page | null>(null);
  const [phase, setPhase] = useState<Phase>("ready");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [job, setJob] = useState<number | null>(null);
  const [readSeconds, setReadSeconds] = useState(0);
  const [result, setResult] = useState<{ clips: number; errors: string[] } | null>(null);
  const status = useStudioStatus(active && (phase === "recording" || phase === "importing"));
  const recordingRef = useRef(false);
  recordingRef.current = phase === "recording";

  useEffect(() => {
    let gone = false;
    studioApi.pages().then((p) => !gone && setPages(p), (e: Error) => !gone && setError(e.message));
    return () => {
      gone = true;
      if (recordingRef.current) studioApi.recordCancel().catch(() => undefined);
    };
  }, []);

  // Leaving the tab must not leave the microphone running.
  useEffect(() => {
    if (active || !recordingRef.current) return;
    studioApi.recordCancel().then(() => setPhase("ready"), (e: Error) => setError(e.message));
  }, [active]);

  // The import of this reading: done once the importer reports this job finished.
  const imported = status?.import;
  useEffect(() => {
    if (phase !== "importing" || !imported || job === null) return;
    if (imported.job === job && imported.finished && !imported.active) {
      setResult({ clips: imported.clips_added, errors: imported.errors });
      setPhase("done");
      onSaved();
    }
  }, [phase, imported, job, onSaved]);

  const run = async (action: () => Promise<void>) => {
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
    run(async () => {
      if (open === null || input === null) return;
      await studioApi.recordPage(voiceId, input, open.index);
      setResult(null);
      setPhase("recording");
    });

  const stop = () =>
    run(async () => {
      let res;
      try {
        res = await studioApi.recordStop();
      } catch (e) {
        await studioApi.recordCancel().catch(() => undefined);
        setPhase("ready");
        throw e;
      }
      if (!res.accepted || !res.imported) {
        setPhase("ready");
        setError(res.message ?? "That reading could not be used.");
        return;
      }
      setReadSeconds(res.seconds ?? 0);
      const now = await studioApi.status();
      setJob(now.import.job);
      setPhase("importing");
    });

  const cancel = () =>
    run(async () => {
      await studioApi.recordCancel();
      setPhase("ready");
    });

  const close = () => {
    setOpen(null);
    setPhase("ready");
    setResult(null);
    setError(null);
  };

  return (
    <div className="pages">
      <p className="hint">
        Read a whole page aloud in one go, at your own pace. It is cut into clips for you afterwards, so a page of
        two or three minutes adds as much as dozens of sentences. Pauses, stumbles and re-reading a line are fine.
      </p>
      <div className="page-list">
        {pages.map((p) => (
          <button key={p.index} className="secondary" onClick={() => setOpen(p)}>
            {p.title}, about {Math.max(1, Math.round(p.minutes))} minutes
          </button>
        ))}
      </div>
      {error && open === null && <p role="alert" className="error">{error}</p>}

      {open !== null && (
        <div className="page-reader" role="dialog" aria-label={open.title}>
          <div className="page-bar">
            <strong>{open.title}</strong>
            {phase === "recording" ? (
              <>
                <progress aria-label="Input level" max={1} value={Math.min(1, status?.level ?? 0)} />
                <span className="page-clock">{clock(status?.seconds ?? 0)}</span>
                <button onClick={stop} disabled={busy}>Stop and add to the voice</button>
                <button className="secondary" onClick={cancel} disabled={busy}>Cancel</button>
              </>
            ) : phase === "importing" ? (
              <span>
                Adding your reading to the voice: {imported?.job === job ? imported?.clips_added ?? 0 : 0} clips so far...
              </span>
            ) : (
              <>
                <button onClick={start} disabled={busy || input === null}>Start reading</button>
                <button className="secondary" onClick={close} disabled={busy}>Close</button>
              </>
            )}
          </div>
          {error && <p role="alert" className="error">{error}</p>}
          {input === null && <p className="hint">Choose a microphone first.</p>}
          {phase === "done" && result && (
            <p className="ok">
              Added {result.clips} clips from {clock(readSeconds)} of reading.
              {result.errors.length > 0 && ` ${result.errors.join(" ")}`} Read another page, or close this one.
            </p>
          )}
          <div className="page-text">
            {open.text.split(/\n\s*\n/).map((para, i) => <p key={i}>{para}</p>)}
          </div>
        </div>
      )}
    </div>
  );
}
