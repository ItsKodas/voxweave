import { useEffect, useRef, useState } from "react";
import { onFilesDropped } from "./dropEvents";
import { ImportStatus, studioApi } from "./studioApi";
import { useStudioStatus } from "./useStudioStatus";

const WAIT_HERE = "Wait for the current import to finish before dropping more files.";
const WAIT_OTHER = "An import into another voice is running. Wait for it to finish before importing here.";
const QUIET_HINT = "Quiet recordings can be made louder in an audio editor and imported again.";

export function ImportDrop({ voiceId, active, onImported }: { voiceId: string; active: boolean; onImported: () => void }) {
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  // The number of this voice's import job that this component follows. The backend keeps one status for the
  // whole app, so a job is "mine" only when it carries my voice id and the number I started or picked up.
  const [tracked, setTracked] = useState<number | null>(null);
  // Polling runs only while an import is running. When my job finishes, its last status is kept here so the
  // result stays on screen with no further requests; the next drop clears it and starts polling again.
  const [watching, setWatching] = useState(false);
  const [result, setResult] = useState<ImportStatus | null>(null);
  const status = useStudioStatus(active && watching);
  const polled = status?.import ?? null;
  const job = polled ?? result;
  const jobRef = useRef(job);
  jobRef.current = job;
  const trackedRef = useRef(tracked);
  trackedRef.current = tracked;
  const starting = useRef(false);
  const reportedJob = useRef<number | null>(null);

  const forMe = job !== null && job.voice_id === voiceId;
  const mine = forMe && tracked === job.job;
  const otherRunning = job !== null && job.active && job.voice_id !== voiceId;

  useEffect(() => {
    if (!active) return;
    let unsubscribe: (() => void) | null = null;
    let gone = false;
    const handle = (paths: string[]) => {
      if (gone) return;
      const current = jobRef.current;
      if (starting.current || current?.active) {
        // An import into another voice already has its own notice on screen.
        setNotice(current?.active && current.voice_id !== voiceId ? null : WAIT_HERE);
        return;
      }
      starting.current = true;
      setError(null);
      setNotice(null);
      setTracked(null); // the previous result is no longer shown, and a stale finished status is not this job's
      setResult(null);
      studioApi
        .importFiles(voiceId, paths)
        .then((res) => {
          setTracked(res.import.job);
          setWatching(true);
        })
        .catch((e: Error) => setError(e.message))
        .finally(() => {
          starting.current = false;
        });
    };
    onFilesDropped(handle)
      .then((off) => {
        if (gone) off();
        else unsubscribe = off;
      })
      .catch((e: Error) => !gone && setError(e.message));
    return () => {
      gone = true;
      unsubscribe?.();
    };
  }, [active, voiceId]);

  // An import outlives this component. On mount (and when the tab comes back) read the status once: watch a job
  // that is running, or one of mine that finished while the tab was hidden. A job already finished at mount is not announced.
  useEffect(() => {
    if (!active) return;
    let gone = false;
    studioApi
      .status()
      .then((s) => {
        const j = s.import;
        if (gone) return;
        if (j.active || (j.finished && j.voice_id === voiceId && j.job === trackedRef.current)) setWatching(true);
      })
      .catch(() => undefined);
    return () => {
      gone = true;
    };
  }, [active, voiceId]);

  // A running job of this voice that this component did not start (picked up on mount) becomes mine.
  useEffect(() => {
    if (job !== null && job.active && job.voice_id === voiceId && tracked !== job.job) setTracked(job.job);
  }, [job, voiceId, tracked]);

  // Nothing is running any more: stop polling, keeping my finished job's status as the result.
  useEffect(() => {
    if (polled === null || polled.active) return;
    if (tracked !== null && polled.job < tracked) return; // an older job's status; mine has not been seen yet
    if (polled.finished && polled.voice_id === voiceId && polled.job === tracked) setResult(polled);
    setWatching(false);
  }, [polled, tracked, voiceId]);

  useEffect(() => {
    if (mine && job.finished && reportedJob.current !== job.job) {
      reportedJob.current = job.job;
      onImported();
    }
  }, [mine, job, onImported]);

  return (
    <section className="import">
      {error && <p role="alert" className="error">{error}</p>}
      <div className="dropzone">
        <p>Drop audio files anywhere on this window to add them to this voice.</p>
        <p className="hint">
          Each file is cut into clips at pauses and transcribed in English. The first import downloads the speech
          model, about 500 MB.
        </p>
      </div>
      {notice && <p className="hint">{notice}</p>}
      {otherRunning && <p className="hint">{WAIT_OTHER}</p>}
      {forMe && job.active && (
        <div>
          <p>
            Importing file {Math.min(job.done_files + 1, job.total_files)} of {job.total_files}
            {job.current ? `: ${job.current}` : ""}. {job.clips_added} clips so far.
          </p>
          <progress max={job.total_files} value={job.done_files} aria-label="Import progress" />
          <div className="actions">
            <button className="secondary" onClick={() => studioApi.importCancel().catch((e: Error) => setError(e.message))}>
              Cancel import
            </button>
          </div>
        </div>
      )}
      {mine && job.finished && (
        <div>
          <p className="ok">
            Added {job.clips_added} clips.{job.skipped > 0 ? ` ${job.skipped} skipped as too quiet. ${QUIET_HINT}` : ""}
          </p>
          {job.errors.length > 0 && (
            <ul className="errors">
              {job.errors.map((line, i) => <li key={i}>{line}</li>)}
            </ul>
          )}
        </div>
      )}
    </section>
  );
}
