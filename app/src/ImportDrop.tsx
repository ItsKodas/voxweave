import { useEffect, useRef, useState } from "react";
import { onFilesDropped } from "./dropEvents";
import { studioApi } from "./studioApi";
import { useStudioStatus } from "./useStudioStatus";

export function ImportDrop({ voiceId, active, onImported }: { voiceId: string; active: boolean; onImported: () => void }) {
  const [started, setStarted] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const status = useStudioStatus(active && started);
  const job = status?.import ?? null;
  const running = started && !(job?.finished ?? false);
  const runningRef = useRef(running);
  runningRef.current = running;
  const reported = useRef(false);

  useEffect(() => {
    if (!active) return;
    let unsubscribe: (() => void) | null = null;
    let gone = false;
    const handle = (paths: string[]) => {
      if (runningRef.current) {
        setNotice("Wait for the current import to finish before dropping more files.");
        return;
      }
      setError(null);
      setNotice(null);
      reported.current = false;
      studioApi
        .importFiles(voiceId, paths)
        .then(() => setStarted(true))
        .catch((e: Error) => setError(e.message));
    };
    onFilesDropped(handle).then((off) => {
      if (gone) off();
      else unsubscribe = off;
    });
    return () => {
      gone = true;
      unsubscribe?.();
    };
  }, [active, voiceId]);

  // An import outlives this component: on mount, pick up a job that is already running.
  useEffect(() => {
    if (!active) return;
    let gone = false;
    studioApi
      .status()
      .then((s) => {
        if (!gone && s.import.active) setStarted(true);
      })
      .catch(() => undefined);
    return () => {
      gone = true;
    };
  }, [active]);

  useEffect(() => {
    if (started && job?.finished && !reported.current) {
      reported.current = true;
      onImported();
    }
  }, [started, job?.finished, onImported]);

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
      {started && job && !job.finished && (
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
      {started && job?.finished && (
        <div>
          <p className="ok">
            Added {job.clips_added} clips.{job.skipped > 0 ? ` ${job.skipped} skipped as too quiet.` : ""}
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
