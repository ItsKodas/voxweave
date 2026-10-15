import { useCallback, useEffect, useRef } from "react";
import { isActive, trainingApi, TrainingStatus } from "./trainingApi";

const RETRY_MS = 3000;

/**
 * Follows the app-wide training job whichever tab is shown, so that a model that finishes while the user waits on
 * the Live tab still appears there. Asks the backend once at start (a job may already be running) and then only
 * while a job is running or cancelling, once a second; when nothing is training it does not poll. The returned
 * function asks again now: a panel calls it when it has started or resumed a job. `onDone` is called once each
 * time the job being followed ends as done; `onStatus`, when given, with every answer (the title bar shows it).
 * One request at a time.
 */
export function useTrainingWatch(onDone: () => void, intervalMs = 1000,
                                 onStatus?: (status: TrainingStatus) => void): () => void {
  const done = useRef(onDone);
  done.current = onDone;
  const status = useRef(onStatus);
  status.current = onStatus;
  const kick = useRef<() => void>(() => {});

  useEffect(() => {
    let stopped = false;
    let inFlight = false;
    let again = false;
    let answered = false;
    /** The voice whose job was running at the last answer, null when none was. */
    let following: string | null = null;
    let timer: ReturnType<typeof setTimeout> | undefined;

    const run = async () => {
      if (inFlight) {
        again = true;
        return;
      }
      inFlight = true;
      clearTimeout(timer);
      try {
        const latest = await trainingApi.status();
        if (stopped) return;
        answered = true;
        status.current?.(latest);
        const active = isActive(latest);
        // The followed job finished, or another one is already running in its place: either way its outcome was
        // not seen running here any more, and a reload of the voice list is cheap.
        if (following !== null && (latest.state === "done" || (active && latest.voice_id !== following))) done.current();
        following = active ? latest.voice_id : null;
      } catch {
        // a job that was running is still followed; the next poll may succeed
      } finally {
        inFlight = false;
      }
      if (stopped) return;
      if (again) {
        again = false;
        void run();
      } else if (following !== null) {
        timer = setTimeout(run, intervalMs);
      } else if (!answered) {
        timer = setTimeout(run, RETRY_MS); // only until the first answer: after that, idle means silent
      }
    };

    kick.current = () => void run();
    void run();
    return () => {
      stopped = true;
      clearTimeout(timer);
      kick.current = () => {};
    };
  }, [intervalMs]);

  return useCallback(() => kick.current(), []);
}
