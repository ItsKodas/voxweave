import { useCallback, useEffect, useRef, useState } from "react";
import { TrainingStatus, isActive, trainingApi } from "./trainingApi";

export type TrainingView = {
  /** This voice's latest job, null until the first answer. */
  status: TrainingStatus | null;
  /** The app-wide latest job when it belongs to another voice, else null. */
  elsewhere: TrainingStatus | null;
  /** Asks again now (after an action); polling resumes by itself if a job is running. */
  refresh: () => void;
};

/**
 * Follows a voice's training from the backend's status. Polls once a second while enabled and a job (this voice's
 * or another's) is running or cancelling; otherwise asks once on mount, on every refresh() and whenever `enabled`
 * turns on again. One request at a time.
 */
export function useTrainingStatus(voiceId: string, enabled: boolean, intervalMs = 1000): TrainingView {
  const [status, setStatus] = useState<TrainingStatus | null>(null);
  const [elsewhere, setElsewhere] = useState<TrainingStatus | null>(null);
  const kick = useRef<() => void>(() => {});

  useEffect(() => {
    if (!enabled) return;
    let stopped = false;
    let inFlight = false;
    let again = false;
    let wasActive = false;
    let timer: ReturnType<typeof setTimeout> | undefined;

    const run = async () => {
      if (inFlight) {
        again = true;
        return;
      }
      inFlight = true;
      clearTimeout(timer);
      try {
        const own = await trainingApi.status(voiceId);
        let other: TrainingStatus | null = null;
        if (!isActive(own)) {
          const latest = await trainingApi.status();
          other = latest.voice_id !== null && latest.voice_id !== voiceId ? latest : null;
        }
        if (!stopped) {
          setStatus(own);
          setElsewhere(other);
          wasActive = isActive(own) || isActive(other);
        }
      } catch {
        // keep what is shown; the next poll may succeed
      }
      inFlight = false;
      if (stopped) return;
      if (again) {
        again = false;
        void run();
      } else if (wasActive) {
        timer = setTimeout(run, intervalMs);
      }
    };

    kick.current = () => void run();
    void run();
    return () => {
      stopped = true;
      clearTimeout(timer);
      kick.current = () => {};
    };
  }, [voiceId, enabled, intervalMs]);

  const refresh = useCallback(() => kick.current(), []);
  return { status, elsewhere, refresh };
}
