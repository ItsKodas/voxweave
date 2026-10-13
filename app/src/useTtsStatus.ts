import { useCallback, useEffect, useRef, useState } from "react";
import { TtsStatus, isFollowing, ttsApi } from "./ttsApi";

const RETRY_MS = 3000;

export type TtsView = {
  /** The last answer, null until the first one. */
  status: TtsStatus | null;
  /** The last attempt to get a status failed (what is shown may be out of date). */
  unreachable: boolean;
  /** Asks again now. */
  refresh: () => void;
  /** Takes an answer the caller already has (the response to speak or stop) and follows it from there. */
  apply: (status: TtsStatus) => void;
};

/**
 * Follows text to speech from the backend's status. Asks once on mount and once each time `visible` turns on; polls
 * twice a second while a request is active (speaking, preparing, stopping, or the engine still starting), whichever
 * tab is shown, so the other tabs know when the speakers are in use. After a failed request it tries again every 3
 * seconds until one succeeds. One request at a time; nothing after unmount.
 */
export function useTtsStatus(visible: boolean, intervalMs = 500): TtsView {
  const [status, setStatus] = useState<TtsStatus | null>(null);
  const [failed, setFailed] = useState(false);
  const kick = useRef<() => void>(() => {});
  const take = useRef<(s: TtsStatus) => void>(() => {});

  useEffect(() => {
    let stopped = false;
    let inFlight = false;
    let again = false;
    let active = false;
    /** Counts the statuses handed in with apply(): an answer requested before the latest one is out of date. */
    let generation = 0;
    let timer: ReturnType<typeof setTimeout> | undefined;

    const schedule = (delay: number) => {
      clearTimeout(timer);
      timer = setTimeout(run, delay);
    };

    async function run() {
      if (inFlight) {
        again = true;
        return;
      }
      inFlight = true;
      clearTimeout(timer);
      let ok = false;
      const asked = generation;
      try {
        const next = await ttsApi.status();
        if (asked !== generation) {
          again = true; // newer information arrived meanwhile: this answer is stale, ask again
        } else if (!stopped) {
          ok = true;
          active = isFollowing(next);
          setStatus(next);
          setFailed(false);
        }
      } catch {
        if (!stopped) setFailed(true); // keep what is shown; the next attempt may succeed
      }
      inFlight = false;
      if (stopped) return;
      if (again) {
        again = false;
        void run();
      } else if (!ok) {
        schedule(RETRY_MS);
      } else if (active) {
        schedule(intervalMs);
      }
    }

    kick.current = () => void run();
    take.current = (next) => {
      if (stopped) return;
      generation++;
      active = isFollowing(next);
      setStatus(next);
      setFailed(false);
      if (active && !inFlight) schedule(intervalMs);
    };
    void run();
    return () => {
      stopped = true;
      clearTimeout(timer);
      kick.current = () => {};
      take.current = () => {};
    };
  }, [intervalMs]);

  const first = useRef(true);
  useEffect(() => {
    if (first.current) {
      first.current = false; // the mount already asked
      return;
    }
    if (visible) kick.current();
  }, [visible]);

  const refresh = useCallback(() => kick.current(), []);
  const apply = useCallback((next: TtsStatus) => take.current(next), []);
  return { status, unreachable: failed, refresh, apply };
}
