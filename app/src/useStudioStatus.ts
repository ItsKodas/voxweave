import { useEffect, useState } from "react";
import { StudioStatus, studioApi } from "./studioApi";

/** Polls recording level and import progress while enabled. One request at a time. */
export function useStudioStatus(enabled: boolean, intervalMs = 200): StudioStatus | null {
  const [status, setStatus] = useState<StudioStatus | null>(null);
  useEffect(() => {
    if (!enabled) {
      setStatus(null);
      return;
    }
    let stopped = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const poll = async () => {
      try {
        const next = await studioApi.status();
        if (!stopped) setStatus(next);
      } catch {
        // keep the last status; the next poll may succeed
      }
      if (!stopped) timer = setTimeout(poll, intervalMs);
    };
    poll();
    return () => {
      stopped = true;
      clearTimeout(timer);
    };
  }, [enabled, intervalMs]);
  return status;
}
