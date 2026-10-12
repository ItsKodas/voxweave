import { useEffect, useState } from "react";
import { api, Stats } from "./api";

export const IDLE: Stats = {
  running: false, mode: "off", error: null, infer_ms: 0, latency_ms: 0,
  in_level: 0, out_level: 0, underruns: 0, slow_chunks: 0, suggest_faster: false,
};

export const CRASH_MESSAGE = "The voice engine stopped unexpectedly. Live mode is off.";

/**
 * Live stats over WebSocket. Reconnects forever, so a restarted backend shows up as stopped.
 * Losing the socket while live means the backend died: that is reported as an error, which stays
 * until the backend reports something newer (running again, or an error of its own).
 */
export function useStats(): Stats {
  const [stats, setStats] = useState<Stats>(IDLE);
  useEffect(() => {
    let socket: WebSocket | null = null;
    let closed = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const retry = () => {
      if (!closed) timer = setTimeout(connect, 1000);
    };
    const connect = async () => {
      try {
        const url = await api.wsUrl();
        if (closed) return;
        const ws = new WebSocket(url);
        socket = ws;
        ws.onmessage = (event) => {
          try {
            const next: Stats = JSON.parse(event.data);
            setStats((prev) =>
              prev.error === CRASH_MESSAGE && !next.running && next.error === null ? { ...next, error: CRASH_MESSAGE } : next,
            );
          } catch {
            // ignore a malformed frame
          }
        };
        ws.onclose = () => {
          setStats((prev) => ({ ...IDLE, error: prev.running ? CRASH_MESSAGE : prev.error }));
          retry();
        };
      } catch {
        retry();
      }
    };
    connect();
    return () => {
      closed = true;
      clearTimeout(timer);
      socket?.close();
    };
  }, []);
  return stats;
}
