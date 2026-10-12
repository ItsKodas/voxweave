import { useEffect, useState } from "react";
import { api, Stats } from "./api";

export const IDLE: Stats = {
  running: false, mode: "off", error: null, infer_ms: 0, latency_ms: 0,
  in_level: 0, out_level: 0, underruns: 0, slow_chunks: 0, suggest_faster: false,
};

/** Live stats over WebSocket. Reconnects forever, so a restarted backend shows up as stopped. */
export function useStats(): Stats {
  const [stats, setStats] = useState<Stats>(IDLE);
  useEffect(() => {
    let socket: WebSocket | null = null;
    let closed = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const connect = async () => {
      socket = new WebSocket(await api.wsUrl());
      socket.onmessage = (event) => setStats(JSON.parse(event.data));
      socket.onclose = () => {
        setStats((prev) => ({ ...IDLE, error: prev.error }));
        if (!closed) timer = setTimeout(connect, 1000);
      };
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
