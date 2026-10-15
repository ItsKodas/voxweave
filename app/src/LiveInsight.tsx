import { useEffect, useRef, useState } from "react";
import { call } from "./api";

/** The last seconds of what live mode did (GET /live/insight; see backend/aivoice/insight.py). */
export type Insight = {
  running: boolean;
  frames_per_s?: number;
  seconds?: number;
  heard?: number[];
  sent?: number[];
  corrected?: boolean[];
  gated?: boolean[];
  in_db?: number[];
  out_db?: number[];
  chunks?: { t: number; infer_ms: number; budget_ms: number }[];
  match?: { t: number; value: number }[];
  match_note?: string | null;
};

export const insightApi = { get: () => call<Insight>("GET", "/live/insight") };

const POLL_MS = 300;
const COLORS = { heard: "#8a90a0", sent: "#4f7cff", fixed: "#f2c66d", gated: "rgba(242, 198, 109, 0.12)", out: "#7bd88f", grid: "#2c303a" };

function canvasContext(canvas: HTMLCanvasElement | null) {
  if (!canvas) return null;
  try {
    const ctx = canvas.getContext("2d");
    if (!ctx) return null;
    const ratio = window.devicePixelRatio || 1;
    const { width, height } = canvas.getBoundingClientRect();
    if (width === 0 || height === 0) return null;
    canvas.width = Math.round(width * ratio);
    canvas.height = Math.round(height * ratio);
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.clearRect(0, 0, width, height);
    return { ctx, width, height };
  } catch {
    return null; // no canvas (tests): the numbers beside the graphs still show
  }
}

function shadeGated(ctx: CanvasRenderingContext2D, gated: boolean[], width: number, height: number) {
  const step = width / Math.max(1, gated.length);
  ctx.fillStyle = COLORS.gated;
  let from = -1; // each muted stretch as one block, so the shading does not overlap itself
  gated.forEach((g, i) => {
    if (g && from < 0) from = i;
    if ((!g || i === gated.length - 1) && from >= 0) {
      const to = g ? i + 1 : i;
      ctx.fillRect(from * step, 0, (to - from) * step, height);
      from = -1;
    }
  });
}

function line(ctx: CanvasRenderingContext2D, values: number[], y: (v: number) => number | null, width: number, color: string) {
  const step = width / Math.max(1, values.length);
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  let drawing = false;
  values.forEach((v, i) => {
    const py = y(v);
    if (py === null) {
      drawing = false;
      return;
    }
    if (drawing) ctx.lineTo(i * step, py);
    else ctx.moveTo(i * step, py);
    drawing = true;
  });
  ctx.stroke();
}

/** Pitch on a log scale from 60 Hz to 800 Hz: your voice, what the model was given, and where the clean-up stepped in. */
function drawPitch(canvas: HTMLCanvasElement | null, data: Insight) {
  const c = canvasContext(canvas);
  if (!c || !data.heard || !data.sent) return;
  const { ctx, width, height } = c;
  const low = Math.log(60), high = Math.log(800);
  const y = (hz: number) => (hz > 0 ? height - ((Math.log(hz) - low) / (high - low)) * height : null);
  shadeGated(ctx, data.gated ?? [], width, height);
  ctx.strokeStyle = COLORS.grid;
  for (const hz of [100, 200, 400]) {
    const py = y(hz)!;
    ctx.beginPath();
    ctx.moveTo(0, py);
    ctx.lineTo(width, py);
    ctx.stroke();
  }
  line(ctx, data.heard, y, width, COLORS.heard);
  line(ctx, data.sent, y, width, COLORS.sent);
  const step = width / Math.max(1, data.sent.length);
  ctx.fillStyle = COLORS.fixed;
  (data.corrected ?? []).forEach((fixed, i) => {
    const py = y(data.sent![i]);
    if (fixed && py !== null) ctx.fillRect(i * step - 1, py - 1, 3, 3);
  });
}

/** Levels from -80 dB to 0 dB: your microphone and the converted voice. */
function drawLevels(canvas: HTMLCanvasElement | null, data: Insight) {
  const c = canvasContext(canvas);
  if (!c || !data.in_db || !data.out_db) return;
  const { ctx, width, height } = c;
  const y = (db: number) => height - (Math.min(0, Math.max(-80, db)) + 80) / 80 * height;
  shadeGated(ctx, data.gated ?? [], width, height);
  line(ctx, data.in_db, y, width, COLORS.heard);
  line(ctx, data.out_db, y, width, COLORS.out);
}

/** Voice match over the last two minutes, 0 to 100 percent. */
function drawMatch(canvas: HTMLCanvasElement | null, data: Insight) {
  const c = canvasContext(canvas);
  if (!c || !data.match || data.match.length < 2) return;
  const { ctx, width, height } = c;
  const values = data.match.map((m) => m.value);
  ctx.strokeStyle = COLORS.grid;
  for (const v of [0.25, 0.5, 0.75]) {
    ctx.beginPath();
    ctx.moveTo(0, height - v * height);
    ctx.lineTo(width, height - v * height);
    ctx.stroke();
  }
  line(ctx, values, (v) => height - Math.max(0, Math.min(1, v)) * height, width, COLORS.sent);
}

const median = (xs: number[]) => {
  if (xs.length === 0) return 0;
  const s = [...xs].sort((a, b) => a - b);
  return s[Math.floor(s.length / 2)];
};

/** Live graphs of what the voice conversion is doing, shown while live mode runs. */
export function LiveInsight({ running, voiceName }: { running: boolean; voiceName: string }) {
  const [data, setData] = useState<Insight | null>(null);
  const pitch = useRef<HTMLCanvasElement>(null);
  const levels = useRef<HTMLCanvasElement>(null);
  const match = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (!running) {
      setData(null);
      return;
    }
    let gone = false;
    let timer: ReturnType<typeof setTimeout>;
    const poll = async () => {
      try {
        const next = await insightApi.get();
        if (!gone) setData(next);
      } catch {
        // the next poll may succeed
      }
      if (!gone) timer = setTimeout(poll, POLL_MS);
    };
    poll();
    return () => {
      gone = true;
      clearTimeout(timer);
    };
  }, [running]);

  useEffect(() => {
    if (!data?.running) return;
    drawPitch(pitch.current, data);
    drawLevels(levels.current, data);
    drawMatch(match.current, data);
  }, [data]);

  if (!running || !data?.running) return null;
  const latest = data.match && data.match.length > 0 ? data.match[data.match.length - 1].value : null;
  const recent = (data.chunks ?? []).slice(-8);
  const infer = median(recent.map((c) => c.infer_ms));
  const budget = recent.length ? recent[recent.length - 1].budget_ms : 0;
  const sent = (data.sent ?? []).filter((x) => x > 0);
  const heard = (data.heard ?? []).filter((x) => x > 0);
  const fixedShare = data.corrected && data.corrected.length
    ? data.corrected.filter(Boolean).length / Math.max(1, (data.sent ?? []).filter((x) => x > 0).length) : 0;
  const gatedShare = data.gated && data.gated.length ? data.gated.filter(Boolean).length / data.gated.length : 0;

  return (
    <section className="insight" aria-label="What live mode is doing">
      <h2>What live mode is doing</h2>
      <div className="insight-row">
        <div className="insight-label">
          <strong>Voice match: {latest === null ? "..." : `${Math.round(latest * 100)}%`}</strong>
          <span>How much you sound like {voiceName}, from the last three seconds of speech.</span>
          {data.match_note && <span className="hint">{data.match_note}</span>}
        </div>
        <canvas ref={match} className="insight-graph" aria-label="Voice match over time" />
      </div>
      <div className="insight-row">
        <div className="insight-label">
          <strong>Pitch</strong>
          <span>
            <i className="key" style={{ background: COLORS.heard }} /> yours
            {heard.length > 0 && ` (${Math.round(median(heard))} Hz)`},{" "}
            <i className="key" style={{ background: COLORS.sent }} /> given to the model
            {sent.length > 0 && ` (${Math.round(median(sent))} Hz)`},{" "}
            <i className="key" style={{ background: COLORS.fixed }} /> cleaned up ({Math.round(fixedShare * 100)}% of the last 10 s)
          </span>
        </div>
        <canvas ref={pitch} className="insight-graph" aria-label="Pitch over the last ten seconds" />
      </div>
      <div className="insight-row">
        <div className="insight-label">
          <strong>Level</strong>
          <span>
            <i className="key" style={{ background: COLORS.heard }} /> your microphone,{" "}
            <i className="key" style={{ background: COLORS.out }} /> converted voice; shaded: background turned down
            ({Math.round(gatedShare * 100)}%)
          </span>
        </div>
        <canvas ref={levels} className="insight-graph" aria-label="Levels over the last ten seconds" />
      </div>
      <div className="insight-row">
        <div className="insight-label">
          <strong>Speed</strong>
          <span>
            Converting takes {Math.round(infer)} ms of each {budget} ms chunk
            {budget > 0 && infer > budget ? ": too slow, move Quality further left." : "."}
          </span>
        </div>
        <progress aria-label="Conversion time per chunk" max={Math.max(1, budget)} value={Math.min(infer, budget)} />
      </div>
    </section>
  );
}
