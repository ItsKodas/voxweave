import { render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { insightApi, LiveInsight } from "./LiveInsight";

const DATA = {
  running: true, frames_per_s: 50, seconds: 12.5,
  heard: [0, 110, 112, 115, 0], sent: [0, 246, 251, 258, 0],
  corrected: [false, true, false, false, false], gated: [true, false, false, false, true],
  in_db: [-70, -25, -24, -23, -70], out_db: [-90, -22, -21, -20, -90],
  chunks: [{ t: 12.0, infer_ms: 130, budget_ms: 750 }, { t: 12.5, infer_ms: 150, budget_ms: 750 }],
  match: [{ t: 11, value: 0.81 }, { t: 12, value: 0.86 }], match_note: null,
};

beforeEach(() => {
  vi.restoreAllMocks();
});

describe("LiveInsight", () => {
  it("shows the voice match, pitch, levels and speed while live mode runs", async () => {
    vi.spyOn(insightApi, "get").mockResolvedValue(DATA);
    render(<LiveInsight running voiceName="Woman" />);
    expect(await screen.findByText("Voice match: 86%")).toBeInTheDocument();
    expect(screen.getByText(/How much you sound like Woman/)).toBeInTheDocument();
    expect(screen.getByText(/yours \(112 Hz\)/)).toBeInTheDocument();
    expect(screen.getByText(/given to the model \(251 Hz\)/)).toBeInTheDocument();
    expect(screen.getByText(/cleaned up \(33% of the last 10 s\)/)).toBeInTheDocument();
    expect(screen.getByText(/background turned down\s*\(40%\)/)).toBeInTheDocument();
    expect(screen.getByText(/Converting takes 150 ms of each 750 ms chunk\./)).toBeInTheDocument();
  });

  it("explains a voice match that is not available", async () => {
    vi.spyOn(insightApi, "get").mockResolvedValue({ ...DATA, match: [], match_note: "Voice match needs the speaker model." });
    render(<LiveInsight running voiceName="Woman" />);
    expect(await screen.findByText("Voice match needs the speaker model.")).toBeInTheDocument();
    expect(screen.getByText("Voice match: ...")).toBeInTheDocument();
  });

  it("warns when converting takes longer than a chunk", async () => {
    vi.spyOn(insightApi, "get").mockResolvedValue({ ...DATA, chunks: [{ t: 1, infer_ms: 900, budget_ms: 750 }] });
    render(<LiveInsight running voiceName="Woman" />);
    expect(await screen.findByText(/too slow, move Quality further left/)).toBeInTheDocument();
  });

  it("shows nothing and asks for nothing when live mode is off", async () => {
    const get = vi.spyOn(insightApi, "get").mockResolvedValue(DATA);
    const { container } = render(<LiveInsight running={false} voiceName="Woman" />);
    await waitFor(() => expect(container).toBeEmptyDOMElement());
    expect(get).not.toHaveBeenCalled();
  });
});
