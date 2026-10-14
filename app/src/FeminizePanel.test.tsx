import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { FeminizePanel } from "./FeminizePanel";
import { FeminizeJob, feminizeApi } from "./feminizeApi";
import { LibraryVoice } from "./studioApi";

vi.mock("./feminizeApi", async (original) => ({
  ...(await original<typeof import("./feminizeApi")>()),
  feminizeApi: { sampleUrl: vi.fn(), start: vi.fn(), status: vi.fn(), cancel: vi.fn(), settings: vi.fn(), saveSettings: vi.fn() },
}));

const VOICE: LibraryVoice = { id: "kai-1", name: "Kai", created: "", minutes: 6, clips: 50, models: [] };
const IDLE: FeminizeJob = { active: false, finished: false, cancelled: false, done: 0, total: 0, source: null,
  voice_id: null, voice_name: null, error: null };
const play = vi.fn().mockResolvedValue(undefined);

beforeEach(() => {
  vi.resetAllMocks();
  vi.mocked(feminizeApi.settings).mockResolvedValue({ pitch_semitones: 6, formant_shift: 1.2, breathiness: 0.1 });
  vi.mocked(feminizeApi.status).mockResolvedValue(IDLE);
  play.mockResolvedValue(undefined);
  vi.stubGlobal("Audio", vi.fn(function (this: { play: typeof play; pause: () => void }) {
    this.play = play;
    this.pause = () => {};
  }));
  URL.revokeObjectURL = vi.fn();
});
afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

describe("FeminizePanel", () => {
  it("starts from the saved live preview settings and plays a sample made with the sliders' values", async () => {
    vi.mocked(feminizeApi.sampleUrl).mockResolvedValue("blob:sample");
    render(<FeminizePanel voice={VOICE} onCreated={() => {}} />);
    expect(await screen.findByText("Pitch: +6 semitones")).toBeInTheDocument();
    expect(screen.getByText("Resonance: +20%")).toBeInTheDocument();
    fireEvent.change(screen.getByLabelText(/Pitch:/), { target: { value: "7.5" } });
    fireEvent.click(screen.getByRole("button", { name: "Hear a sample" }));
    await waitFor(() => expect(play).toHaveBeenCalled());
    expect(feminizeApi.sampleUrl).toHaveBeenCalledWith("kai-1", { pitch_semitones: 7.5, formant_shift: 1.2, breathiness: 0.1 });
    expect(Audio).toHaveBeenCalledWith("blob:sample");
  });

  it("saves the settings for the live preview and says where to find it", async () => {
    vi.mocked(feminizeApi.saveSettings).mockImplementation(async (s) => s);
    render(<FeminizePanel voice={VOICE} onCreated={() => {}} />);
    await screen.findByText("Pitch: +6 semitones");
    fireEvent.click(screen.getByRole("button", { name: "Use for the live preview" }));
    expect(await screen.findByText(/choose the voice "My voice, feminized/)).toBeInTheDocument();
    expect(feminizeApi.saveSettings).toHaveBeenCalledWith({ pitch_semitones: 6, formant_shift: 1.2, breathiness: 0.1 });
  });

  it("makes a feminized voice, follows the progress and hands over the new voice when it is ready", async () => {
    vi.useFakeTimers();
    const onCreated = vi.fn();
    const running: FeminizeJob = { ...IDLE, active: true, total: 50, done: 0, source: "kai-1", voice_id: "kai-f", voice_name: "Kai (feminized)" };
    vi.mocked(feminizeApi.start).mockResolvedValue(running);
    render(<FeminizePanel voice={VOICE} onCreated={onCreated} />);
    await act(async () => {});
    fireEvent.click(screen.getByRole("button", { name: "Make a feminized voice" }));
    await act(async () => {});
    expect(screen.getByText('Making "Kai (feminized)": clip 0 of 50.')).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Make a feminized voice" })).toBeDisabled();
    vi.mocked(feminizeApi.status).mockResolvedValue({ ...running, done: 25 });
    await act(async () => { await vi.advanceTimersByTimeAsync(500); });
    expect(screen.getByText('Making "Kai (feminized)": clip 25 of 50.')).toBeInTheDocument();
    vi.mocked(feminizeApi.status).mockResolvedValue({ ...running, active: false, finished: true, done: 50, skipped: 1 });
    await act(async () => { await vi.advanceTimersByTimeAsync(500); });
    expect(screen.getByText(/"Kai \(feminized\)" is ready with 49 clips/)).toBeInTheDocument();
    expect(onCreated).toHaveBeenCalledWith("kai-f");
  });

  it("shows refusals and failures as messages", async () => {
    vi.mocked(feminizeApi.sampleUrl).mockRejectedValue(new Error("This voice has no clip of 3 to 12 seconds to make a sample from"));
    render(<FeminizePanel voice={VOICE} onCreated={() => {}} />);
    await screen.findByText("Pitch: +6 semitones");
    fireEvent.click(screen.getByRole("button", { name: "Hear a sample" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("no clip of 3 to 12 seconds");
  });

  it("cannot make anything from a voice without clips", async () => {
    render(<FeminizePanel voice={{ ...VOICE, clips: 0 }} onCreated={() => {}} />);
    await screen.findByText("Pitch: +6 semitones");
    expect(screen.getByRole("button", { name: "Hear a sample" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Make a feminized voice" })).toBeDisabled();
  });
});
