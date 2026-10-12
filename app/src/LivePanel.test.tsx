import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { LivePanel } from "./LivePanel";
import { IDLE } from "./useStats";

vi.mock("./api", () => ({
  api: {
    devices: vi.fn(), voices: vi.fn(), presets: vi.fn(),
    start: vi.fn(), stop: vi.fn(), mode: vi.fn(), settings: vi.fn(),
  },
}));
import { api } from "./api";

const PRESETS = {
  low_latency: { chunk_ms: 120, crossfade_ms: 30, context_ms: 400, f0_method: "dio", index_rate: 0 },
  balanced: { chunk_ms: 250, crossfade_ms: 50, context_ms: 800, f0_method: "dio", index_rate: 0.3 },
  high_quality: { chunk_ms: 500, crossfade_ms: 80, context_ms: 1500, f0_method: "dio", index_rate: 0.6 },
} as const;

beforeEach(() => {
  vi.resetAllMocks();
  vi.mocked(api.devices).mockResolvedValue({
    devices: [
      { index: 1, name: "Headset Mic", max_input: 1, max_output: 0 },
      { index: 2, name: "Headset", max_input: 0, max_output: 2 },
      { index: 3, name: "VoiceMeeter Input", max_input: 0, max_output: 8 },
    ],
    virtual_mic: 3,
  });
  vi.mocked(api.voices).mockResolvedValue([{ id: "demo", name: "Demo", has_index: true }]);
  vi.mocked(api.presets).mockResolvedValue(PRESETS as never);
  vi.mocked(api.start).mockResolvedValue({ ...IDLE, running: true, mode: "converted" });
  vi.mocked(api.stop).mockResolvedValue(IDLE);
  vi.mocked(api.mode).mockResolvedValue({ ...IDLE, running: true, mode: "muted" });
  vi.mocked(api.settings).mockResolvedValue({ ...IDLE, running: true, mode: "converted" });
});

async function ready() {
  const button = await screen.findByRole("button", { name: "Start" });
  await vi.waitFor(() => expect(button).toBeEnabled());
  return button;
}

describe("LivePanel", () => {
  it("preselects the virtual mic and starts with the chosen settings", async () => {
    render(<LivePanel stats={IDLE} />);
    await userEvent.click(await ready());
    expect(api.start).toHaveBeenCalledWith({
      input_device: 1, mic_device: 3, voice: "demo", monitor_device: null, monitor_gain: 1,
      preset: "balanced", overrides: null, pitch_shift: 0,
    });
  });

  it("only offers inputs as the microphone and outputs as destinations", async () => {
    render(<LivePanel stats={IDLE} />);
    await ready();
    expect(screen.getByLabelText("Microphone").querySelectorAll("option")).toHaveLength(1);
    expect(screen.getByLabelText("Virtual mic").querySelectorAll("option")).toHaveLength(2);
  });

  it("sends the monitor device when monitoring is on", async () => {
    render(<LivePanel stats={IDLE} />);
    const start = await ready();
    await userEvent.click(screen.getByLabelText("Listen to output"));
    await userEvent.click(start);
    expect(vi.mocked(api.start).mock.calls[0][0].monitor_device).toBe(2);
  });

  it("sends advanced overrides only when changed", async () => {
    render(<LivePanel stats={IDLE} />);
    const start = await ready();
    await userEvent.click(screen.getByText("Advanced"));
    const chunk = screen.getByLabelText("Chunk (ms)");
    await userEvent.clear(chunk);
    await userEvent.type(chunk, "300");
    await userEvent.click(start);
    expect(vi.mocked(api.start).mock.calls[0][0].overrides).toEqual({ chunk_ms: 300 });
  });

  it("shows the backend's error message when start fails", async () => {
    vi.mocked(api.start).mockRejectedValue(new Error("Monitor and virtual mic must be different devices"));
    render(<LivePanel stats={IDLE} />);
    await userEvent.click(await ready());
    expect(await screen.findByRole("alert")).toHaveTextContent("must be different");
  });

  it("shows Stop and Mute while running, plus latency", async () => {
    render(<LivePanel stats={{ ...IDLE, running: true, mode: "converted", latency_ms: 312.4, infer_ms: 40 }} />);
    await userEvent.click(await screen.findByRole("button", { name: "Stop" }));
    expect(api.stop).toHaveBeenCalled();
    await userEvent.click(screen.getByRole("button", { name: "Mute" }));
    expect(api.mode).toHaveBeenCalledWith("muted");
    expect(screen.getByText(/312 ms/)).toBeInTheDocument();
  });

  it("recommends a faster preset when the GPU cannot keep up", async () => {
    render(<LivePanel stats={{ ...IDLE, running: true, mode: "converted", suggest_faster: true }} />);
    expect(await screen.findByText(/try a faster preset/i)).toBeInTheDocument();
  });

  it("surfaces a session error from stats", async () => {
    render(<LivePanel stats={{ ...IDLE, error: "An audio device was disconnected" }} />);
    expect(await screen.findByRole("alert")).toHaveTextContent("disconnected");
  });

  it("explains when no voices are installed", async () => {
    vi.mocked(api.voices).mockResolvedValue([]);
    render(<LivePanel stats={IDLE} />);
    expect(await screen.findByText(/no voice models/i)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Start" })).toBeDisabled();
  });

  it("shows Unmute while muted and unmutes", async () => {
    render(<LivePanel stats={{ ...IDLE, running: true, mode: "muted" }} />);
    await userEvent.click(await screen.findByRole("button", { name: "Unmute" }));
    expect(api.mode).toHaveBeenCalledWith("converted");
  });

  it("does not store an override of 0 when a number field is cleared", async () => {
    render(<LivePanel stats={IDLE} />);
    const start = await ready();
    await userEvent.click(screen.getByText("Advanced"));
    await userEvent.clear(screen.getByLabelText("Chunk (ms)"));
    await userEvent.click(start);
    expect(vi.mocked(api.start).mock.calls[0][0].overrides).toBeNull();
  });

  it("shows an error when a pitch change fails while running", async () => {
    vi.mocked(api.settings).mockRejectedValue(new Error("pitch failed"));
    render(<LivePanel stats={{ ...IDLE, running: true, mode: "converted" }} />);
    await screen.findByRole("button", { name: "Stop" });
    fireEvent.change(screen.getByLabelText(/Pitch shift/), { target: { value: "3" } });
    expect(await screen.findByRole("alert")).toHaveTextContent("pitch failed");
  });

  it("keeps the monitor checkbox unchanged and shows the error when disabling fails", async () => {
    const { rerender } = render(<LivePanel stats={IDLE} />);
    await ready();
    await userEvent.click(screen.getByLabelText("Listen to output"));
    rerender(<LivePanel stats={{ ...IDLE, running: true, mode: "converted" }} />);
    vi.mocked(api.settings).mockRejectedValue(new Error("monitor failed"));
    await userEvent.click(screen.getByLabelText("Listen to output"));
    expect(await screen.findByRole("alert")).toHaveTextContent("monitor failed");
    expect(screen.getByLabelText("Listen to output")).toBeChecked();
  });

  it("unchecks the monitor only after the backend accepts", async () => {
    const { rerender } = render(<LivePanel stats={IDLE} />);
    await ready();
    await userEvent.click(screen.getByLabelText("Listen to output"));
    rerender(<LivePanel stats={{ ...IDLE, running: true, mode: "converted" }} />);
    await userEvent.click(screen.getByLabelText("Listen to output"));
    expect(api.settings).toHaveBeenCalledWith({ monitor_enabled: false });
    await vi.waitFor(() => expect(screen.getByLabelText("Listen to output")).not.toBeChecked());
  });

  it("shows an error when a monitor volume change fails", async () => {
    const { rerender } = render(<LivePanel stats={IDLE} />);
    await ready();
    await userEvent.click(screen.getByLabelText("Listen to output"));
    rerender(<LivePanel stats={{ ...IDLE, running: true, mode: "converted" }} />);
    vi.mocked(api.settings).mockRejectedValue(new Error("gain failed"));
    fireEvent.change(screen.getByLabelText("Listen volume"), { target: { value: "1.5" } });
    expect(await screen.findByRole("alert")).toHaveTextContent("gain failed");
  });

  it("shows the external action error prop in the alert", async () => {
    render(<LivePanel stats={IDLE} actionError="Could not mute: boom" />);
    expect(await screen.findByRole("alert")).toHaveTextContent("Could not mute: boom");
  });

  it("shows a load failure with a Retry button that reloads", async () => {
    vi.mocked(api.devices).mockRejectedValueOnce(new Error("load failed"));
    render(<LivePanel stats={IDLE} />);
    expect(await screen.findByRole("alert")).toHaveTextContent("load failed");
    await userEvent.click(screen.getByRole("button", { name: "Retry" }));
    expect(await ready()).toBeEnabled();
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });
});
