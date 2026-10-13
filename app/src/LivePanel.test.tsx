import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { LivePanel } from "./LivePanel";
import { IDLE } from "./useStats";

vi.mock("./api", () => ({
  api: {
    devices: vi.fn(), refreshDevices: vi.fn(), voices: vi.fn(), presets: vi.fn(),
    start: vi.fn(), stop: vi.fn(), mode: vi.fn(), settings: vi.fn(),
  },
}));
import { api } from "./api";

const PRESETS = {
  low_latency: { chunk_ms: 120, crossfade_ms: 30, context_ms: 400, f0_method: "dio", index_rate: 0 },
  balanced: { chunk_ms: 250, crossfade_ms: 50, context_ms: 800, f0_method: "dio", index_rate: 0.3 },
  high_quality: { chunk_ms: 500, crossfade_ms: 80, context_ms: 1500, f0_method: "dio", index_rate: 0.6 },
} as const;

const MIC = { index: 1, name: "Headset Mic", max_input: 1, max_output: 0 };
const WEBCAM = { index: 5, name: "Webcam Mic", max_input: 1, max_output: 0 };
const HEADSET = { index: 2, name: "Headset", max_input: 0, max_output: 2 };
const SPEAKERS = { index: 6, name: "Speakers", max_input: 0, max_output: 2 };
const VM = { index: 3, name: "VoiceMeeter Input", max_input: 0, max_output: 8 };
const VM_AUX = { index: 4, name: "VoiceMeeter Aux Input", max_input: 0, max_output: 8 };
const DEVICES = { devices: [MIC, HEADSET, VM], virtual_mic: 3 as number | null, default_input: 1 as number | null, default_output: 2 as number | null };
const NO_VIRTUAL_MIC = { ...DEVICES, devices: [MIC, SPEAKERS, HEADSET], virtual_mic: null, default_output: 6 };

beforeEach(() => {
  vi.resetAllMocks();
  vi.mocked(api.devices).mockResolvedValue(DEVICES);
  vi.mocked(api.refreshDevices).mockResolvedValue(DEVICES);
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
    const destinations = [...screen.getByLabelText("Virtual mic").querySelectorAll("option")].map((o) => o.textContent);
    expect(destinations).toEqual(["None (listen only)", "Headset", "VoiceMeeter Input"]);
  });

  describe("without a virtual mic", () => {
    beforeEach(() => vi.mocked(api.devices).mockResolvedValue(NO_VIRTUAL_MIC));

    it("selects None instead of a real output, explains why, and keeps Start disabled", async () => {
      render(<LivePanel stats={IDLE} />);
      await screen.findByText(/other apps will not hear anything until VoiceMeeter is installed/i);
      expect(screen.getByLabelText("Virtual mic")).toHaveValue("");
      expect(screen.getByLabelText("Virtual mic")).toHaveDisplayValue("None (listen only)");
      expect(screen.getByRole("button", { name: "Start" })).toBeDisabled();
      expect(screen.getByText(/choose a virtual mic or tick listen to output/i)).toBeInTheDocument();
    });

    it("starts listen-only with mic_device null and the listen device", async () => {
      render(<LivePanel stats={IDLE} />);
      await screen.findByText(/other apps will not hear anything/i);
      await userEvent.click(screen.getByLabelText("Listen to output"));
      await userEvent.click(await ready());
      expect(api.start).toHaveBeenCalledWith(expect.objectContaining({ input_device: 1, mic_device: null, monitor_device: 6 }));
    });
  });

  it("shows no missing-virtual-mic hint when one exists", async () => {
    render(<LivePanel stats={IDLE} />);
    await ready();
    expect(screen.queryByText(/other apps will not hear anything/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/choose a virtual mic or tick/i)).not.toBeInTheDocument();
  });

  it("disables Start when neither a virtual mic nor a listen device is chosen", async () => {
    render(<LivePanel stats={IDLE} />);
    const start = await ready();
    await userEvent.selectOptions(screen.getByLabelText("Virtual mic"), "None (listen only)");
    expect(start).toBeDisabled();
    await userEvent.click(screen.getByLabelText("Listen to output"));
    expect(start).toBeEnabled();
    await userEvent.click(start);
    expect(api.start).toHaveBeenCalledWith(expect.objectContaining({ mic_device: null, monitor_device: 2 }));
  });

  describe("default devices", () => {
    const startListening = async () => {
      render(<LivePanel stats={IDLE} />);
      const start = await ready();
      await userEvent.click(screen.getByLabelText("Listen to output"));
      await userEvent.click(start);
      return vi.mocked(api.start).mock.calls[0][0];
    };
    const outputs = [VM_AUX, SPEAKERS, HEADSET, VM];

    it("preselects the Windows default microphone and playback device", async () => {
      vi.mocked(api.devices).mockResolvedValue({
        devices: [WEBCAM, MIC, ...outputs], virtual_mic: 3, default_input: 1, default_output: 2,
      });
      const sent = await startListening();
      expect(sent).toEqual(expect.objectContaining({ input_device: 1, mic_device: 3, monitor_device: 2 }));
    });

    it("falls back to the first microphone and the first non-virtual output", async () => {
      vi.mocked(api.devices).mockResolvedValue({
        devices: [WEBCAM, MIC, ...outputs], virtual_mic: 3, default_input: null, default_output: null,
      });
      const sent = await startListening();
      expect(sent).toEqual(expect.objectContaining({ input_device: 5, monitor_device: 6 }));
    });

    it("never listens on the virtual mic even when it is the Windows default output", async () => {
      vi.mocked(api.devices).mockResolvedValue({
        devices: [WEBCAM, MIC, ...outputs], virtual_mic: 3, default_input: 1, default_output: 3,
      });
      const sent = await startListening();
      expect(sent).toEqual(expect.objectContaining({ mic_device: 3, monitor_device: 6 }));
    });

    it("uses a virtual output to listen only when nothing else exists", async () => {
      vi.mocked(api.devices).mockResolvedValue({
        devices: [MIC, VM_AUX, VM], virtual_mic: 3, default_input: 1, default_output: null,
      });
      const sent = await startListening();
      expect(sent).toEqual(expect.objectContaining({ mic_device: 3, monitor_device: 4 }));
    });
  });

  describe("Refresh devices", () => {
    it("reloads the list, keeps selections that still exist, and reports the refresh", async () => {
      vi.mocked(api.devices).mockResolvedValue({ ...DEVICES, devices: [MIC, WEBCAM, HEADSET, SPEAKERS, VM] });
      const onRefreshed = vi.fn();
      render(<LivePanel stats={IDLE} onDevicesRefreshed={onRefreshed} />);
      const start = await ready();
      await userEvent.selectOptions(screen.getByLabelText("Microphone"), "Webcam Mic");
      await userEvent.click(screen.getByLabelText("Listen to output"));
      await userEvent.selectOptions(screen.getByLabelText("Listen on"), "Speakers");
      const usb = { index: 9, name: "USB Headset", max_input: 0, max_output: 2 };
      vi.mocked(api.refreshDevices).mockResolvedValue({ ...DEVICES, devices: [MIC, WEBCAM, HEADSET, SPEAKERS, VM, usb], default_output: 9 });
      await userEvent.click(screen.getByRole("button", { name: "Refresh devices" }));
      expect(await screen.findAllByRole("option", { name: "USB Headset" })).toHaveLength(2);
      expect(api.refreshDevices).toHaveBeenCalledTimes(1);
      expect(onRefreshed).toHaveBeenCalledTimes(1);
      await userEvent.click(start);
      expect(api.start).toHaveBeenCalledWith(expect.objectContaining({ input_device: 5, mic_device: 3, monitor_device: 6 }));
    });

    it("falls back to the defaults for selections that disappeared", async () => {
      vi.mocked(api.devices).mockResolvedValue({ ...DEVICES, devices: [MIC, WEBCAM, HEADSET, SPEAKERS, VM] });
      render(<LivePanel stats={IDLE} />);
      const start = await ready();
      await userEvent.selectOptions(screen.getByLabelText("Microphone"), "Webcam Mic");
      await userEvent.click(screen.getByLabelText("Listen to output"));
      await userEvent.selectOptions(screen.getByLabelText("Listen on"), "Speakers");
      vi.mocked(api.refreshDevices).mockResolvedValue({ ...DEVICES, devices: [MIC, HEADSET, VM] });
      await userEvent.click(screen.getByRole("button", { name: "Refresh devices" }));
      await vi.waitFor(() => expect(screen.queryByRole("option", { name: "Speakers" })).not.toBeInTheDocument());
      await userEvent.click(start);
      expect(api.start).toHaveBeenCalledWith(expect.objectContaining({ input_device: 1, mic_device: 3, monitor_device: 2 }));
    });

    it("selects a virtual mic that appears after it was missing", async () => {
      vi.mocked(api.devices).mockResolvedValue(NO_VIRTUAL_MIC);
      render(<LivePanel stats={IDLE} />);
      await screen.findByText(/other apps will not hear anything/i);
      vi.mocked(api.refreshDevices).mockResolvedValue({ ...NO_VIRTUAL_MIC, devices: [MIC, SPEAKERS, HEADSET, VM], virtual_mic: 3 });
      await userEvent.click(screen.getByRole("button", { name: "Refresh devices" }));
      await vi.waitFor(() => expect(screen.getByLabelText("Virtual mic")).toHaveValue("3"));
      expect(screen.queryByText(/other apps will not hear anything/i)).not.toBeInTheDocument();
    });

    it("keeps a deliberate None choice when a virtual mic was available all along", async () => {
      render(<LivePanel stats={IDLE} />);
      await ready();
      await userEvent.selectOptions(screen.getByLabelText("Virtual mic"), "None (listen only)");
      await userEvent.click(screen.getByRole("button", { name: "Refresh devices" }));
      await vi.waitFor(() => expect(api.refreshDevices).toHaveBeenCalled());
      expect(screen.getByLabelText("Virtual mic")).toHaveValue("");
    });

    it("shows the error when the refresh fails", async () => {
      vi.mocked(api.refreshDevices).mockRejectedValue(new Error("Stop live mode before refreshing devices"));
      render(<LivePanel stats={IDLE} />);
      await ready();
      await userEvent.click(screen.getByRole("button", { name: "Refresh devices" }));
      expect(await screen.findByRole("alert")).toHaveTextContent("Stop live mode before refreshing devices");
    });

    it("is disabled while live mode is running", async () => {
      render(<LivePanel stats={{ ...IDLE, running: true, mode: "converted" }} />);
      await screen.findByRole("button", { name: "Stop" });
      expect(screen.getByRole("button", { name: "Refresh devices" })).toBeDisabled();
    });
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

  describe("when a trained voice appears (voicesVersion changes)", () => {
    const DEMO = { id: "demo", name: "Demo", has_index: true };
    const ALPHA = { id: "alpha", name: "Alpha", has_index: false };

    it("adds the new voice to the list and keeps the selection and the rest of the form", async () => {
      vi.mocked(api.voices).mockResolvedValue([DEMO, ALPHA]);
      const { rerender } = render(<LivePanel stats={IDLE} voicesVersion={0} />);
      await ready();
      await userEvent.selectOptions(screen.getByLabelText("Voice"), "alpha");
      await userEvent.selectOptions(screen.getByLabelText("Speed / quality"), "high_quality");
      const NEW = { id: "mine", name: "Mine", has_index: false };
      vi.mocked(api.voices).mockResolvedValue([DEMO, ALPHA, NEW]);
      const devicesCalls = vi.mocked(api.devices).mock.calls.length;
      rerender(<LivePanel stats={IDLE} voicesVersion={1} />);
      expect(await screen.findByRole("option", { name: "Mine" })).toBeInTheDocument();
      expect(screen.getByLabelText("Voice")).toHaveValue("alpha");
      expect(screen.getByLabelText("Speed / quality")).toHaveValue("high_quality");
      expect(api.devices).toHaveBeenCalledTimes(devicesCalls); // only the voice list is reloaded
      await userEvent.click(screen.getByRole("button", { name: "Start" }));
      expect(api.start).toHaveBeenCalledWith(expect.objectContaining({ voice: "alpha", preset: "high_quality" }));
    });

    it("falls back to the first voice when the selected one is gone", async () => {
      vi.mocked(api.voices).mockResolvedValue([DEMO, ALPHA]);
      const { rerender } = render(<LivePanel stats={IDLE} voicesVersion={0} />);
      await ready();
      await userEvent.selectOptions(screen.getByLabelText("Voice"), "alpha");
      vi.mocked(api.voices).mockResolvedValue([DEMO]);
      rerender(<LivePanel stats={IDLE} voicesVersion={1} />);
      await vi.waitFor(() => expect(screen.queryByRole("option", { name: "Alpha" })).not.toBeInTheDocument());
      expect(screen.getByLabelText("Voice")).toHaveValue("demo");
    });

    it("does not reload on first render or when the version is unchanged", async () => {
      const { rerender } = render(<LivePanel stats={IDLE} voicesVersion={3} />);
      await ready();
      rerender(<LivePanel stats={IDLE} voicesVersion={3} />);
      expect(api.voices).toHaveBeenCalledTimes(1);
    });

    it("leaves a running session alone and ignores a failed reload", async () => {
      const running = { ...IDLE, running: true, mode: "converted" as const };
      const { rerender } = render(<LivePanel stats={IDLE} voicesVersion={0} />);
      await ready();
      rerender(<LivePanel stats={running} voicesVersion={0} />);
      vi.mocked(api.voices).mockRejectedValue(new Error("down"));
      rerender(<LivePanel stats={running} voicesVersion={1} />);
      await vi.waitFor(() => expect(api.voices).toHaveBeenCalledTimes(2));
      expect(screen.queryByRole("alert")).not.toBeInTheDocument();
      expect(screen.getByLabelText("Voice")).toHaveValue("demo");
      expect(api.stop).not.toHaveBeenCalled();
    });
  });

  it("disables Start and Refresh devices with a reason while speech is active", async () => {
    render(<LivePanel stats={IDLE} speechActive />);
    await vi.waitFor(() => expect(screen.getByRole("button", { name: "Refresh devices" })).toBeDisabled());
    expect(screen.getByRole("button", { name: "Start" })).toBeDisabled();
    expect(screen.getByText(/Speech is playing. Stop it on the Speak tab/)).toBeInTheDocument();
  });
});
