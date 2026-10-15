import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { LivePanel } from "./LivePanel";
import { IDLE } from "./useStats";

vi.mock("./calibrationApi", () => ({
  calibrationApi: {
    status: vi.fn(), start: vi.fn(), finish: vi.fn(), cancel: vi.fn(), forget: vi.fn(), suggestion: vi.fn(),
    speakerModel: vi.fn(), downloadSpeakerModel: vi.fn(), tune: vi.fn(), tuneStatus: vi.fn(), tuneCancel: vi.fn(),
    preview: vi.fn(),
  },
}));
const NO_PROFILE = { recording: false, seconds: 0, level: 0, phase: null, quiet_seconds: 3, total_seconds: 18, text: "Read me.", error: null, profile: null };
const MY_VOICE = { pitch_hz: 118, pitch_low_hz: 95, pitch_high_hz: 160, level_db: -22, noise_db: -62, gate_db: -56, input_gain: 1.3, device: "Headset Mic", created: 5 };
vi.mock("./LiveInsight", () => ({ LiveInsight: () => null }));
vi.mock("./api", () => ({
  api: {
    devices: vi.fn(), refreshDevices: vi.fn(), voices: vi.fn(), presets: vi.fn(),
    start: vi.fn(), stop: vi.fn(), mode: vi.fn(), settings: vi.fn(),
    learning: vi.fn(), setLearning: vi.fn(), saveLearning: vi.fn(),
  },
}));
import { api } from "./api";
import { calibrationApi } from "./calibrationApi";
import { rememberMode, SetupState } from "./setupApi";

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
  rememberMode(null);
  vi.mocked(calibrationApi.status).mockResolvedValue(NO_PROFILE as never);
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
    const destinations = [...screen.getByLabelText(/^Virtual microphone/).querySelectorAll("option")].map((o) => o.textContent);
    expect(destinations).toEqual(["None (listen only)", "Headset", "VoiceMeeter Output (sent through VoiceMeeter Input)"]);
  });

  describe("without a virtual mic", () => {
    beforeEach(() => vi.mocked(api.devices).mockResolvedValue(NO_VIRTUAL_MIC));

    it("selects None instead of a real output, explains why, and keeps Start disabled", async () => {
      render(<LivePanel stats={IDLE} />);
      await screen.findByText(/other apps will not hear anything until VoiceMeeter is installed/i);
      expect(screen.getByLabelText(/^Virtual microphone/)).toHaveValue("");
      expect(screen.getByLabelText(/^Virtual microphone/)).toHaveDisplayValue("None (listen only)");
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
    await userEvent.selectOptions(screen.getByLabelText(/^Virtual microphone/), "None (listen only)");
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
    it("reports the chosen listen device so the Speak tab can default to it", async () => {
      vi.mocked(api.devices).mockResolvedValue({ ...DEVICES, devices: [MIC, HEADSET, SPEAKERS, VM] });
      const onMonitorChosen = vi.fn();
      render(<LivePanel stats={IDLE} onMonitorChosen={onMonitorChosen} />);
      await ready();
      await userEvent.click(screen.getByLabelText("Listen to output"));
      await userEvent.selectOptions(screen.getByLabelText("Listen on"), "Speakers");
      expect(onMonitorChosen).toHaveBeenLastCalledWith(6);
    });

    it("reports the chosen virtual mic so the Speak tab can default to it", async () => {
      vi.mocked(api.devices).mockResolvedValue({ ...DEVICES, devices: [MIC, HEADSET, SPEAKERS, VM] });
      const onMicChosen = vi.fn();
      render(<LivePanel stats={IDLE} onMicChosen={onMicChosen} />);
      await ready();
      await waitFor(() => expect(onMicChosen).toHaveBeenLastCalledWith(VM.index));
      await userEvent.selectOptions(screen.getByLabelText(/^Virtual microphone/), "");
      expect(onMicChosen).toHaveBeenLastCalledWith(null);
    });

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
      await vi.waitFor(() => expect(screen.getByLabelText(/^Virtual microphone/)).toHaveValue("3"));
      expect(screen.queryByText(/other apps will not hear anything/i)).not.toBeInTheDocument();
    });

    it("keeps a deliberate None choice when a virtual mic was available all along", async () => {
      render(<LivePanel stats={IDLE} />);
      await ready();
      await userEvent.selectOptions(screen.getByLabelText(/^Virtual microphone/), "None (listen only)");
      await userEvent.click(screen.getByRole("button", { name: "Refresh devices" }));
      await vi.waitFor(() => expect(api.refreshDevices).toHaveBeenCalled());
      expect(screen.getByLabelText(/^Virtual microphone/)).toHaveValue("");
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
    expect(screen.getByText(/0.31 s/)).toBeInTheDocument();
  });

  it("recommends a lower quality level when the GPU cannot keep up", async () => {
    render(<LivePanel stats={{ ...IDLE, running: true, mode: "converted", suggest_faster: true }} />);
    expect(await screen.findByText(/Move Quality further left/i)).toBeInTheDocument();
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
      fireEvent.change(screen.getByLabelText(/^Quality:/), { target: { value: "4" } });
      const NEW = { id: "mine", name: "Mine", has_index: false };
      vi.mocked(api.voices).mockResolvedValue([DEMO, ALPHA, NEW]);
      const devicesCalls = vi.mocked(api.devices).mock.calls.length;
      rerender(<LivePanel stats={IDLE} voicesVersion={1} />);
      expect(await screen.findByRole("option", { name: "Mine" })).toBeInTheDocument();
      expect(screen.getByLabelText("Voice")).toHaveValue("alpha");
      expect(screen.getByLabelText(/^Quality:/)).toHaveValue("4");
      expect(api.devices).toHaveBeenCalledTimes(devicesCalls); // only the voice list is reloaded
      await userEvent.click(screen.getByRole("button", { name: "Start" }));
      expect(api.start).toHaveBeenCalledWith(expect.objectContaining({ voice: "alpha" }));
      expect(vi.mocked(api.start).mock.calls[0][0].overrides).toMatchObject({ chunk_ms: 1000, context_ms: 3000, crossfade_ms: 120 });
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

describe("dedicated microphone setup", () => {
  const CABLE = { index: 41, name: "CABLE Input (VB-Audio Virtual Cable)", max_input: 0, max_output: 2 };

  it("offers the VB-CABLE steps when no cable is installed, and finds the cable after it is", async () => {
    const writeText = vi.fn().mockResolvedValue(undefined);
    Object.defineProperty(navigator, "clipboard", { value: { writeText }, configurable: true });
    render(<LivePanel stats={IDLE} />);
    await ready();
    expect(screen.getByText("Set up a dedicated Voxweave microphone")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Copy link" }));
    expect(writeText).toHaveBeenCalledWith("https://vb-audio.com/Cable/");
    expect(await screen.findByRole("button", { name: "Copied" })).toBeInTheDocument();

    vi.mocked(api.refreshDevices).mockResolvedValue({ ...DEVICES, devices: [...DEVICES.devices, CABLE], virtual_mic: 41 });
    fireEvent.click(screen.getByRole("button", { name: "I have installed it" }));
    await waitFor(() => expect(screen.queryByText("Set up a dedicated Voxweave microphone")).not.toBeInTheDocument());
    expect(api.refreshDevices).toHaveBeenCalledTimes(1);
    expect(screen.getByText(/as the microphone\./)).toHaveTextContent("CABLE Output");
    expect(screen.getByLabelText(/^Virtual microphone/)).toHaveValue("41"); // moved from VoiceMeeter to the new cable
  });

  it("says so when the cable is not there yet", async () => {
    render(<LivePanel stats={IDLE} />);
    await ready();
    fireEvent.click(screen.getByRole("button", { name: "I have installed it" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("No virtual cable was found yet");
    expect(screen.getByLabelText(/^Virtual microphone/)).toHaveValue("3");
  });

  it("shows which microphone other apps should choose when a cable is already there", async () => {
    vi.mocked(api.devices).mockResolvedValue({ ...DEVICES, devices: [...DEVICES.devices, CABLE], virtual_mic: 41 });
    render(<LivePanel stats={IDLE} />);
    await ready();
    expect(screen.getByLabelText(/^Virtual microphone/)).toHaveValue("41");
    expect(screen.getByText(/as the microphone\./)).toHaveTextContent("CABLE Output");
    expect(screen.queryByText("Set up a dedicated Voxweave microphone")).not.toBeInTheDocument();
  });
});

describe("my voice calibration", () => {
  it("without a calibration nothing about it is sent and the pitch stays where it is", async () => {
    render(<LivePanel stats={IDLE} />);
    await userEvent.click(await ready());
    expect(vi.mocked(api.start).mock.calls[0][0]).not.toHaveProperty("use_calibration");
    expect(calibrationApi.suggestion).not.toHaveBeenCalled();
    expect(screen.queryByLabelText(/Use my calibration/)).not.toBeInTheDocument();
  });

  it("with a calibration the pitch is set for the chosen voice and the calibration is used at start", async () => {
    vi.mocked(calibrationApi.status).mockResolvedValue({ ...NO_PROFILE, profile: MY_VOICE } as never);
    vi.mocked(calibrationApi.suggestion).mockResolvedValue({ pitch_shift: 7, reason: null, speaker_hz: 118, target_hz: 177 });
    render(<LivePanel stats={IDLE} />);
    const start = await ready();
    expect(await screen.findByText(/Set from your calibration: you speak at about 118 Hz, this voice at about 177 Hz/)).toBeInTheDocument();
    expect(calibrationApi.suggestion).toHaveBeenCalledWith("demo");
    expect(screen.getByText("Pitch shift (+7 semitones)")).toBeInTheDocument();
    await userEvent.click(start);
    expect(api.start).toHaveBeenCalledWith(expect.objectContaining({ pitch_shift: 7, use_calibration: true }));
  });

  it("a pitch moved by hand is kept, and the calibration can be switched off for a start", async () => {
    vi.mocked(calibrationApi.status).mockResolvedValue({ ...NO_PROFILE, profile: MY_VOICE } as never);
    vi.mocked(calibrationApi.suggestion).mockResolvedValue({ pitch_shift: 7, reason: null, speaker_hz: 118, target_hz: 177 });
    render(<LivePanel stats={IDLE} />);
    const start = await ready();
    await screen.findByText(/Set from your calibration/);
    fireEvent.change(screen.getByLabelText(/Pitch shift/), { target: { value: "3" } });
    expect(screen.queryByText(/Set from your calibration/)).not.toBeInTheDocument();
    await userEvent.click(screen.getByLabelText(/Use my calibration/));
    await userEvent.click(start);
    expect(api.start).toHaveBeenCalledWith(expect.objectContaining({ pitch_shift: 3, use_calibration: false }));
  });

  it("says why when a voice has nothing to measure, and leaves the pitch alone", async () => {
    vi.mocked(calibrationApi.status).mockResolvedValue({ ...NO_PROFILE, profile: MY_VOICE } as never);
    vi.mocked(calibrationApi.suggestion).mockResolvedValue({ pitch_shift: null, reason: "This voice has no recordings in the library to measure, so set the pitch by hand." });
    render(<LivePanel stats={IDLE} />);
    await ready();
    expect(await screen.findByText(/no recordings in the library to measure/)).toBeInTheDocument();
    expect(screen.getByText("Pitch shift (0 semitones)")).toBeInTheDocument();
  });
});

describe("naming the virtual microphone", () => {
  it("says which microphone other apps should choose, and what VoiceMeeter needs", async () => {
    render(<LivePanel stats={IDLE} />);
    await ready();
    const hint = screen.getByText(/In other apps \(Discord/);
    expect(hint).toHaveTextContent("choose VoiceMeeter Output as the microphone");
    expect(hint).toHaveTextContent("B1");
  });

  it("warns when the chosen device is an ordinary playback device", async () => {
    render(<LivePanel stats={IDLE} />);
    await ready();
    await userEvent.selectOptions(screen.getByLabelText(/^Virtual microphone/), "Headset");
    expect(screen.getByText(/not a virtual microphone/)).toBeInTheDocument();
  });
});

describe("LivePanel without any voice", () => {
  beforeEach(() => vi.mocked(api.voices).mockResolvedValue([]));

  it("points the source checkout to the README", async () => {
    render(<LivePanel stats={IDLE} />);
    expect(await screen.findByText("No voice models installed. See the README to add one.")).toBeInTheDocument();
  });

  it("points the installed app to voice packs, not to a README it does not have", async () => {
    rememberMode({ mode: "installed", status: "ready" } as SetupState);
    render(<LivePanel stats={IDLE} />);
    expect(await screen.findByText(/No voice models installed\. Bring in a trained voice as a voice pack/)).toBeInTheDocument();
    expect(screen.queryByText(/README/)).not.toBeInTheDocument();
  });
});

describe("likeness", () => {
  beforeEach(() => localStorage.clear());

  it("starts at the preset's value, is sent when changed, and is remembered for that voice", async () => {
    const { unmount } = render(<LivePanel stats={IDLE} />);
    const start = await ready();
    const slider = screen.getByLabelText(/^Likeness:/);
    expect(slider.closest("label")).toHaveTextContent("Likeness: 30%");
    fireEvent.change(slider, { target: { value: "0.55" } });
    expect(slider.closest("label")).toHaveTextContent("Likeness: 55%");
    expect(localStorage.getItem("likeness:demo")).toBe("0.55");
    await userEvent.click(start);
    expect(vi.mocked(api.start).mock.calls[0][0].overrides).toMatchObject({ index_rate: 0.55 });
    unmount();
    render(<LivePanel stats={IDLE} />);
    await ready();
    await waitFor(() => expect(screen.getByLabelText(/^Likeness:/).closest("label")).toHaveTextContent("Likeness: 55%"));
  });
});

describe("quality", () => {
  beforeEach(() => localStorage.clear());

  it("starts at level 2 (the old Balanced, nothing extra sent), goes up to two-second chunks and is remembered", async () => {
    const { unmount } = render(<LivePanel stats={IDLE} />);
    const start = await ready();
    const slider = screen.getByLabelText(/^Quality:/);
    expect(slider.closest("label")).toHaveTextContent("Quality: 2 of 7, about 0.6 s behind your voice");
    await userEvent.click(start);
    expect(vi.mocked(api.start).mock.calls[0][0].overrides).toBeNull();
    fireEvent.change(slider, { target: { value: "6" } });
    expect(slider.closest("label")).toHaveTextContent("Quality: 7 of 7, about 4.2 s behind your voice");
    await userEvent.click(start);
    expect(vi.mocked(api.start).mock.calls[1][0].overrides).toMatchObject({ chunk_ms: 2000, context_ms: 5000, crossfade_ms: 200 });
    unmount();
    render(<LivePanel stats={IDLE} />);
    await ready();
    await waitFor(() => expect(screen.getByLabelText(/^Quality:/)).toHaveValue("6"));
  });
});

describe("the shelved feminized preview voice", () => {
  it("is not offered in the voice list even though the backend still lists it", async () => {
    vi.mocked(api.voices).mockResolvedValue([
      { id: "demo", name: "Demo", has_index: true },
      { id: "@feminize", name: "My voice, feminized (preview, no training)", has_index: false },
    ]);
    render(<LivePanel stats={IDLE} />);
    await ready();
    expect(screen.queryByRole("option", { name: /feminized/ })).not.toBeInTheDocument();
    expect(screen.getByRole("option", { name: "Demo" })).toBeInTheDocument();
  });
});

describe("saved settings per voice", () => {
  const TWO_VOICES = [{ id: "demo", name: "Demo", has_index: true }, { id: "other", name: "Other", has_index: true }];
  beforeEach(() => {
    localStorage.clear();
    vi.mocked(api.voices).mockResolvedValue(TWO_VOICES);
    vi.mocked(calibrationApi.status).mockResolvedValue({ ...NO_PROFILE, profile: MY_VOICE } as never);
    vi.mocked(calibrationApi.suggestion).mockResolvedValue({ pitch_shift: 7, reason: null, speaker_hz: 118, target_hz: 177 });
  });

  async function setUp() {
    render(<LivePanel stats={IDLE} />);
    await ready();
    await screen.findByText(/Set from your calibration/);
    fireEvent.change(screen.getByLabelText(/Pitch shift/), { target: { value: "12" } });
    fireEvent.change(screen.getByLabelText(/^Quality:/), { target: { value: "4" } });
    fireEvent.change(screen.getByLabelText(/^Likeness:/), { target: { value: "0.7" } });
  }

  it("brings back the saved pitch, quality and likeness when switching back to the voice", async () => {
    await setUp();
    await userEvent.click(screen.getByRole("button", { name: "Save for this voice" }));
    expect(screen.getByText("Using your saved settings for Demo.")).toBeInTheDocument();
    await userEvent.selectOptions(screen.getByLabelText("Voice"), "other");
    await waitFor(() => expect(screen.getByText("Pitch shift (+7 semitones)")).toBeInTheDocument());
    fireEvent.change(screen.getByLabelText(/^Quality:/), { target: { value: "1" } });
    await userEvent.selectOptions(screen.getByLabelText("Voice"), "demo");
    await waitFor(() => expect(screen.getByText("Pitch shift (+12 semitones)")).toBeInTheDocument());
    expect(screen.getByLabelText(/^Quality:/)).toHaveValue("4");
    expect(screen.getByLabelText(/^Likeness:/).closest("label")).toHaveTextContent("Likeness: 70%");
    expect(screen.getByText("Using your saved settings for Demo.")).toBeInTheDocument();
  });

  it("is kept after the app restarts, ahead of the calibration's pitch", async () => {
    await setUp();
    await userEvent.click(screen.getByRole("button", { name: "Save for this voice" }));
    cleanup();
    render(<LivePanel stats={IDLE} />);
    await ready();
    await waitFor(() => expect(screen.getByText("Pitch shift (+12 semitones)")).toBeInTheDocument());
    await new Promise((r) => setTimeout(r, 20)); // the calibration's suggestion has had time to arrive
    expect(screen.getByText("Pitch shift (+12 semitones)")).toBeInTheDocument();
    expect(screen.getByLabelText(/^Quality:/)).toHaveValue("4");
  });

  it("says when the settings changed since they were saved", async () => {
    await setUp();
    await userEvent.click(screen.getByRole("button", { name: "Save for this voice" }));
    fireEvent.change(screen.getByLabelText(/Pitch shift/), { target: { value: "10" } });
    expect(screen.getByText(/Changed since you saved/)).toBeInTheDocument();
  });

  it("Reset goes back to the calibrated pitch, quality 2 and the default likeness, and forgets the saved settings", async () => {
    await setUp();
    await userEvent.click(screen.getByRole("button", { name: "Save for this voice" }));
    await userEvent.click(screen.getByRole("button", { name: "Reset to recommended" }));
    await waitFor(() => expect(screen.getByText("Pitch shift (+7 semitones)")).toBeInTheDocument());
    expect(screen.getByLabelText(/^Quality:/)).toHaveValue("1");
    expect(screen.getByLabelText(/^Likeness:/).closest("label")).toHaveTextContent("Likeness: 30%");
    expect(screen.queryByText(/Using your saved settings/)).not.toBeInTheDocument();
    expect(localStorage.getItem("voice-settings:demo")).toBeNull();
  });

  it("cannot reset while live mode runs, since quality and likeness are set before Start", async () => {
    render(<LivePanel stats={{ ...IDLE, running: true, mode: "converted" }} />);
    expect(await screen.findByRole("button", { name: "Reset to recommended" })).toBeDisabled();
  });
});

describe("deep calibration", () => {
  const TUNED = {
    voice_id: "demo", pitch_shift: 9, likeness: 0.5, similarity: 0.86, clarity: 0.95, quality: 3,
    baseline: { pitch_shift: 7, likeness: 0.3, similarity: 0.84, clarity: 0.96, quality: 1 }, tried: [], created: 1,
  };
  const MODEL = { ready: true, size_mb: 26.5, source: "https://huggingface.co/Wespeaker/x", licence: "CC BY 4.0" };
  beforeEach(() => {
    localStorage.clear();
    vi.mocked(calibrationApi.status).mockResolvedValue({ ...NO_PROFILE, profile: MY_VOICE, has_sample: true } as never);
    vi.mocked(calibrationApi.suggestion).mockResolvedValue({ pitch_shift: 7, reason: null, speaker_hz: 118, target_hz: 177 });
    vi.mocked(calibrationApi.speakerModel).mockResolvedValue(MODEL);
    vi.mocked(calibrationApi.tune).mockResolvedValue({ active: true, voice_id: "demo", step: 0, steps: 10, message: "Getting ready", error: null, result: null });
    vi.mocked(calibrationApi.tuneStatus)
      .mockResolvedValueOnce({ active: true, voice_id: "demo", step: 3, steps: 10, message: "Trying pitch +8, likeness 30%", error: null, result: null })
      .mockResolvedValue({ active: false, voice_id: "demo", step: 10, steps: 10, message: "Done", error: null, result: TUNED });
  });

  it("tries settings for the voice, then uses and saves the best", async () => {
    render(<LivePanel stats={IDLE} />);
    await ready();
    await screen.findByText(/Set from your calibration/);
    await userEvent.click(screen.getByRole("button", { name: "Find best settings" }));
    expect(calibrationApi.tune).toHaveBeenCalledWith(expect.objectContaining({ voice_id: "demo", quality: 1, pitch_shift: 7, likeness: 0.3 }));
    expect(vi.mocked(calibrationApi.tune).mock.calls[0][0].advanced).toMatchObject({ creak_repair: true, f0_method: "dio" });
    expect(await screen.findByText(/Trying pitch \+8, likeness 30% \(3 of 10\)/)).toBeInTheDocument();
    expect(await screen.findByText(/Best for Demo: pitch \+9, likeness 50%, quality 4 \(about 1\.6 s behind\)/, {}, { timeout: 4000 })).toBeInTheDocument();
    expect(screen.getByText(/sounds 86% like Demo \(84% with your settings before\)/)).toBeInTheDocument();
    expect(screen.getByText("Pitch shift (+9 semitones)")).toBeInTheDocument();
    expect(screen.getByLabelText(/^Likeness:/).closest("label")).toHaveTextContent("Likeness: 50%");
    expect(screen.getByLabelText(/^Quality:/)).toHaveValue("3");
    expect(JSON.parse(localStorage.getItem("voice-settings:demo")!)).toMatchObject({ pitch: 9, quality: 3, likeness: 0.5 });
  });

  it("asks before downloading the speaker model, and does nothing when declined", async () => {
    vi.mocked(calibrationApi.speakerModel).mockResolvedValue({ ...MODEL, ready: false });
    const confirm = vi.spyOn(window, "confirm").mockReturnValue(false);
    render(<LivePanel stats={IDLE} />);
    await ready();
    await userEvent.click(screen.getByRole("button", { name: "Find best settings" }));
    expect(confirm.mock.calls[0][0]).toMatch(/26.5 MB.*Hugging Face.*CC BY 4.0/s);
    expect(calibrationApi.downloadSpeakerModel).not.toHaveBeenCalled();
    expect(calibrationApi.tune).not.toHaveBeenCalled();
    confirm.mockRestore();
  });

  it("downloads the speaker model once agreed, then runs", async () => {
    vi.mocked(calibrationApi.speakerModel).mockResolvedValue({ ...MODEL, ready: false });
    vi.mocked(calibrationApi.downloadSpeakerModel).mockResolvedValue(MODEL);
    const confirm = vi.spyOn(window, "confirm").mockReturnValue(true);
    render(<LivePanel stats={IDLE} />);
    await ready();
    await userEvent.click(screen.getByRole("button", { name: "Find best settings" }));
    await waitFor(() => expect(calibrationApi.tune).toHaveBeenCalled());
    expect(calibrationApi.downloadSpeakerModel).toHaveBeenCalled();
    confirm.mockRestore();
  });

  it("asks for a new calibration when the old one kept no reading", async () => {
    vi.mocked(calibrationApi.status).mockResolvedValue({ ...NO_PROFILE, profile: MY_VOICE, has_sample: false } as never);
    render(<LivePanel stats={IDLE} />);
    await ready();
    await userEvent.click(screen.getByRole("button", { name: "Find best settings" }));
    expect(await screen.findByText(/Calibrate your voice again first/)).toBeInTheDocument();
    expect(calibrationApi.tune).not.toHaveBeenCalled();
  });

  it("is not offered without a calibration", async () => {
    vi.mocked(calibrationApi.status).mockResolvedValue(NO_PROFILE as never);
    render(<LivePanel stats={IDLE} />);
    await ready();
    expect(screen.queryByRole("button", { name: "Find best settings" })).not.toBeInTheDocument();
  });
});


describe("clipping", () => {
  it("says when the microphone clips", async () => {
    render(<LivePanel stats={{ ...IDLE, running: true, mode: "converted", clipping: true }} />);
    expect(await screen.findByText(/Your microphone is clipping/)).toBeInTheDocument();
  });
});

describe("voice quality extras", () => {
  beforeEach(() => localStorage.clear());

  it("offers every extra under Advanced and sends the ones changed", async () => {
    render(<LivePanel stats={IDLE} />);
    const start = await ready();
    await userEvent.click(screen.getByText("Advanced"));
    expect(screen.getByRole("option", { name: /RMVPE/ })).toBeInTheDocument();
    for (const name of [/Pitch clean-up/, /Creak repair/, /Join smoothing/]) {
      expect(screen.getByLabelText(name)).toBeChecked();
    }
    await userEvent.click(screen.getByLabelText(/Creak repair/));
    fireEvent.change(screen.getByLabelText(/^Natural waver/), { target: { value: "0" } });
    fireEvent.change(screen.getByLabelText(/^Background cut/), { target: { value: "45" } });
    fireEvent.change(screen.getByLabelText(/^Loudness follow/), { target: { value: "1" } });
    fireEvent.change(screen.getByLabelText(/^Consonant protection/), { target: { value: "0.5" } });
    await userEvent.click(start);
    expect(vi.mocked(api.start).mock.calls[0][0].overrides).toMatchObject({
      creak_repair: false, natural_variation_st: 0, background_cut_db: 45, volume_follow: 1, protect: 0.5,
    });
  });

  it("are saved for the voice with the rest of its settings", async () => {
    vi.mocked(api.voices).mockResolvedValue([{ id: "demo", name: "Demo", has_index: true }, { id: "other", name: "Other", has_index: true }]);
    render(<LivePanel stats={IDLE} />);
    await ready();
    await userEvent.click(screen.getByText("Advanced"));
    await userEvent.click(screen.getByLabelText(/Join smoothing/));
    await userEvent.click(screen.getByRole("button", { name: "Save for this voice" }));
    expect(JSON.parse(localStorage.getItem("voice-settings:demo")!).advanced).toMatchObject({ join_smoothing: false });
    await userEvent.selectOptions(screen.getByLabelText("Voice"), "other");
    await waitFor(() => expect(screen.getByLabelText(/Join smoothing/)).toBeChecked());
    await userEvent.selectOptions(screen.getByLabelText("Voice"), "demo");
    await waitFor(() => expect(screen.getByLabelText(/Join smoothing/)).not.toBeChecked());
    await userEvent.click(screen.getByRole("button", { name: "Reset to recommended" }));
    await waitFor(() => expect(screen.getByLabelText(/Join smoothing/)).toBeChecked());
  });
});


describe("live learning", () => {
  const RUNNING = { ...IDLE, running: true, mode: "converted" as const };
  const ON = { enabled: true, speech_seconds: 52, passes: 1, message: "1 tuning pass done", last_change: "likeness 30% to 40%",
               saved: false, learned: null, current: { pitch_shift: 8, likeness: 0.4, advanced: { creak_repair: true } } };
  beforeEach(() => {
    localStorage.clear();
    vi.mocked(calibrationApi.status).mockResolvedValue({ ...NO_PROFILE, profile: MY_VOICE } as never);
    vi.mocked(calibrationApi.suggestion).mockResolvedValue({ pitch_shift: 7, reason: null, speaker_hz: 118, target_hz: 177 });
    vi.mocked(api.setLearning).mockResolvedValue(ON);
    vi.mocked(api.learning).mockResolvedValue(ON);
  });

  it("can be switched on while live mode runs and says what it has done", async () => {
    render(<LivePanel stats={RUNNING} />);
    await userEvent.click(await screen.findByLabelText("Keep learning while I talk"));
    expect(api.setLearning).toHaveBeenCalledWith(true);
    expect(await screen.findByText(/Heard 52 s of your speech\. 1 tuning pass done\. Last change: likeness 30% to 40%\./)).toBeInTheDocument();
  });

  it("saves what it learned for the voice", async () => {
    vi.mocked(api.saveLearning).mockResolvedValue({ settings: { pitch_shift: 8, likeness: 0.4, advanced: { creak_repair: false } } });
    render(<LivePanel stats={RUNNING} />);
    await userEvent.click(await screen.findByLabelText("Keep learning while I talk"));
    await userEvent.click(await screen.findByRole("button", { name: "Save learned calibration" }));
    expect(await screen.findByText(/Saved\. Your calibration now follows how you talk live/)).toBeInTheDocument();
    expect(JSON.parse(localStorage.getItem("voice-settings:demo")!)).toMatchObject({ pitch: 8, likeness: 0.4, advanced: { creak_repair: false } });
  });

  it("turning it off says unsaved changes were put back", async () => {
    vi.mocked(api.setLearning).mockResolvedValueOnce(ON).mockResolvedValueOnce({ enabled: false });
    render(<LivePanel stats={RUNNING} />);
    const toggle = await screen.findByLabelText("Keep learning while I talk");
    await userEvent.click(toggle);
    await userEvent.click(toggle);
    expect(api.setLearning).toHaveBeenLastCalledWith(false);
    expect(await screen.findByText(/did not save was put back/)).toBeInTheDocument();
  });

  it("is not offered without a calibration or while live mode is off", async () => {
    vi.mocked(calibrationApi.status).mockResolvedValue(NO_PROFILE as never);
    render(<LivePanel stats={RUNNING} />);
    await screen.findByRole("button", { name: "Stop" });
    expect(screen.queryByLabelText("Keep learning while I talk")).not.toBeInTheDocument();
  });
});


describe("previewing a voice", () => {
  beforeEach(() => {
    localStorage.clear();
    vi.mocked(calibrationApi.status).mockResolvedValue({ ...NO_PROFILE, profile: MY_VOICE } as never);
    vi.mocked(calibrationApi.suggestion).mockResolvedValue({ pitch_shift: 7, reason: null, speaker_hz: 118, target_hz: 177 });
  });

  it("plays the calibration reading in the voice with the current settings", async () => {
    const played: string[] = [];
    class FakeAudio {
      onended: (() => void) | null = null;
      onerror: (() => void) | null = null;
      constructor(public src: string) {}
      play() { played.push(this.src); return Promise.resolve(); }
      pause() {}
    }
    vi.stubGlobal("Audio", FakeAudio);
    vi.mocked(calibrationApi.preview).mockResolvedValue("blob:preview");
    render(<LivePanel stats={IDLE} />);
    await ready();
    await screen.findByText(/Set from your calibration/);
    await userEvent.click(screen.getByRole("button", { name: "Play my calibration as Demo" }));
    expect(calibrationApi.preview).toHaveBeenCalledWith(expect.objectContaining({ voice_id: "demo", pitch_shift: 7, quality: 1, likeness: 0.3 }));
    await waitFor(() => expect(played).toEqual(["blob:preview"]));
    expect(screen.getByRole("button", { name: "Stop the preview" })).toBeInTheDocument();
    vi.unstubAllGlobals();
  });

  it("is not offered without a calibration", async () => {
    vi.mocked(calibrationApi.status).mockResolvedValue(NO_PROFILE as never);
    render(<LivePanel stats={IDLE} />);
    await ready();
    expect(screen.queryByRole("button", { name: /Play my calibration/ })).not.toBeInTheDocument();
  });
});

describe("effects", () => {
  beforeEach(() => localStorage.clear());

  it("has a tab of effects that are sent at start and remembered", async () => {
    const { unmount } = render(<LivePanel stats={IDLE} />);
    const start = await ready();
    await userEvent.click(screen.getByRole("tab", { name: "Effects" }));
    fireEvent.change(screen.getByLabelText(/^Echo:/), { target: { value: "0.4" } });
    fireEvent.change(screen.getByLabelText(/^Reverb:/), { target: { value: "0.3" } });
    await userEvent.click(start);
    expect(vi.mocked(api.start).mock.calls[0][0].overrides).toMatchObject({ fx_echo: 0.4, fx_reverb: 0.3 });
    unmount();
    render(<LivePanel stats={IDLE} />);
    await ready();
    expect(screen.getByLabelText(/^Echo:/)).toHaveValue("0.4");
  });

  it("change while live and have presets and a way to turn them all off", async () => {
    render(<LivePanel stats={{ ...IDLE, running: true, mode: "converted" }} />);
    await screen.findByRole("button", { name: "Stop" });
    await userEvent.click(screen.getByRole("tab", { name: "Effects" }));
    fireEvent.change(screen.getByLabelText(/^Static:/), { target: { value: "0.5" } });
    expect(api.settings).toHaveBeenLastCalledWith({ overrides: { fx_static: 0.5 } });
    await userEvent.click(screen.getByRole("button", { name: "Radio" }));
    expect(vi.mocked(api.settings).mock.calls[vi.mocked(api.settings).mock.calls.length - 1][0].overrides).toMatchObject({ fx_lofi: expect.any(Number), fx_static: expect.any(Number) });
    expect(Number((screen.getByLabelText(/^Lo-fi:/) as HTMLInputElement).value)).toBeGreaterThan(0);
    await userEvent.click(screen.getByRole("button", { name: "Turn all off" }));
    expect(vi.mocked(api.settings).mock.calls[vi.mocked(api.settings).mock.calls.length - 1][0].overrides).toMatchObject({ fx_lofi: 0, fx_static: 0, fx_reverb: 0, fx_echo: 0 });
  });
});
