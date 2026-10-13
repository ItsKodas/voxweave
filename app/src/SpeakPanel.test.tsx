import { act, fireEvent, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { SpeakPanel } from "./SpeakPanel";
import { TtsEnvironment, TtsStatus, TtsVoice } from "./ttsApi";
import { TtsView } from "./useTtsStatus";

vi.mock("./api", () => ({ api: { devices: vi.fn() } }));
vi.mock("./ttsApi", async (importOriginal) => ({
  ...(await importOriginal<typeof import("./ttsApi")>()),
  ttsApi: { environment: vi.fn(), voices: vi.fn(), speak: vi.fn(), stop: vi.fn() },
}));
import { api } from "./api";
import { ttsApi } from "./ttsApi";

const idle = (over: Partial<TtsStatus> = {}): TtsStatus => ({
  engine: "ready", state: "idle", voice_id: null, preset: null, piece: 0, pieces: 0, error: null,
  saved_path: null, started: null, ...over,
});
const view = (status: TtsStatus | null = idle(), over: Partial<TtsView> = {}): TtsView => ({
  status, unreachable: false, refresh: vi.fn(), apply: vi.fn(), ...over,
});

const HEADSET = { index: 2, name: "Headset", max_input: 0, max_output: 2 };
const SPEAKERS = { index: 6, name: "Speakers", max_input: 0, max_output: 2 };
const VM = { index: 3, name: "VoiceMeeter Input", max_input: 0, max_output: 8 };
const CABLE = { index: 4, name: "CABLE Input", max_input: 0, max_output: 2 };
const DEVICES = { devices: [HEADSET, SPEAKERS, VM, CABLE], virtual_mic: 3 as number | null, default_input: 1, default_output: 2 as number | null };

const DEMO: TtsVoice = { id: "demo", name: "Demo", has_model: true, reference: { ok: true, seconds: 6, message: "Clip 1 (6.0 seconds) will be used as the reference." } };
const PLAIN: TtsVoice = { id: "plain", name: "Plain", has_model: false, reference: { ok: true, seconds: 6, message: "ok" } };
const BAD: TtsVoice = { id: "bad", name: "Bad", has_model: false, reference: { ok: false, seconds: null, message: "No clip between 3 and 12 seconds." } };
const READY: TtsEnvironment = { ok: true, ready: true, checks: [] };

async function open(v: TtsView = view(), props: { liveRunning?: boolean } = {}) {
  const utils = render(<SpeakPanel tts={v} active liveRunning={props.liveRunning ?? false} />);
  await screen.findByRole("option", { name: /Demo/ });
  return utils;
}
const text = () => screen.getByRole("textbox", { name: "Text to speak" });
const speakButton = () => screen.getByRole("button", { name: "Speak" });
const type = (s: string) => fireEvent.change(text(), { target: { value: s } });

beforeEach(() => {
  vi.resetAllMocks();
  vi.mocked(api.devices).mockResolvedValue(DEVICES);
  vi.mocked(ttsApi.voices).mockResolvedValue([DEMO, PLAIN, BAD]);
  vi.mocked(ttsApi.environment).mockResolvedValue(READY);
  vi.mocked(ttsApi.speak).mockResolvedValue(idle({ state: "preparing" }));
  vi.mocked(ttsApi.stop).mockResolvedValue(idle());
});

describe("SpeakPanel form", () => {
  it("lists voices, disables one without a usable reference and shows why", async () => {
    await open();
    const bad = screen.getByRole("option", { name: /Bad/ });
    expect(bad).toBeDisabled();
    expect(bad).toHaveTextContent("No clip between 3 and 12 seconds.");
    expect(screen.getByRole("option", { name: /Demo/ })).toBeEnabled();
    expect(screen.getByRole("combobox", { name: "Voice" })).toHaveValue("demo");
  });

  it("shows the backend's reason beside a voice whose reference clip is too quiet", async () => {
    const quiet: TtsVoice = { ...BAD, id: "quiet", name: "Quiet", reference: { ok: false, seconds: 6, message: "The reference clip is too quiet." } };
    vi.mocked(ttsApi.voices).mockResolvedValue([DEMO, quiet]);
    await open();
    const option = screen.getByRole("option", { name: /Quiet/ });
    expect(option).toBeDisabled();
    expect(option).toHaveTextContent("Quiet (The reference clip is too quiet.)");
  });

  it("explains that a voice needs one recorded sentence when there are none", async () => {
    vi.mocked(ttsApi.voices).mockResolvedValue([]);
    render(<SpeakPanel tts={view()} active liveRunning={false} />);
    expect(await screen.findByText(/one recorded sentence/i)).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Speak" })).not.toBeInTheDocument();
  });

  it("says plainly when no voice can be spoken yet, next to the disabled Speak button", async () => {
    vi.mocked(ttsApi.voices).mockResolvedValue([BAD]);
    render(<SpeakPanel tts={view()} active liveRunning={false} />);
    expect(await screen.findByText(/No voice can be spoken yet. Record or import clips with transcripts in the Voices tab./)).toBeInTheDocument();
    expect(speakButton()).toBeDisabled();
  });

  it("does not show that hint when a voice can be spoken", async () => {
    await open();
    expect(screen.queryByText(/No voice can be spoken yet/)).not.toBeInTheDocument();
  });

  it("counts characters against 5000 and disables Speak when empty or over", async () => {
    await open();
    expect(screen.getByText("0 of 5000 characters")).toBeInTheDocument();
    expect(speakButton()).toBeDisabled();
    type("   ");
    expect(speakButton()).toBeDisabled();
    type("Hello");
    expect(screen.getByText("5 of 5000 characters")).toBeInTheDocument();
    expect(speakButton()).toBeEnabled();
    type("a".repeat(5001));
    expect(screen.getByText("5001 of 5000 characters")).toBeInTheDocument();
    expect(speakButton()).toBeDisabled();
    type("a".repeat(5000));
    expect(speakButton()).toBeEnabled();
  });

  it("counts an emoji as one character like the backend does", async () => {
    await open();
    type("😀".repeat(5000));
    expect(speakButton()).toBeEnabled();
  });

  it("offers Fast, Balanced and Best, with Best disabled and explained for a voice without a model", async () => {
    await open();
    const preset = screen.getByRole("combobox", { name: "Speed / quality" });
    expect(preset).toHaveValue("balanced");
    expect(screen.getByRole("option", { name: "Best" })).toBeEnabled();
    await userEvent.selectOptions(preset, "best");
    await userEvent.selectOptions(screen.getByRole("combobox", { name: "Voice" }), "plain");
    expect(screen.getByRole("option", { name: "Best" })).toBeDisabled();
    expect(preset).toHaveValue("balanced");
    expect(screen.getByText(/Best needs a trained model/)).toBeInTheDocument();
  });

  it("listens on the Windows default output, never a VoiceMeeter or cable device, and sends to the virtual mic", async () => {
    await open();
    const listen = screen.getByRole("combobox", { name: "Listen on" });
    expect(listen).toHaveValue("2");
    expect(within(listen).queryByRole("option", { name: /VoiceMeeter|CABLE/ })).not.toBeInTheDocument();
    expect(screen.getByRole("checkbox", { name: /Send to a virtual microphone/ })).toBeChecked();
    type("Hello");
    await userEvent.click(speakButton());
    expect(ttsApi.speak).toHaveBeenCalledWith({
      voice_id: "demo", text: "Hello", preset: "balanced", mic_device: 3, monitor_device: 2, save: false,
    });
  });

  it("falls back to a real output when the default output is a virtual device", async () => {
    vi.mocked(api.devices).mockResolvedValue({ ...DEVICES, default_output: 4 });
    await open();
    expect(screen.getByRole("combobox", { name: "Listen on" })).toHaveValue("2");
  });

  it("shows no virtual mic checkbox when there is none, and sends only to the listen device", async () => {
    vi.mocked(api.devices).mockResolvedValue({ ...DEVICES, devices: [HEADSET, SPEAKERS], virtual_mic: null });
    await open();
    expect(screen.queryByRole("checkbox", { name: /Send to a virtual microphone/ })).not.toBeInTheDocument();
    expect(screen.queryByRole("combobox", { name: "Virtual microphone" })).not.toBeInTheDocument();
    type("Hello");
    await userEvent.click(speakButton());
    expect(ttsApi.speak).toHaveBeenCalledWith(expect.objectContaining({ mic_device: null, monitor_device: 2 }));
  });

  it("lets you choose which virtual mic the speech goes to", async () => {
    await open();
    const mic = screen.getByRole("combobox", { name: "Virtual microphone" });
    expect(mic).toHaveValue("3"); // the detected one
    expect(within(mic).getAllByRole("option").map((o) => o.textContent)).toEqual(["VoiceMeeter Output (sent through VoiceMeeter Input)", "CABLE Output (sent through CABLE Input)"]);
    expect(screen.getByText(/as the microphone\./)).toHaveTextContent("choose VoiceMeeter Output as the microphone");
    await userEvent.selectOptions(mic, "4");
    expect(screen.getByText(/as the microphone\./)).toHaveTextContent("choose CABLE Output as the microphone");
    type("Hello");
    await userEvent.click(speakButton());
    expect(ttsApi.speak).toHaveBeenCalledWith(expect.objectContaining({ mic_device: 4, monitor_device: 2 }));
    // Unticking hides the choice and sends nothing to a virtual mic.
    await userEvent.click(screen.getByRole("checkbox", { name: /Send to a virtual microphone/ }));
    expect(screen.queryByRole("combobox", { name: "Virtual microphone" })).not.toBeInTheDocument();
  });

  it("starts on the virtual mic chosen on the Live tab, and shows no list when there is only one", async () => {
    const { unmount } = render(<SpeakPanel tts={view()} active liveRunning={false} preferredMic={4} />);
    await screen.findByRole("option", { name: /Demo/ });
    expect(screen.getByRole("combobox", { name: "Virtual microphone" })).toHaveValue("4");
    unmount();
    vi.mocked(api.devices).mockResolvedValue({ ...DEVICES, devices: [HEADSET, SPEAKERS, VM] });
    await open();
    expect(screen.getByRole("checkbox", { name: /Send to a virtual microphone/ })).toBeChecked();
    expect(screen.queryByRole("combobox", { name: "Virtual microphone" })).not.toBeInTheDocument();
  });

  it("needs at least one output", async () => {
    await open();
    type("Hello");
    await userEvent.selectOptions(screen.getByRole("combobox", { name: "Listen on" }), "");
    expect(speakButton()).toBeEnabled(); // the virtual mic is still on
    await userEvent.click(screen.getByRole("checkbox", { name: /Send to a virtual microphone/ }));
    expect(speakButton()).toBeDisabled();
    expect(screen.getByText(/Choose a listen device or the virtual mic/)).toBeInTheDocument();
  });

  it("sends save when Keep as a WAV file is ticked", async () => {
    await open();
    type("Hello");
    await userEvent.click(screen.getByRole("checkbox", { name: "Keep as a WAV file" }));
    await userEvent.click(speakButton());
    expect(ttsApi.speak).toHaveBeenCalledWith(expect.objectContaining({ save: true }));
  });

  it("shows a refusal, returns to idle, and clears it on the next attempt", async () => {
    const v = view();
    vi.mocked(ttsApi.speak).mockRejectedValueOnce(new Error("Stop live mode before speaking"));
    await open(v);
    type("Hello");
    await userEvent.click(speakButton());
    expect(await screen.findByRole("alert")).toHaveTextContent("Stop live mode before speaking");
    expect(v.apply).not.toHaveBeenCalled();
    expect(speakButton()).toBeEnabled(); // back to idle
    vi.mocked(ttsApi.speak).mockResolvedValueOnce(idle({ state: "preparing" }));
    await userEvent.click(speakButton());
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
    expect(v.apply).toHaveBeenCalledWith(idle({ state: "preparing" }));
  });

  it("ignores a second press while the request is in flight", async () => {
    let release: (s: TtsStatus) => void = () => {};
    vi.mocked(ttsApi.speak).mockReturnValue(new Promise((r) => { release = r; }));
    await open();
    type("Hello");
    await userEvent.click(speakButton());
    expect(speakButton()).toBeDisabled();
    await userEvent.type(text(), "{Control>}{Enter}{/Control}");
    expect(ttsApi.speak).toHaveBeenCalledTimes(1);
    await act(async () => { release(idle({ state: "preparing" })); });
  });

  it("Ctrl+Enter in the text area speaks, plain Enter does not", async () => {
    await open();
    type("Hello");
    await userEvent.type(text(), "{Enter}");
    expect(ttsApi.speak).not.toHaveBeenCalled();
    await userEvent.type(text(), "{Control>}{Enter}{/Control}");
    expect(ttsApi.speak).toHaveBeenCalledTimes(1);
  });

  it("Ctrl+Enter does nothing when Speak is disabled", async () => {
    await open();
    await userEvent.type(text(), "{Control>}{Enter}{/Control}");
    expect(ttsApi.speak).not.toHaveBeenCalled();
  });

  it("disables Speak with the reason while live mode is running", async () => {
    await open(view(), { liveRunning: true });
    type("Hello");
    expect(speakButton()).toBeDisabled();
    expect(screen.getByText(/Stop live mode on the Live tab to speak/)).toBeInTheDocument();
  });

  it("keeps the typed text when the panel is hidden and shown again", async () => {
    const { rerender } = await open();
    type("Hello");
    rerender(<SpeakPanel tts={view()} active={false} liveRunning={false} />);
    rerender(<SpeakPanel tts={view()} active liveRunning={false} />);
    expect(text()).toHaveValue("Hello");
  });

  it("asks the backend nothing while it was never shown", () => {
    render(<SpeakPanel tts={view()} active={false} liveRunning={false} />);
    expect(ttsApi.voices).not.toHaveBeenCalled();
    expect(api.devices).not.toHaveBeenCalled();
    expect(ttsApi.environment).not.toHaveBeenCalled();
  });

  it("shows why the voices or devices could not be loaded and retries", async () => {
    vi.mocked(ttsApi.voices).mockRejectedValueOnce(new Error("Could not load"));
    render(<SpeakPanel tts={view()} active liveRunning={false} />);
    expect(await screen.findByRole("alert")).toHaveTextContent("Could not load");
    await userEvent.click(screen.getByRole("button", { name: "Retry" }));
    expect(await screen.findByRole("option", { name: /Demo/ })).toBeInTheDocument();
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });
});

describe("SpeakPanel devices", () => {
  const rendered = (props: { devicesVersion?: number; preferredMonitor?: number | null } = {}) => (
    <SpeakPanel tts={view()} active liveRunning={false} {...props} />
  );
  const listen = () => screen.getByRole("combobox", { name: "Listen on" });

  it("reloads the devices when devicesVersion changes and keeps a selection that still exists", async () => {
    const { rerender } = render(rendered({ devicesVersion: 0 }));
    await screen.findByRole("option", { name: /Demo/ });
    await userEvent.selectOptions(listen(), "6");
    const plugged = { index: 9, name: "USB Speaker", max_input: 0, max_output: 2 };
    vi.mocked(api.devices).mockResolvedValue({ ...DEVICES, devices: [...DEVICES.devices, plugged] });
    rerender(rendered({ devicesVersion: 1 }));
    expect(await screen.findByRole("option", { name: "USB Speaker" })).toBeInTheDocument();
    expect(api.devices).toHaveBeenCalledTimes(2);
    expect(listen()).toHaveValue("6");
  });

  it("falls back to the default output when the chosen device is gone after a refresh", async () => {
    const { rerender } = render(rendered({ devicesVersion: 0 }));
    await screen.findByRole("option", { name: /Demo/ });
    await userEvent.selectOptions(listen(), "6");
    vi.mocked(api.devices).mockResolvedValue({ ...DEVICES, devices: [HEADSET, VM, CABLE] });
    rerender(rendered({ devicesVersion: 1 }));
    await act(async () => {});
    expect(screen.queryByRole("option", { name: "Speakers" })).not.toBeInTheDocument();
    expect(listen()).toHaveValue("2");
  });

  it("does not refetch devices when devicesVersion is unchanged", async () => {
    const { rerender } = render(rendered({ devicesVersion: 3 }));
    await screen.findByRole("option", { name: /Demo/ });
    rerender(rendered({ devicesVersion: 3 }));
    expect(api.devices).toHaveBeenCalledTimes(1);
  });

  it("listens on the monitor chosen on the Live tab when it is a real output", async () => {
    render(rendered({ preferredMonitor: 6 }));
    await screen.findByRole("option", { name: /Demo/ });
    expect(listen()).toHaveValue("6");
  });

  it("ignores a Live-tab monitor that is virtual or missing and uses the default output", async () => {
    const { unmount } = render(rendered({ preferredMonitor: 4 }));
    await screen.findByRole("option", { name: /Demo/ });
    expect(listen()).toHaveValue("2");
    unmount();
    render(rendered({ preferredMonitor: 77 }));
    await screen.findByRole("option", { name: /Demo/ });
    expect(listen()).toHaveValue("2");
  });
});

describe("SpeakPanel status", () => {
  it("shows Getting ready while the engine starts, not an error, with Stop available", async () => {
    await open(view(idle({ engine: "starting", state: "preparing" })));
    expect(screen.getByRole("status")).toHaveTextContent("Starting the voice engine. This takes about 15 seconds, and up to half a minute the first time.");
    expect(screen.getByRole("status")).toHaveTextContent("Getting ready");
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Speak" })).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Stop" })).toBeEnabled();
  });

  it("says only that the engine is starting when it starts with no request (after a Stop), without Stop", async () => {
    await open(view(idle({ engine: "starting", state: "idle" })));
    expect(screen.getByRole("status")).toHaveTextContent("The voice engine is starting.");
    expect(screen.getByRole("status")).not.toHaveTextContent("Getting ready");
    expect(screen.queryByRole("button", { name: "Stop" })).not.toBeInTheDocument();
  });

  it("shows a plain line when the engine failed while idle and no request error exists", async () => {
    await open(view(idle({ engine: "failed" })));
    expect(screen.getByRole("alert")).toHaveTextContent(
      "The voice engine stopped. It starts again the next time you speak.",
    );
  });

  it("prefers the request's own error over the engine-failed line", async () => {
    await open(view(idle({ engine: "failed", error: "The worker ran out of memory." })));
    expect(screen.getAllByRole("alert")).toHaveLength(1);
    expect(screen.getByRole("alert")).toHaveTextContent("The worker ran out of memory.");
  });

  it("shows no engine-failed line for a stopped or ready engine", async () => {
    await open(view(idle({ engine: "stopped" })));
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });

  it("shows the sentence being spoken and a Stop button that stops", async () => {
    const v = view(idle({ state: "speaking", piece: 2, pieces: 5 }));
    await open(v);
    expect(screen.getByRole("status")).toHaveTextContent("Speaking sentence 2 of 5");
    expect(screen.queryByRole("button", { name: "Speak" })).not.toBeInTheDocument();
    expect(screen.getByRole("combobox", { name: "Voice" })).toBeDisabled();
    await userEvent.click(screen.getByRole("button", { name: "Stop" }));
    expect(ttsApi.stop).toHaveBeenCalled();
    expect(v.apply).toHaveBeenCalledWith(idle());
  });

  it("shows Stopping while the request ends and does not offer Stop again", async () => {
    await open(view(idle({ state: "stopping", piece: 2, pieces: 5 })));
    expect(screen.getByRole("status")).toHaveTextContent("Stopping");
    expect(screen.getByRole("button", { name: "Stop" })).toBeDisabled();
  });

  it("Escape stops while speaking and does nothing otherwise", async () => {
    const { unmount } = await open(view(idle({ state: "speaking", piece: 1, pieces: 2 })));
    await userEvent.type(text(), "{Escape}");
    expect(ttsApi.stop).toHaveBeenCalledTimes(1);
    unmount();
    vi.mocked(ttsApi.stop).mockClear();
    await open();
    await userEvent.type(text(), "{Escape}");
    expect(ttsApi.stop).not.toHaveBeenCalled();
  });

  it("shows a failed stop and keeps Stop available", async () => {
    vi.mocked(ttsApi.stop).mockRejectedValue(new Error("Could not stop"));
    await open(view(idle({ state: "speaking", piece: 1, pieces: 2 })));
    await userEvent.click(screen.getByRole("button", { name: "Stop" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Could not stop");
    expect(screen.getByRole("button", { name: "Stop" })).toBeEnabled();
  });

  it("shows how the last request ended, and its saved file, until the next one", async () => {
    const { rerender } = await open(view(idle({ error: "Playback failed: device lost" })));
    expect(screen.getByRole("alert")).toHaveTextContent("Playback failed: device lost");
    rerender(<SpeakPanel tts={view(idle({ saved_path: "C:\\data\\tts\\one.wav" }))} active liveRunning={false} />);
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
    expect(screen.getByRole("status")).toHaveTextContent("Saved to C:\\data\\tts\\one.wav");
  });

  it("keeps a failed request's message visible while the user edits the text", async () => {
    vi.mocked(ttsApi.speak).mockRejectedValue(new Error("Unknown voice"));
    await open();
    type("Hello");
    await userEvent.click(speakButton());
    await screen.findByRole("alert");
    type("Hello again");
    expect(screen.getByRole("alert")).toHaveTextContent("Unknown voice");
  });

  it("shows the backend is not reachable, keeps the form, and Retry asks again", async () => {
    const v = view(idle(), { unreachable: true });
    await open(v);
    expect(screen.getByRole("alert")).toHaveTextContent("Could not reach the voice engine. Retrying…");
    await userEvent.click(screen.getByRole("button", { name: "Retry" }));
    expect(v.refresh).toHaveBeenCalled();
    expect(speakButton()).toBeInTheDocument();
  });

  it("waits for the first status before enabling Speak", async () => {
    await open(view(null));
    type("Hello");
    expect(speakButton()).toBeDisabled();
  });
});

describe("SpeakPanel environment", () => {
  const BROKEN: TtsEnvironment = {
    ok: false, ready: false,
    checks: [{ id: "env", ok: false, message: "The TTS environment is missing." }, { id: "gpu", ok: true, message: "GPU fine" }],
  };

  it("lists each failing check, points at the README, and offers Check again instead of Speak", async () => {
    vi.mocked(ttsApi.environment).mockResolvedValue(BROKEN);
    render(<SpeakPanel tts={view()} active liveRunning={false} />);
    expect(await screen.findByText("Text to speech is not set up on this computer. See Text to speech setup in the README.")).toBeInTheDocument();
    expect(screen.getByText("The TTS environment is missing.")).toBeInTheDocument();
    expect(screen.queryByText("GPU fine")).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Speak" })).not.toBeInTheDocument();
    vi.mocked(ttsApi.environment).mockResolvedValue(READY);
    await userEvent.click(screen.getByRole("button", { name: "Check again" }));
    expect(ttsApi.environment).toHaveBeenLastCalledWith(true);
    expect(await screen.findByRole("button", { name: "Speak" })).toBeInTheDocument();
    expect(screen.queryByText(/not set up/)).not.toBeInTheDocument();
  });

  it("does not treat an unchecked environment as broken", async () => {
    vi.mocked(ttsApi.environment).mockResolvedValue({ ok: false, ready: false, checks: [], unknown: true, checked_while_training: true });
    await open();
    expect(screen.queryByText(/not set up/)).not.toBeInTheDocument();
    expect(screen.getByText(/cannot be checked right now/)).toBeInTheDocument();
    expect(speakButton()).toBeInTheDocument();
  });

  it("shows a failed check request and offers Check again", async () => {
    vi.mocked(ttsApi.environment).mockRejectedValue(new Error("Check failed to run"));
    await open();
    expect(await screen.findByRole("alert")).toHaveTextContent("Check failed to run");
    expect(screen.getByRole("button", { name: "Check again" })).toBeInTheDocument();
  });
});
