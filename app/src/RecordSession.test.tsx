import { act, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { RecordSession } from "./RecordSession";

vi.mock("./api", () => ({ api: { devices: vi.fn() } }));
vi.mock("./studioApi", () => ({
  studioApi: { prompt: vi.fn(), recordStart: vi.fn(), recordStop: vi.fn(), recordCancel: vi.fn(), clipAudioUrl: vi.fn() },
}));
const status = vi.hoisted(() => ({ current: null as null | { recording: boolean; level: number; seconds: number; full: boolean } }));
vi.mock("./useStudioStatus", () => ({ useStudioStatus: () => status.current }));
import { api } from "./api";
import { studioApi } from "./studioApi";

const P0 = { index: 0, text: "The quiet river bends north.", total: 200, recorded: 0 };
const P1 = { index: 1, text: "She packed a thick wool jumper.", total: 200, recorded: 1 };
const CLIP = { index: 7, text: P0.text, seconds: 2.4, source: "recorded" as const, prompt: 0, origin: null };
const IDLE = { recording: false, level: 0, seconds: 0, full: false, import: {} as never };

beforeEach(() => {
  vi.resetAllMocks();
  status.current = null;
  vi.mocked(api.devices).mockResolvedValue({
    devices: [
      { index: 1, name: "Webcam Mic", max_input: 1, max_output: 0 },
      { index: 4, name: "Headset Mic", max_input: 1, max_output: 0 },
      { index: 5, name: "Headset", max_input: 0, max_output: 2 },
    ],
    virtual_mic: null, default_input: 4, default_output: 5,
  });
  vi.mocked(studioApi.prompt).mockResolvedValue(P0);
  vi.mocked(studioApi.recordStart).mockResolvedValue(IDLE);
  vi.mocked(studioApi.recordCancel).mockResolvedValue(IDLE);
  vi.mocked(studioApi.recordStop).mockResolvedValue({ accepted: true, reason: null, message: null, clip: CLIP });
  vi.mocked(studioApi.clipAudioUrl).mockResolvedValue("blob:take-7");
  URL.revokeObjectURL = vi.fn();
});

const ready = () => screen.findByRole("button", { name: "Record" });

describe("RecordSession", () => {
  it("shows the sentence, progress, and only input devices with the default chosen", async () => {
    render(<RecordSession voiceId="v-1" active onSaved={() => {}} />);
    await ready();
    expect(screen.getByText(P0.text)).toBeInTheDocument();
    expect(screen.getByText(/Sentence 1 of 200/)).toBeInTheDocument();
    const select = screen.getByLabelText("Microphone") as HTMLSelectElement;
    expect(select.querySelectorAll("option")).toHaveLength(2);
    expect(select.value).toBe("4");
  });

  it("records, stops, and shows the saved take with playback", async () => {
    const onSaved = vi.fn();
    render(<RecordSession voiceId="v-1" active onSaved={onSaved} />);
    await userEvent.click(await ready());
    expect(studioApi.recordStart).toHaveBeenCalledWith("v-1", 4, 0);
    await userEvent.click(await screen.findByRole("button", { name: "Stop" }));
    expect(await screen.findByText(/Saved/)).toBeInTheDocument();
    expect(onSaved).toHaveBeenCalledTimes(1);
    expect(studioApi.clipAudioUrl).toHaveBeenCalledWith("v-1", 7);
    expect(await screen.findByLabelText("Play back the take")).toHaveAttribute("src", "blob:take-7");
    expect(screen.getByRole("button", { name: "Next sentence" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Redo" })).toBeInTheDocument();
  });

  it("releases the take's object URL when it is replaced and when unmounted", async () => {
    vi.mocked(studioApi.clipAudioUrl).mockResolvedValueOnce("blob:take-7").mockResolvedValueOnce("blob:take-8");
    const { unmount } = render(<RecordSession voiceId="v-1" active onSaved={() => {}} />);
    await userEvent.click(await ready());
    await userEvent.click(await screen.findByRole("button", { name: "Stop" }));
    await screen.findByLabelText("Play back the take");
    expect(URL.revokeObjectURL).not.toHaveBeenCalled();
    await userEvent.click(screen.getByRole("button", { name: "Redo" }));
    await userEvent.click(await screen.findByRole("button", { name: "Stop" }));
    await vi.waitFor(() => expect(screen.getByLabelText("Play back the take")).toHaveAttribute("src", "blob:take-8"));
    expect(URL.revokeObjectURL).toHaveBeenCalledWith("blob:take-7");
    unmount();
    expect(URL.revokeObjectURL).toHaveBeenCalledWith("blob:take-8");
  });

  it("shows the level and elapsed time while recording", async () => {
    const { rerender } = render(<RecordSession voiceId="v-1" active onSaved={() => {}} />);
    await userEvent.click(await ready());
    await screen.findByRole("button", { name: "Stop" });
    status.current = { recording: true, level: 0.6, seconds: 3.2, full: false };
    rerender(<RecordSession voiceId="v-1" active onSaved={() => {}} />);
    expect(screen.getByLabelText("Input level")).toHaveValue(0.6);
    expect(screen.getByText(/3\.2 s/)).toBeInTheDocument();
  });

  it("warns that long takes are rejected once the take passes 15 seconds, and not before", async () => {
    const warning = "Takes over 20 seconds are rejected. Finish the sentence and press Stop.";
    const ui = () => <RecordSession voiceId="v-1" active onSaved={() => {}} />;
    const { rerender } = render(ui());
    await ready();
    expect(screen.queryByText(warning)).not.toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Record" }));
    await screen.findByRole("button", { name: "Stop" });
    status.current = { recording: true, level: 0.3, seconds: 15.0, full: false };
    rerender(ui());
    expect(screen.queryByText(warning)).not.toBeInTheDocument();
    status.current = { recording: true, level: 0.3, seconds: 15.2, full: false };
    rerender(ui());
    expect(screen.getByText(warning)).toBeInTheDocument();
    status.current = { recording: true, level: 0.3, seconds: 30, full: true };
    rerender(ui());
    expect(screen.getByText(warning)).toBeInTheDocument();
    expect(screen.getByText(/maximum length/)).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Cancel" }));
    await ready();
    expect(screen.queryByText(warning)).not.toBeInTheDocument();
  });

  it("reloads the microphone list after Refresh devices and keeps the chosen microphone", async () => {
    const ui = (v: number) => <RecordSession voiceId="v-1" active devicesVersion={v} onSaved={() => {}} />;
    const { rerender } = render(ui(0));
    await ready();
    const select = screen.getByLabelText("Microphone") as HTMLSelectElement;
    await userEvent.selectOptions(select, "1");
    expect(api.devices).toHaveBeenCalledTimes(1);
    vi.mocked(api.devices).mockResolvedValue({
      devices: [
        { index: 1, name: "Webcam Mic", max_input: 1, max_output: 0 },
        { index: 4, name: "Headset Mic", max_input: 1, max_output: 0 },
        { index: 6, name: "USB Mic", max_input: 1, max_output: 0 },
      ],
      virtual_mic: null, default_input: 4, default_output: null,
    });
    rerender(ui(1));
    expect(await screen.findByRole("option", { name: "USB Mic" })).toBeInTheDocument();
    expect(api.devices).toHaveBeenCalledTimes(2);
    expect(select.querySelectorAll("option")).toHaveLength(3);
    expect(select.value).toBe("1"); // still the Webcam Mic, not back to the default
    expect(studioApi.prompt).toHaveBeenCalledTimes(1); // the sentence is not reloaded
    rerender(ui(1));
    expect(api.devices).toHaveBeenCalledTimes(2);
  });

  it("follows the chosen microphone to its new number, and falls back to the default when it is gone", async () => {
    const ui = (v: number) => <RecordSession voiceId="v-1" active devicesVersion={v} onSaved={() => {}} />;
    const { rerender } = render(ui(0));
    await ready();
    const select = screen.getByLabelText("Microphone") as HTMLSelectElement;
    await userEvent.selectOptions(select, "1");
    // After a refresh the system may number devices differently: number 1 is now another microphone.
    vi.mocked(api.devices).mockResolvedValue({
      devices: [
        { index: 1, name: "USB Mic", max_input: 1, max_output: 0 },
        { index: 2, name: "Webcam Mic", max_input: 1, max_output: 0 },
        { index: 4, name: "Headset Mic", max_input: 1, max_output: 0 },
      ],
      virtual_mic: null, default_input: 4, default_output: null,
    });
    rerender(ui(1));
    await screen.findByRole("option", { name: "USB Mic" });
    expect(select.value).toBe("2");
    vi.mocked(api.devices).mockResolvedValue({
      devices: [
        { index: 1, name: "USB Mic", max_input: 1, max_output: 0 },
        { index: 4, name: "Headset Mic", max_input: 1, max_output: 0 },
      ],
      virtual_mic: null, default_input: 4, default_output: null,
    });
    rerender(ui(2));
    await vi.waitFor(() => expect(select.querySelectorAll("option")).toHaveLength(2));
    expect(select.value).toBe("4");
    await userEvent.click(screen.getByRole("button", { name: "Record" }));
    expect(studioApi.recordStart).toHaveBeenCalledWith("v-1", 4, 0);
  });

  it("shows a microphone that was plugged in after the screen said none was found", async () => {
    vi.mocked(api.devices).mockResolvedValue({ devices: [], virtual_mic: null, default_input: null, default_output: null });
    const ui = (v: number) => <RecordSession voiceId="v-1" active devicesVersion={v} onSaved={() => {}} />;
    const { rerender } = render(ui(0));
    expect(await ready()).toBeDisabled();
    vi.mocked(api.devices).mockResolvedValue({
      devices: [{ index: 3, name: "New Mic", max_input: 1, max_output: 0 }],
      virtual_mic: null, default_input: null, default_output: null,
    });
    rerender(ui(1));
    await vi.waitFor(() => expect(screen.getByRole("button", { name: "Record" })).toBeEnabled());
    expect(screen.queryByText(/no microphone was found/i)).not.toBeInTheDocument();
    expect((screen.getByLabelText("Microphone") as HTMLSelectElement).value).toBe("3");
  });

  it("never reloads the microphone list during a take, and catches up when the take ends", async () => {
    const ui = (v: number) => <RecordSession voiceId="v-1" active devicesVersion={v} onSaved={() => {}} />;
    const { rerender } = render(ui(0));
    await userEvent.click(await ready());
    await screen.findByRole("button", { name: "Stop" });
    rerender(ui(1));
    await act(async () => {});
    expect(api.devices).toHaveBeenCalledTimes(1);
    await userEvent.click(screen.getByRole("button", { name: "Cancel" }));
    await ready();
    await vi.waitFor(() => expect(api.devices).toHaveBeenCalledTimes(2));
  });

  it("ignores a microphone list that arrives after a take has started", async () => {
    type Devices = Awaited<ReturnType<typeof api.devices>>;
    const ui = (v: number) => <RecordSession voiceId="v-1" active devicesVersion={v} onSaved={() => {}} />;
    const { rerender } = render(ui(0));
    await ready();
    let arrive: (list: Devices) => void = () => {};
    vi.mocked(api.devices).mockReturnValueOnce(new Promise((r) => { arrive = r; }));
    rerender(ui(1));
    await userEvent.click(screen.getByRole("button", { name: "Record" }));
    await screen.findByRole("button", { name: "Stop" });
    await act(async () => {
      arrive({ devices: [{ index: 9, name: "Other Mic", max_input: 1, max_output: 0 }], virtual_mic: null, default_input: 9, default_output: null });
    });
    const select = screen.getByLabelText("Microphone") as HTMLSelectElement;
    expect(select.value).toBe("4");
    expect(select.querySelectorAll("option")).toHaveLength(2);
    await userEvent.click(screen.getByRole("button", { name: "Cancel" }));
    await ready();
    await vi.waitFor(() => expect(api.devices).toHaveBeenCalledTimes(3)); // asked again once the take is over
  });

  it("shows the error when the microphone list cannot be reloaded, and keeps the old list", async () => {
    const ui = (v: number) => <RecordSession voiceId="v-1" active devicesVersion={v} onSaved={() => {}} />;
    const { rerender } = render(ui(0));
    await ready();
    vi.mocked(api.devices).mockRejectedValueOnce(new Error("Backend unreachable"));
    rerender(ui(1));
    expect(await screen.findByRole("alert")).toHaveTextContent("Backend unreachable");
    expect((screen.getByLabelText("Microphone") as HTMLSelectElement).querySelectorAll("option")).toHaveLength(2);
  });

  it("keeps the same sentence and shows why when a take is rejected", async () => {
    vi.mocked(studioApi.recordStop).mockResolvedValue({ accepted: false, reason: "clipped", message: "The recording distorted. Move back.", clip: null });
    const onSaved = vi.fn();
    render(<RecordSession voiceId="v-1" active onSaved={onSaved} />);
    await userEvent.click(await ready());
    await userEvent.click(await screen.findByRole("button", { name: "Stop" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("The recording distorted");
    expect(screen.getByText(P0.text)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Record" })).toBeInTheDocument();
    expect(onSaved).not.toHaveBeenCalled();
  });

  it("moves on with Next sentence", async () => {
    render(<RecordSession voiceId="v-1" active onSaved={() => {}} />);
    await userEvent.click(await ready());
    await userEvent.click(await screen.findByRole("button", { name: "Stop" }));
    vi.mocked(studioApi.prompt).mockResolvedValue(P1);
    await userEvent.click(await screen.findByRole("button", { name: "Next sentence" }));
    expect(await screen.findByText(P1.text)).toBeInTheDocument();
    expect(screen.getByText(/Sentence 2 of 200/)).toBeInTheDocument();
    expect(screen.queryByText(/Saved/)).not.toBeInTheDocument();
  });

  it("Redo records the same sentence again", async () => {
    render(<RecordSession voiceId="v-1" active onSaved={() => {}} />);
    await userEvent.click(await ready());
    await userEvent.click(await screen.findByRole("button", { name: "Stop" }));
    await userEvent.click(await screen.findByRole("button", { name: "Redo" }));
    expect(studioApi.recordStart).toHaveBeenLastCalledWith("v-1", 4, 0);
  });

  it("Skip asks for the sentence after this one", async () => {
    render(<RecordSession voiceId="v-1" active onSaved={() => {}} />);
    await ready();
    vi.mocked(studioApi.prompt).mockResolvedValue(P1);
    await userEvent.click(screen.getByRole("button", { name: "Skip" }));
    expect(studioApi.prompt).toHaveBeenLastCalledWith("v-1", 0);
    expect(await screen.findByText(P1.text)).toBeInTheDocument();
  });

  it("Cancel discards the take", async () => {
    render(<RecordSession voiceId="v-1" active onSaved={() => {}} />);
    await userEvent.click(await ready());
    await userEvent.click(await screen.findByRole("button", { name: "Cancel" }));
    expect(studioApi.recordCancel).toHaveBeenCalled();
    expect(await ready()).toBeInTheDocument();
    expect(studioApi.recordStop).not.toHaveBeenCalled();
  });

  it("shows the backend's message when recording cannot start", async () => {
    vi.mocked(studioApi.recordStart).mockRejectedValue(new Error("Stop live mode before recording"));
    render(<RecordSession voiceId="v-1" active onSaved={() => {}} />);
    await userEvent.click(await ready());
    expect(await screen.findByRole("alert")).toHaveTextContent("Stop live mode before recording");
    expect(screen.getByRole("button", { name: "Record" })).toBeInTheDocument();
  });

  it("says when every sentence is recorded", async () => {
    vi.mocked(studioApi.prompt).mockResolvedValue({ index: null, text: null, total: 200, recorded: 200 });
    render(<RecordSession voiceId="v-1" active onSaved={() => {}} />);
    expect(await screen.findByText(/every sentence is recorded/i)).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Record" })).not.toBeInTheDocument();
  });

  it("cannot record without a microphone", async () => {
    vi.mocked(api.devices).mockResolvedValue({ devices: [], virtual_mic: null, default_input: null, default_output: null });
    render(<RecordSession voiceId="v-1" active onSaved={() => {}} />);
    expect(await ready()).toBeDisabled();
    expect(screen.getByText(/no microphone was found/i)).toBeInTheDocument();
  });

  it("cancels an in-progress take when it is unmounted", async () => {
    const { unmount } = render(<RecordSession voiceId="v-1" active onSaved={() => {}} />);
    await userEvent.click(await ready());
    await screen.findByRole("button", { name: "Stop" });
    unmount();
    expect(studioApi.recordCancel).toHaveBeenCalled();
  });

  it("cancels the take and returns to idle when the tab becomes inactive", async () => {
    const { rerender } = render(<RecordSession voiceId="v-1" active onSaved={() => {}} />);
    await userEvent.click(await ready());
    await screen.findByRole("button", { name: "Stop" });
    rerender(<RecordSession voiceId="v-1" active={false} onSaved={() => {}} />);
    expect(studioApi.recordCancel).toHaveBeenCalledTimes(1);
    expect(await ready()).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Stop" })).not.toBeInTheDocument();
  });

  it("does not cancel when the tab becomes inactive while idle", async () => {
    const { rerender } = render(<RecordSession voiceId="v-1" active onSaved={() => {}} />);
    await ready();
    rerender(<RecordSession voiceId="v-1" active={false} onSaved={() => {}} />);
    expect(studioApi.recordCancel).not.toHaveBeenCalled();
  });

  it("cancels a take whose start request finishes after the tab went inactive", async () => {
    let finish: (v: typeof IDLE) => void = () => {};
    vi.mocked(studioApi.recordStart).mockReturnValue(new Promise((r) => { finish = r; }));
    const { rerender } = render(<RecordSession voiceId="v-1" active onSaved={() => {}} />);
    await userEvent.click(await ready());
    rerender(<RecordSession voiceId="v-1" active={false} onSaved={() => {}} />);
    finish(IDLE);
    await vi.waitFor(() => expect(studioApi.recordCancel).toHaveBeenCalledTimes(1));
    expect(await ready()).toBeInTheDocument();
  });

  it("cancels the take and shows the error when Stop fails, leaving the UI idle", async () => {
    vi.mocked(studioApi.recordStop).mockRejectedValue(new Error("Backend unreachable"));
    render(<RecordSession voiceId="v-1" active onSaved={() => {}} />);
    await userEvent.click(await ready());
    await userEvent.click(await screen.findByRole("button", { name: "Stop" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Backend unreachable");
    expect(studioApi.recordCancel).toHaveBeenCalledTimes(1);
    expect(await ready()).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Stop" })).not.toBeInTheDocument();
  });

  it("keeps the recording controls and shows the error when Cancel fails, so it can be retried", async () => {
    vi.mocked(studioApi.recordCancel).mockRejectedValueOnce(new Error("Cancel failed"));
    render(<RecordSession voiceId="v-1" active onSaved={() => {}} />);
    await userEvent.click(await ready());
    await userEvent.click(await screen.findByRole("button", { name: "Cancel" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Cancel failed");
    expect(screen.getByRole("button", { name: "Stop" })).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Cancel" }));
    expect(studioApi.recordCancel).toHaveBeenCalledTimes(2);
    expect(await ready()).toBeInTheDocument();
  });

  it("stays in the recording state when Stop fails and the cleanup cancel fails too", async () => {
    vi.mocked(studioApi.recordStop).mockRejectedValue(new Error("Backend unreachable"));
    vi.mocked(studioApi.recordCancel).mockRejectedValue(new Error("Still unreachable"));
    render(<RecordSession voiceId="v-1" active onSaved={() => {}} />);
    await userEvent.click(await ready());
    await userEvent.click(await screen.findByRole("button", { name: "Stop" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Backend unreachable");
    expect(screen.getByRole("button", { name: "Stop" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Cancel" })).toBeInTheDocument();
  });

  it("keeps recording state when the automatic cancel fails, and retries it when the tab is active again", async () => {
    vi.mocked(studioApi.recordCancel).mockRejectedValueOnce(new Error("Cancel failed"));
    const { rerender } = render(<RecordSession voiceId="v-1" active onSaved={() => {}} />);
    await userEvent.click(await ready());
    await screen.findByRole("button", { name: "Stop" });
    rerender(<RecordSession voiceId="v-1" active={false} onSaved={() => {}} />);
    expect(await screen.findByRole("alert")).toHaveTextContent("Cancel failed");
    expect(screen.getByRole("button", { name: "Stop" })).toBeInTheDocument();
    expect(studioApi.recordCancel).toHaveBeenCalledTimes(1);
    rerender(<RecordSession voiceId="v-1" active onSaved={() => {}} />);
    await vi.waitFor(() => expect(studioApi.recordCancel).toHaveBeenCalledTimes(2));
    expect(await ready()).toBeInTheDocument();
  });

  it("releases a take's audio that arrives after the component was unmounted", async () => {
    let arrive: (url: string) => void = () => {};
    vi.mocked(studioApi.clipAudioUrl).mockReturnValue(new Promise((r) => { arrive = r; }));
    const { unmount } = render(<RecordSession voiceId="v-1" active onSaved={() => {}} />);
    await userEvent.click(await ready());
    await userEvent.click(await screen.findByRole("button", { name: "Stop" }));
    await vi.waitFor(() => expect(studioApi.clipAudioUrl).toHaveBeenCalled());
    unmount();
    arrive("blob:late");
    await vi.waitFor(() => expect(URL.revokeObjectURL).toHaveBeenCalledWith("blob:late"));
  });
});
