import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { CalibrationPanel, levelZone } from "./CalibrationPanel";
import { calibrationApi, CalibrationStatus, VoiceProfile } from "./calibrationApi";

vi.mock("./calibrationApi", () => ({
  calibrationApi: { status: vi.fn(), start: vi.fn(), finish: vi.fn(), cancel: vi.fn(), forget: vi.fn(), suggestion: vi.fn(), testAudioUrl: vi.fn() },
}));

const PROFILE: VoiceProfile = {
  pitch_hz: 118.4, pitch_low_hz: 95, pitch_high_hz: 160, level_db: -22.3, noise_db: -61.7, gate_db: -55.7,
  input_gain: 1.3, device: "Headset Mic", created: 1,
};
const idle = (over: Partial<CalibrationStatus> = {}): CalibrationStatus => ({
  recording: false, seconds: 0, level: 0, phase: null, quiet_seconds: 3, test_seconds: 4, test_phrase: "Testing, one, two, three", min_read_seconds: 8, total_seconds: 22,
  text: "The ferry leaves the harbour.", error: null, profile: null, ...over,
});
const taking = (seconds: number): CalibrationStatus =>
  idle({ recording: true, seconds, level: 0.4, phase: seconds < 3 ? "quiet" : seconds < 7 ? "testing" : "reading" });

beforeEach(() => {
  vi.resetAllMocks();
  vi.mocked(calibrationApi.status).mockResolvedValue(idle());
  vi.mocked(calibrationApi.cancel).mockResolvedValue(idle());
});
afterEach(() => vi.useRealTimers());

const tick = async (ms: number) => {
  await act(async () => {
    await vi.advanceTimersByTimeAsync(ms);
  });
};

describe("CalibrationPanel", () => {
  it("says it is not calibrated and reports no profile", async () => {
    const onProfile = vi.fn();
    render(<CalibrationPanel input={4} disabled={false} onProfile={onProfile} />);
    expect(await screen.findByText(/Not calibrated yet/)).toBeInTheDocument();
    expect(onProfile).toHaveBeenLastCalledWith(null);
    expect(screen.getByRole("button", { name: "Calibrate my voice" })).toBeEnabled();
  });

  it("shows what an earlier calibration measured and can forget it", async () => {
    vi.mocked(calibrationApi.status).mockResolvedValue(idle({ profile: PROFILE }));
    vi.mocked(calibrationApi.forget).mockResolvedValue(idle());
    const onProfile = vi.fn();
    render(<CalibrationPanel input={4} disabled={false} onProfile={onProfile} />);
    expect(await screen.findByText(/Calibrated on Headset Mic: your usual pitch is 118 Hz/)).toBeInTheDocument();
    expect(onProfile).toHaveBeenLastCalledWith(PROFILE);
    fireEvent.click(screen.getByRole("button", { name: "Forget calibration" }));
    expect(await screen.findByText(/Not calibrated yet/)).toBeInTheDocument();
    expect(onProfile).toHaveBeenLastCalledWith(null);
  });

  it("cannot start without a microphone or while the devices are in use", async () => {
    const { rerender } = render(<CalibrationPanel input={null} disabled={false} onProfile={() => {}} />);
    expect(await screen.findByRole("button", { name: "Calibrate my voice" })).toBeDisabled();
    rerender(<CalibrationPanel input={4} disabled onProfile={() => {}} />);
    expect(screen.getByRole("button", { name: "Calibrate my voice" })).toBeDisabled();
  });

  it("leads through quiet, the test phrase, then reading, and finishes by itself when the time is up", async () => {
    vi.useFakeTimers();
    vi.mocked(calibrationApi.start).mockResolvedValue(taking(0));
    vi.mocked(calibrationApi.finish).mockResolvedValue(idle({ profile: PROFILE }));
    const onProfile = vi.fn();
    render(<CalibrationPanel input={4} disabled={false} onProfile={onProfile} />);
    await tick(0);
    fireEvent.click(screen.getByRole("button", { name: "Calibrate my voice" }));
    await tick(0);
    expect(calibrationApi.start).toHaveBeenCalledWith(4);
    expect(screen.getByText(/Stay quiet for a moment/)).toBeInTheDocument();

    vi.mocked(calibrationApi.status).mockResolvedValue(taking(4));
    await tick(250);
    expect(screen.getByText(/It sets your microphone level/)).toBeInTheDocument();
    expect(screen.getByText(/Testing, one, two, three/)).toBeInTheDocument();

    vi.mocked(calibrationApi.status).mockResolvedValue(taking(12));
    await tick(250);
    expect(screen.getByText(/read the whole paragraph at your normal pace, then press Done/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Done" })).toBeDisabled(); // under 8 s of reading yet
    expect(screen.getByText(/stops by itself after 10 more seconds/)).toBeInTheDocument();
    expect(screen.getByText("The ferry leaves the harbour.")).toBeInTheDocument();
    expect(calibrationApi.finish).not.toHaveBeenCalled();

    vi.mocked(calibrationApi.status).mockResolvedValue(taking(22.1));
    await tick(250);
    expect(calibrationApi.finish).toHaveBeenCalledTimes(1);
    expect(screen.getByText(/Calibrated on Headset Mic/)).toBeInTheDocument();
    expect(onProfile).toHaveBeenLastCalledWith(PROFILE);
    await tick(1000);
    expect(calibrationApi.finish).toHaveBeenCalledTimes(1); // the polling stopped with the take
  });

  it("shows why a take could not be used and offers another go", async () => {
    vi.useFakeTimers();
    vi.mocked(calibrationApi.start).mockResolvedValue(taking(0));
    vi.mocked(calibrationApi.finish).mockRejectedValue(new Error("Not enough speech was heard."));
    render(<CalibrationPanel input={4} disabled={false} onProfile={() => {}} />);
    await tick(0);
    fireEvent.click(screen.getByRole("button", { name: "Calibrate my voice" }));
    await tick(0);
    vi.mocked(calibrationApi.status).mockResolvedValueOnce(taking(22.5)).mockResolvedValue(idle());
    await tick(250);
    await tick(0);
    expect(screen.getByRole("alert")).toHaveTextContent("Not enough speech was heard.");
    expect(screen.getByRole("button", { name: "Calibrate my voice" })).toBeEnabled();
  });

  it("Cancel drops the take, and a start that is refused shows the reason", async () => {
    vi.mocked(calibrationApi.start).mockResolvedValue(taking(1));
    render(<CalibrationPanel input={4} disabled={false} onProfile={() => {}} />);
    fireEvent.click(await screen.findByRole("button", { name: "Calibrate my voice" }));
    fireEvent.click(await screen.findByRole("button", { name: "Cancel" }));
    await waitFor(() => expect(calibrationApi.cancel).toHaveBeenCalledTimes(1));
    expect(await screen.findByRole("button", { name: "Calibrate my voice" })).toBeInTheDocument();

    vi.mocked(calibrationApi.start).mockRejectedValue(new Error("Stop live mode before calibrating"));
    fireEvent.click(screen.getByRole("button", { name: "Calibrate my voice" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Stop live mode before calibrating");
  });

  it("closes the microphone if the panel goes away during a take", async () => {
    vi.mocked(calibrationApi.start).mockResolvedValue(taking(1));
    const { unmount } = render(<CalibrationPanel input={4} disabled={false} onProfile={() => {}} />);
    fireEvent.click(await screen.findByRole("button", { name: "Calibrate my voice" }));
    await screen.findByRole("button", { name: "Cancel" });
    unmount();
    expect(calibrationApi.cancel).toHaveBeenCalledTimes(1);
  });
});


describe("finishing, the level bar and the mic check", () => {
  it("Done finishes once enough has been read", async () => {
    vi.useFakeTimers();
    vi.mocked(calibrationApi.start).mockResolvedValue(taking(0));
    vi.mocked(calibrationApi.finish).mockResolvedValue(idle({ profile: PROFILE }));
    render(<CalibrationPanel input={4} disabled={false} onProfile={() => {}} />);
    await tick(0);
    fireEvent.click(screen.getByRole("button", { name: "Calibrate my voice" }));
    await tick(0);
    vi.mocked(calibrationApi.status).mockResolvedValue(taking(16));
    await tick(250);
    fireEvent.click(screen.getByRole("button", { name: "Done" }));
    await tick(0);
    expect(calibrationApi.finish).toHaveBeenCalledTimes(1);
    expect(screen.getByText(/Calibrated on Headset Mic/)).toBeInTheDocument();
  });

  it("colours the level: too quiet and too loud red, good green, loud yellow", () => {
    expect(levelZone(0.01)).toEqual({ label: "Too quiet", zone: "quiet" });
    expect(levelZone(0.2)).toEqual({ label: "Good", zone: "good" });
    expect(levelZone(0.7)).toEqual({ label: "Loud", zone: "loud" });
    expect(levelZone(0.95)).toEqual({ label: "Too loud", zone: "clip" });
  });

  it("shows the level while recording", async () => {
    vi.useFakeTimers();
    vi.mocked(calibrationApi.start).mockResolvedValue({ ...taking(4), level: 0.01 });
    render(<CalibrationPanel input={4} disabled={false} onProfile={() => {}} />);
    await tick(0);
    fireEvent.click(screen.getByRole("button", { name: "Calibrate my voice" }));
    await tick(0);
    expect(screen.getByText("Too quiet")).toBeInTheDocument();
  });

  it("plays back the mic check after calibrating", async () => {
    const played: string[] = [];
    class FakeAudio {
      onended: (() => void) | null = null;
      onerror: (() => void) | null = null;
      constructor(public src: string) {}
      play() { played.push(this.src); return Promise.resolve(); }
    }
    vi.stubGlobal("Audio", FakeAudio);
    URL.revokeObjectURL = vi.fn();
    vi.mocked(calibrationApi.status).mockResolvedValue(idle({ profile: PROFILE }));
    vi.mocked(calibrationApi.testAudioUrl).mockResolvedValue("blob:check");
    render(<CalibrationPanel input={4} disabled={false} onProfile={() => {}} />);
    fireEvent.click(await screen.findByRole("button", { name: "Play my mic check" }));
    await waitFor(() => expect(played).toEqual(["blob:check"]));
    vi.unstubAllGlobals();
  });
});
