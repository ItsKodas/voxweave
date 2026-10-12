import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { LibraryVoice } from "./studioApi";
import { TrainPanel } from "./TrainPanel";
import { TrainingEnvironment, TrainingStatus } from "./trainingApi";

vi.mock("./trainingApi", async (importOriginal) => ({
  ...(await importOriginal<typeof import("./trainingApi")>()),
  trainingApi: { status: vi.fn(), environment: vi.fn(), start: vi.fn(), cancel: vi.fn() },
}));
import { trainingApi } from "./trainingApi";

const VOICE: LibraryVoice = { id: "v1", name: "Alpha", created: "1", minutes: 10, clips: 60, models: [] };
const make = (over: Partial<TrainingStatus> = {}): TrainingStatus => ({
  state: "idle", voice_id: null, stage: null, stage_index: null, stages: 7, epoch: null, epochs: null,
  percent: 0, eta_seconds: null, message: null, error: null, started: null, finished: null, can_resume: false, ...over,
});
const READY: TrainingEnvironment = { ready: true, ok: true, checks: [{ id: "gpu", ok: true, message: "GPU found" }] };
const running = (over: Partial<TrainingStatus> = {}) =>
  make({ state: "running", voice_id: "v1", stage: "train", stage_index: 5, epoch: 12, epochs: 200, percent: 37.4, eta_seconds: 720, ...over });

/** Answers by voice: `own` for this voice, `latest` for the app-wide question. */
function answer(own: TrainingStatus, latest: TrainingStatus = own) {
  vi.mocked(trainingApi.status).mockImplementation(async (id?: string) => (id === undefined ? latest : own));
}

beforeEach(() => {
  vi.resetAllMocks();
  answer(make());
  vi.mocked(trainingApi.environment).mockResolvedValue(READY);
  vi.mocked(trainingApi.start).mockResolvedValue(running());
  vi.mocked(trainingApi.cancel).mockResolvedValue(make({ state: "cancelling", voice_id: "v1" }));
  vi.spyOn(window, "confirm").mockReturnValue(true);
});

const panel = (voice = VOICE, onModelReady = vi.fn(), active = true) =>
  render(<TrainPanel voice={voice} active={active} onModelReady={onModelReady} />);

describe("TrainPanel before training", () => {
  it("asks for more audio below 5 minutes and offers no Start", async () => {
    panel({ ...VOICE, minutes: 3.25 });
    expect(await screen.findByText("Record or import at least 5 minutes to train. You have 3.2.")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /start training|train again/i })).not.toBeInTheDocument();
  });

  it("offers epochs, advanced batch size, an estimate and Start when ready", async () => {
    panel();
    const epochs = await screen.findByLabelText("Epochs");
    expect(epochs).toHaveValue("200");
    expect(within(epochs).getAllByRole("option").map((o) => o.textContent)).toEqual(["100", "200", "300"]);
    // 200 epochs x 10 min x 0.95 s = 31.7 min, rounded up to the next 5
    expect(screen.getByText("About 35 minutes")).toBeInTheDocument();
    await userEvent.selectOptions(epochs, "100");
    expect(screen.getByText("About 20 minutes")).toBeInTheDocument(); // 15.8 min
    await userEvent.click(screen.getByText("Advanced"));
    const batch = screen.getByLabelText("Batch size");
    expect(batch).toHaveValue("8");
    expect(within(batch).getAllByRole("option").map((o) => o.textContent)).toEqual(["4", "8", "16"]);
    await userEvent.selectOptions(batch, "16");
    await userEvent.click(screen.getByRole("button", { name: "Start training" }));
    expect(trainingApi.start).toHaveBeenCalledWith({ voice_id: "v1", epochs: 100, batch_size: 16, resume: false });
  });

  it("lists each failing check, explains, and offers Check again instead of Start", async () => {
    vi.mocked(trainingApi.environment).mockResolvedValue({
      ready: false, ok: false,
      checks: [{ id: "a", ok: true, message: "fine" }, { id: "b", ok: false, message: "GPU not visible" }, { id: "c", ok: false, message: "ffmpeg missing" }],
    });
    panel();
    expect(await screen.findByText("GPU not visible")).toBeInTheDocument();
    expect(screen.getByText("ffmpeg missing")).toBeInTheDocument();
    expect(screen.queryByText("fine")).not.toBeInTheDocument();
    expect(screen.getByText("Training is not set up on this computer. See Training setup in the README.")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /start training/i })).not.toBeInTheDocument();
    vi.mocked(trainingApi.environment).mockResolvedValue(READY);
    await userEvent.click(screen.getByRole("button", { name: "Check again" }));
    expect(trainingApi.environment).toHaveBeenLastCalledWith(true);
    expect(await screen.findByRole("button", { name: "Start training" })).toBeInTheDocument();
  });

  it("shows a refusal from the backend in an alert", async () => {
    vi.mocked(trainingApi.start).mockRejectedValue(new Error("Stop live mode before training"));
    panel();
    await userEvent.click(await screen.findByRole("button", { name: "Start training" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Stop live mode before training");
    expect(screen.getByRole("button", { name: "Start training" })).toBeEnabled();
  });
});

describe("TrainPanel while training", () => {
  beforeEach(() => answer(running()));

  it("shows the stage, epoch, percent, time left and the unavailable notice", async () => {
    panel();
    expect(await screen.findByText("Training")).toBeInTheDocument();
    expect(screen.getByText("Epoch 12 of 200")).toBeInTheDocument();
    expect(screen.getByRole("progressbar", { name: "Training progress" })).toHaveValue(37.4);
    expect(screen.getByText("37%")).toBeInTheDocument();
    expect(screen.getByText("About 12 minutes left")).toBeInTheDocument();
    expect(screen.getByText(/Live mode and recording are unavailable until training finishes/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Cancel" })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /start training/i })).not.toBeInTheDocument();
  });

  it.each([
    ["prepare", "Preparing"], ["preprocess", "Slicing audio"], ["pitch", "Finding pitch"],
    ["features", "Extracting features"], ["index", "Building index"], ["export", "Exporting"],
  ] as const)("names the %s stage in plain words", async (stage, words) => {
    answer(running({ stage, epoch: null }));
    panel();
    expect(await screen.findByText(words)).toBeInTheDocument();
    expect(screen.queryByText(/^Epoch/)).not.toBeInTheDocument();
  });

  it.each([
    [null, "Estimating time…"], [30, "Less than a minute left"], [59, "Less than a minute left"],
    [60, "About 1 minute left"], [90, "About 2 minutes left"], [3540, "About 59 minutes left"],
    [3600, "About 1 h 0 min left"], [5400, "About 1 h 30 min left"], [7500, "About 2 h 5 min left"],
  ])("formats %s seconds left as %s", async (eta, words) => {
    answer(running({ eta_seconds: eta }));
    panel();
    expect(await screen.findByText(words)).toBeInTheDocument();
  });

  it("asks before stopping, then shows Stopping… until the state changes", async () => {
    panel();
    await userEvent.click(await screen.findByRole("button", { name: "Cancel" }));
    expect(window.confirm).toHaveBeenCalledWith("Stop training? You can resume it later.");
    expect(trainingApi.cancel).toHaveBeenCalledTimes(1);
    answer(make({ state: "cancelling", voice_id: "v1", stage: "train", percent: 37 }));
    expect(await screen.findByText("Stopping…")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Cancel" })).not.toBeInTheDocument();
    answer(make({ state: "cancelled", voice_id: "v1", can_resume: true }));
    expect(await screen.findByText("Training was stopped", {}, { timeout: 3000 })).toBeInTheDocument();
    expect(screen.queryByText("Stopping…")).not.toBeInTheDocument();
  });

  it("does nothing when the stop is not confirmed", async () => {
    vi.mocked(window.confirm).mockReturnValue(false);
    panel();
    await userEvent.click(await screen.findByRole("button", { name: "Cancel" }));
    expect(trainingApi.cancel).not.toHaveBeenCalled();
    expect(screen.queryByText("Stopping…")).not.toBeInTheDocument();
  });

  it("shows a stop that the backend refuses", async () => {
    vi.mocked(trainingApi.cancel).mockRejectedValue(new Error("Training is not running"));
    panel();
    await userEvent.click(await screen.findByRole("button", { name: "Cancel" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Training is not running");
  });

  it("stays truthful when mounted again: the running job is shown from the backend alone", async () => {
    const first = panel();
    await screen.findByText("Epoch 12 of 200");
    first.unmount();
    panel();
    expect(await screen.findByText("Epoch 12 of 200")).toBeInTheDocument();
  });
});

describe("TrainPanel after training", () => {
  it("announces a finished run and tells the Live tab exactly once", async () => {
    answer(running());
    const onModelReady = vi.fn();
    const { rerender } = panel(VOICE, onModelReady);
    await screen.findByText("Epoch 12 of 200");
    answer(make({ state: "done", voice_id: "v1", percent: 100 }));
    expect(await screen.findByText("Training finished. This voice is now available on the Live tab.", {}, { timeout: 3000 })).toBeInTheDocument();
    expect(onModelReady).toHaveBeenCalledTimes(1);
    rerender(<TrainPanel voice={{ ...VOICE, models: ["rvc"] }} active onModelReady={onModelReady} />);
    await new Promise((r) => setTimeout(r, 50));
    expect(onModelReady).toHaveBeenCalledTimes(1);
  });

  it("does not announce a finished run it never saw running", async () => {
    answer(make({ state: "done", voice_id: "v1", percent: 100 }));
    const onModelReady = vi.fn();
    panel({ ...VOICE, models: ["rvc"] }, onModelReady);
    expect(await screen.findByText(/Training finished/)).toBeInTheDocument();
    expect(onModelReady).not.toHaveBeenCalled();
  });

  it("shows a failure with its log location, Resume when the backend says so, and Start over", async () => {
    answer(make({ state: "failed", voice_id: "v1", error: "Out of memory", can_resume: true }));
    panel();
    expect(await screen.findByRole("alert")).toHaveTextContent("Out of memory");
    expect(screen.getByText(/data\/voices\/v1\/training\/worker\.log/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Resume" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Start over" })).toBeInTheDocument();
  });

  it("offers no Resume for a failure the backend says cannot resume", async () => {
    answer(make({ state: "failed", voice_id: "v1", error: "The clips changed since this training run started", can_resume: false }));
    panel();
    expect(await screen.findByRole("alert")).toHaveTextContent("The clips changed");
    expect(screen.queryByRole("button", { name: "Resume" })).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Start over" })).toBeInTheDocument();
  });

  it.each([
    ["cancelled", "Training was stopped"],
    ["interrupted", "Training was interrupted when the app closed"],
  ] as const)("says what happened after %s", async (state, words) => {
    answer(make({ state, voice_id: "v1", can_resume: true }));
    panel();
    expect(await screen.findByText(words)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Resume" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Start over" })).toBeInTheDocument();
  });

  it("resumes with the stopped run's own settings", async () => {
    answer(make({ state: "interrupted", voice_id: "v1", can_resume: true }));
    panel();
    await userEvent.click(await screen.findByRole("button", { name: "Resume" }));
    expect(trainingApi.start).toHaveBeenCalledWith({ voice_id: "v1", resume: true });
    expect(window.confirm).not.toHaveBeenCalled();
  });

  it("confirms before Start over discards the progress", async () => {
    answer(make({ state: "cancelled", voice_id: "v1", can_resume: true }));
    panel();
    vi.mocked(window.confirm).mockReturnValueOnce(false);
    await userEvent.click(await screen.findByRole("button", { name: "Start over" }));
    expect(window.confirm).toHaveBeenCalledWith("Start over? The progress so far will be discarded.");
    expect(trainingApi.start).not.toHaveBeenCalled();
    await userEvent.click(screen.getByRole("button", { name: "Start over" }));
    expect(trainingApi.start).toHaveBeenCalledWith({ voice_id: "v1", epochs: 200, batch_size: 8, resume: false });
  });
});

describe("TrainPanel with a trained model", () => {
  const TRAINED = { ...VOICE, models: ["rvc"] };

  it("shows the model is ready and offers Train again after confirming the replacement", async () => {
    panel(TRAINED);
    expect(await screen.findByText("Trained model: ready")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Start training" })).not.toBeInTheDocument();
    vi.mocked(window.confirm).mockReturnValueOnce(false);
    await userEvent.click(screen.getByRole("button", { name: "Train again" }));
    expect(window.confirm).toHaveBeenCalledWith("Train again? The current model is replaced only when the new training finishes.");
    expect(trainingApi.start).not.toHaveBeenCalled();
    await userEvent.click(screen.getByRole("button", { name: "Train again" }));
    expect(trainingApi.start).toHaveBeenCalledWith({ voice_id: "v1", epochs: 200, batch_size: 8, resume: false });
  });

  it("shows nothing about a model for a voice without one", async () => {
    panel();
    await screen.findByRole("button", { name: "Start training" });
    expect(screen.queryByText("Trained model: ready")).not.toBeInTheDocument();
  });
});

describe("TrainPanel while another voice trains", () => {
  it("says so with that job's percent and offers no Start", async () => {
    answer(make(), running({ voice_id: "v2", percent: 61.2 }));
    panel();
    expect(await screen.findByText("Another voice is training")).toBeInTheDocument();
    expect(screen.getByText("61%")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /start training|train again|resume|start over/i })).not.toBeInTheDocument();
  });

  it("tells the Live tab when that other job finishes", async () => {
    answer(make(), running({ voice_id: "v2" }));
    const onModelReady = vi.fn();
    panel(VOICE, onModelReady);
    await screen.findByText("Another voice is training");
    answer(make(), make({ state: "done", voice_id: "v2", percent: 100 }));
    await waitFor(() => expect(onModelReady).toHaveBeenCalledTimes(1), { timeout: 3000 });
    expect(await screen.findByRole("button", { name: "Start training" })).toBeInTheDocument();
  });
});

describe("TrainPanel visibility", () => {
  it("does not ask the backend while the Voices tab is hidden", async () => {
    panel(VOICE, vi.fn(), false);
    await new Promise((r) => setTimeout(r, 50));
    expect(trainingApi.status).not.toHaveBeenCalled();
    expect(trainingApi.environment).not.toHaveBeenCalled();
  });
});
