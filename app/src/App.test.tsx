import { act, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import App from "./App";

vi.mock("./api", () => ({ api: { health: vi.fn() } }));
vi.mock("./useStats", () => ({ useStats: () => ({ running: false }) }));
vi.mock("./useLiveShortcuts", () => ({ useLiveShortcuts: () => {} }));
vi.mock("./SetupCheck", () => ({
  SetupCheck: ({ refreshKey }: { refreshKey: number }) => <p>setup check {refreshKey}</p>,
}));
vi.mock("./LivePanel", () => ({
  LivePanel: ({ onDevicesRefreshed, voicesVersion }: { onDevicesRefreshed: () => void; voicesVersion?: number }) => (
    <>
      <button onClick={onDevicesRefreshed}>live panel</button>
      <p>live voices {voicesVersion}</p>
    </>
  ),
}));
vi.mock("./VoicesPanel", () => ({
  VoicesPanel: ({ active, devicesVersion, onModelReady, onTrainingStarted }: {
    active: boolean; devicesVersion?: number; onModelReady?: () => void; onTrainingStarted?: () => void;
  }) => (
    <>
      <button onClick={onModelReady}>model ready</button>
      <button onClick={onTrainingStarted}>training started</button>
      <p>voices panel {active ? "active" : "idle"}</p>
      <p>voices devices {devicesVersion}</p>
    </>
  ),
}));
vi.mock("./trainingApi", async (importOriginal) => ({
  ...(await importOriginal<typeof import("./trainingApi")>()),
  trainingApi: { status: vi.fn() },
}));
import { api } from "./api";
import { TrainingStatus, trainingApi } from "./trainingApi";

const job = (over: Partial<TrainingStatus> = {}): TrainingStatus => ({
  state: "idle", voice_id: null, stage: null, stage_index: null, stages: 7, epoch: null, epochs: null, percent: 0,
  eta_seconds: null, message: null, error: null, started: null, finished: null, can_resume: false, has_checkpoint: false, ...over,
});
const RUNNING = job({ state: "running", voice_id: "v1", stage: "train", percent: 40 });
const statusCalls = () => vi.mocked(trainingApi.status).mock.calls.length;

const advance = (ms: number) => act(async () => { await vi.advanceTimersByTimeAsync(ms); });

beforeEach(() => {
  vi.useFakeTimers();
  vi.mocked(api.health).mockReset();
  vi.mocked(trainingApi.status).mockReset().mockResolvedValue(job());
});
afterEach(() => vi.useRealTimers());

describe("App", () => {
  it("shows the panel as soon as the voice engine answers", async () => {
    vi.mocked(api.health).mockResolvedValue({ status: "ok" });
    render(<App />);
    await advance(0);
    expect(screen.getByRole("button", { name: "live panel" })).toBeInTheDocument();
    expect(screen.queryByText(/starting the voice engine/i)).not.toBeInTheDocument();
  });

  it("shows only the starting message for the first 15 seconds", async () => {
    vi.mocked(api.health).mockRejectedValue(new Error("refused"));
    render(<App />);
    await advance(14_000);
    expect(screen.getByText(/starting the voice engine/i)).toBeInTheDocument();
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
    expect(trainingApi.status).not.toHaveBeenCalled(); // nothing is asked about training before the engine answers
  });

  it("explains the failure, its likely causes and the log after 15 seconds without an answer", async () => {
    vi.mocked(api.health).mockRejectedValue(new Error("refused"));
    render(<App />);
    await advance(15_000);
    const alert = screen.getByRole("alert");
    expect(alert).toHaveTextContent(/the voice engine failed to start/i);
    expect(alert).toHaveTextContent(/backend environment is not set up/i);
    expect(alert).toHaveTextContent(/README/);
    expect(alert).toHaveTextContent("data/settings.json");
    expect(alert).toHaveTextContent(/GPU runtime/i);
    expect(alert).toHaveTextContent("data/backend.log");
    expect(screen.queryByText(/starting the voice engine/i)).not.toBeInTheDocument();
  });

  it("keeps polling after the failure message and recovers on its own", async () => {
    vi.mocked(api.health).mockRejectedValue(new Error("refused"));
    render(<App />);
    await advance(15_000);
    const calls = vi.mocked(api.health).mock.calls.length;
    await advance(3_000);
    expect(vi.mocked(api.health).mock.calls.length).toBeGreaterThan(calls);
    vi.mocked(api.health).mockResolvedValue({ status: "ok" });
    await advance(1_000);
    expect(screen.getByRole("button", { name: "live panel" })).toBeInTheDocument();
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });

  it("re-runs the setup check when the panel reports a device refresh", async () => {
    vi.mocked(api.health).mockResolvedValue({ status: "ok" });
    render(<App />);
    await advance(0);
    expect(screen.getByText("setup check 0")).toBeInTheDocument();
    act(() => screen.getByRole("button", { name: "live panel" }).click());
    expect(screen.getByText("setup check 1")).toBeInTheDocument();
  });

  it("tells the Voices tab when the Live tab refreshed the devices", async () => {
    vi.mocked(api.health).mockResolvedValue({ status: "ok" });
    render(<App />);
    await advance(0);
    expect(screen.getByText("voices devices 0")).toBeInTheDocument();
    act(() => screen.getByRole("button", { name: "live panel" }).click());
    expect(screen.getByText("voices devices 1")).toBeInTheDocument();
  });

  it("reloads the Live tab's voice list when a model is ready, without touching the devices", async () => {
    vi.mocked(api.health).mockResolvedValue({ status: "ok" });
    render(<App />);
    await advance(0);
    expect(screen.getByText("live voices 0")).toBeInTheDocument();
    act(() => screen.getByRole("button", { name: "model ready", hidden: true }).click());
    expect(screen.getByText("live voices 1")).toBeInTheDocument();
    expect(screen.getByText("setup check 0")).toBeInTheDocument();
    expect(screen.getByText("voices devices 0")).toBeInTheDocument();
  });

  it("starts on the Live tab with the Voices panel idle", async () => {
    vi.mocked(api.health).mockResolvedValue({ status: "ok" });
    render(<App />);
    await advance(0);
    expect(screen.getByRole("tab", { name: "Live" })).toHaveAttribute("aria-selected", "true");
    expect(screen.getByRole("button", { name: "live panel" })).toBeVisible();
    expect(screen.getByText("voices panel idle")).not.toBeVisible();
  });

  it("reloads the Live tab's voice list whenever the Live tab is shown again", async () => {
    vi.mocked(api.health).mockResolvedValue({ status: "ok" });
    render(<App />);
    await advance(0);
    act(() => screen.getByRole("tab", { name: "Live" }).click()); // already shown: nothing to reload
    expect(screen.getByText("live voices 0")).toBeInTheDocument();
    act(() => screen.getByRole("tab", { name: "Voices" }).click());
    expect(screen.getByText("live voices 0")).toBeInTheDocument();
    act(() => screen.getByRole("tab", { name: "Live" }).click());
    expect(screen.getByText("live voices 1")).toBeInTheDocument();
    expect(screen.getByText("setup check 0")).toBeInTheDocument();
  });

  it("switches to the Voices tab and keeps the Live panel mounted", async () => {
    vi.mocked(api.health).mockResolvedValue({ status: "ok" });
    render(<App />);
    await advance(0);
    act(() => screen.getByRole("tab", { name: "Voices" }).click());
    expect(screen.getByText("voices panel active")).toBeVisible();
    expect(screen.getByRole("button", { name: "live panel", hidden: true })).not.toBeVisible();
    act(() => screen.getByRole("tab", { name: "Live" }).click());
    expect(screen.getByRole("button", { name: "live panel" })).toBeVisible();
  });
});

describe("App follows training whichever tab is shown", () => {
  beforeEach(() => vi.mocked(api.health).mockResolvedValue({ status: "ok" }));

  it("asks once at start and does not poll while nothing is training", async () => {
    render(<App />);
    await advance(0);
    expect(trainingApi.status).toHaveBeenCalledTimes(1);
    expect(trainingApi.status).toHaveBeenCalledWith(); // the app-wide job, not one voice's
    await advance(60_000);
    expect(statusCalls()).toBe(1);
    expect(screen.getByText("live voices 0")).toBeInTheDocument();
  });

  it("picks up a job that is already running and reloads the Live voices exactly once when it finishes there", async () => {
    vi.mocked(trainingApi.status).mockResolvedValue(RUNNING);
    render(<App />);
    await advance(0);
    expect(screen.getByRole("tab", { name: "Live" })).toHaveAttribute("aria-selected", "true");
    await advance(3000);
    expect(statusCalls()).toBe(4); // at once, then every second
    expect(screen.getByText("live voices 0")).toBeInTheDocument();
    vi.mocked(trainingApi.status).mockResolvedValue(job({ state: "done", voice_id: "v1", percent: 100 }));
    await advance(1000);
    expect(screen.getByText("live voices 1")).toBeInTheDocument();
    const settled = statusCalls();
    await advance(30_000);
    expect(statusCalls()).toBe(settled); // and the polling has stopped
    expect(screen.getByText("live voices 1")).toBeInTheDocument();
    expect(screen.getByText("setup check 0")).toBeInTheDocument();
  });

  it("starts polling when a panel reports a start or a resume", async () => {
    render(<App />);
    await advance(0);
    vi.mocked(trainingApi.status).mockResolvedValue(RUNNING);
    await advance(5000);
    expect(statusCalls()).toBe(1); // nobody said a job started
    act(() => screen.getByRole("button", { name: "training started", hidden: true }).click());
    await advance(0);
    expect(statusCalls()).toBe(2);
    await advance(2000);
    expect(statusCalls()).toBe(4);
    vi.mocked(trainingApi.status).mockResolvedValue(job({ state: "done", voice_id: "v1", percent: 100 }));
    await advance(1000);
    expect(screen.getByText("live voices 1")).toBeInTheDocument();
  });

  it.each(["cancelled", "failed", "interrupted"] as const)("does not reload the Live voices when the job ends %s", async (state) => {
    vi.mocked(trainingApi.status).mockResolvedValue(RUNNING);
    render(<App />);
    await advance(1000);
    vi.mocked(trainingApi.status).mockResolvedValue(job({ state, voice_id: "v1" }));
    await advance(1000);
    const settled = statusCalls();
    await advance(10_000);
    expect(statusCalls()).toBe(settled);
    expect(screen.getByText("live voices 0")).toBeInTheDocument();
  });

  it("keeps following a running job through a failed poll", async () => {
    vi.mocked(trainingApi.status).mockResolvedValue(RUNNING);
    render(<App />);
    await advance(0);
    vi.mocked(trainingApi.status).mockRejectedValueOnce(new Error("blip"));
    await advance(1000);
    vi.mocked(trainingApi.status).mockResolvedValue(job({ state: "done", voice_id: "v1", percent: 100 }));
    await advance(1000);
    expect(screen.getByText("live voices 1")).toBeInTheDocument();
  });

  it("asks again every 3 seconds until the first answer arrives", async () => {
    vi.mocked(trainingApi.status).mockRejectedValue(new Error("down"));
    render(<App />);
    await advance(0);
    expect(statusCalls()).toBe(1);
    await advance(2999);
    expect(statusCalls()).toBe(1);
    vi.mocked(trainingApi.status).mockResolvedValue(job());
    await advance(1);
    expect(statusCalls()).toBe(2);
    await advance(30_000);
    expect(statusCalls()).toBe(2);
  });
});
