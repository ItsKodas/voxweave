import { act, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { TrainingStatus } from "./trainingApi";
import { useTrainingStatus } from "./useTrainingStatus";

vi.mock("./trainingApi", async (importOriginal) => ({
  ...(await importOriginal<typeof import("./trainingApi")>()),
  trainingApi: { status: vi.fn() },
}));
import { trainingApi } from "./trainingApi";

const make = (over: Partial<TrainingStatus> = {}): TrainingStatus => ({
  state: "idle", voice_id: null, stage: null, stage_index: null, stages: 7, epoch: null, epochs: null,
  percent: 0, eta_seconds: null, message: null, error: null, started: null, finished: null, can_resume: false, ...over,
});
const RUNNING = make({ state: "running", voice_id: "v1", stage: "train", percent: 40 });
const DONE = make({ state: "done", voice_id: "v1", percent: 100 });
const tick = (ms: number) => act(async () => { await vi.advanceTimersByTimeAsync(ms); });
const calls = () => vi.mocked(trainingApi.status).mock.calls.length;

beforeEach(() => {
  vi.useFakeTimers();
  vi.mocked(trainingApi.status).mockReset().mockResolvedValue(make());
});
afterEach(() => vi.useRealTimers());

describe("useTrainingStatus", () => {
  it("is silent while disabled", async () => {
    const { result } = renderHook(() => useTrainingStatus("v1", false));
    await tick(3000);
    expect(trainingApi.status).not.toHaveBeenCalled();
    expect(result.current.status).toBeNull();
  });

  it("asks once when nothing is running and does not poll", async () => {
    const { result } = renderHook(() => useTrainingStatus("v1", true));
    await tick(5000);
    expect(result.current.status?.state).toBe("idle");
    // this voice's status, then the app-wide one to learn about another voice's job
    expect(trainingApi.status).toHaveBeenNthCalledWith(1, "v1");
    expect(trainingApi.status).toHaveBeenNthCalledWith(2);
    expect(calls()).toBe(2);
  });

  it("polls once a second while running, then stops when the job ends", async () => {
    vi.mocked(trainingApi.status).mockResolvedValue(RUNNING);
    const { result } = renderHook(() => useTrainingStatus("v1", true));
    await tick(0);
    expect(result.current.status?.percent).toBe(40);
    await tick(3000);
    expect(calls()).toBe(4); // immediately, then at 1 s, 2 s, 3 s (own status only: it is running)
    vi.mocked(trainingApi.status).mockResolvedValue(DONE);
    await tick(1000);
    expect(result.current.status?.state).toBe("done");
    const settled = calls();
    await tick(5000);
    expect(calls()).toBe(settled);
  });

  it("never overlaps requests when the backend is slow", async () => {
    let release: (s: TrainingStatus) => void = () => {};
    vi.mocked(trainingApi.status).mockReturnValue(new Promise((r) => { release = r; }));
    renderHook(() => useTrainingStatus("v1", true));
    await tick(5000);
    expect(calls()).toBe(1);
    release(RUNNING);
  });

  it("a refresh during a slow request waits for it, then asks again", async () => {
    let release: (s: TrainingStatus) => void = () => {};
    vi.mocked(trainingApi.status).mockReturnValueOnce(new Promise((r) => { release = r; }));
    const { result } = renderHook(() => useTrainingStatus("v1", true));
    await tick(0);
    act(() => result.current.refresh());
    await tick(0);
    expect(calls()).toBe(1);
    await act(async () => { release(make()); });
    await tick(0);
    expect(calls()).toBeGreaterThan(1);
  });

  it("stops polling when disabled and refreshes once immediately when enabled again", async () => {
    vi.mocked(trainingApi.status).mockResolvedValue(RUNNING);
    const { rerender } = renderHook(({ on }) => useTrainingStatus("v1", on), { initialProps: { on: true } });
    await tick(1500);
    rerender({ on: false });
    const before = calls();
    await tick(5000);
    expect(calls()).toBe(before);
    rerender({ on: true });
    await tick(0);
    expect(calls()).toBe(before + 1);
  });

  it("keeps the last status through a failed poll and keeps polling", async () => {
    vi.mocked(trainingApi.status).mockResolvedValue(RUNNING);
    const { result } = renderHook(() => useTrainingStatus("v1", true));
    await tick(0);
    vi.mocked(trainingApi.status).mockRejectedValueOnce(new Error("blip"));
    await tick(1000);
    expect(result.current.status?.percent).toBe(40);
    const before = calls();
    await tick(2000);
    expect(calls()).toBeGreaterThan(before);
  });

  it("refresh asks again now and starts polling when a job turned out to be running", async () => {
    const { result } = renderHook(() => useTrainingStatus("v1", true));
    await tick(0);
    vi.mocked(trainingApi.status).mockResolvedValue(RUNNING);
    act(() => result.current.refresh());
    await tick(0);
    expect(result.current.status?.state).toBe("running");
    const before = calls();
    await tick(2000);
    expect(calls()).toBe(before + 2);
  });

  it("reports another voice's running job as elsewhere and keeps polling for it", async () => {
    const other = make({ state: "running", voice_id: "v2", percent: 55 });
    vi.mocked(trainingApi.status).mockImplementation(async (id?: string) => (id === undefined ? other : make()));
    const { result } = renderHook(() => useTrainingStatus("v1", true));
    await tick(0);
    expect(result.current.elsewhere?.percent).toBe(55);
    const before = calls();
    await tick(2000);
    expect(calls()).toBeGreaterThan(before);
    vi.mocked(trainingApi.status).mockImplementation(async (id?: string) => (id === undefined ? make({ state: "done", voice_id: "v2" }) : make()));
    await tick(1000);
    expect(result.current.elsewhere?.state).toBe("done");
  });

  it("does not call this voice's own job elsewhere", async () => {
    vi.mocked(trainingApi.status).mockResolvedValue(DONE);
    const { result } = renderHook(() => useTrainingStatus("v1", true));
    await tick(0);
    expect(result.current.elsewhere).toBeNull();
  });
});
