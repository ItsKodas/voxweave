import { act, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useStudioStatus } from "./useStudioStatus";

vi.mock("./studioApi", () => ({ studioApi: { status: vi.fn() } }));
import { studioApi } from "./studioApi";

const STATUS = {
  recording: true, level: 0.4, seconds: 1.2, full: false,
  import: { active: false, finished: false, total_files: 0, done_files: 0, current: null, clips_added: 0, skipped: 0, errors: [], voice_id: null, job: 0 },
};
const tick = (ms: number) => act(async () => { await vi.advanceTimersByTimeAsync(ms); });

beforeEach(() => {
  vi.useFakeTimers();
  vi.mocked(studioApi.status).mockReset().mockResolvedValue(STATUS);
});
afterEach(() => vi.useRealTimers());

describe("useStudioStatus", () => {
  it("is null and silent while disabled", async () => {
    const { result } = renderHook(() => useStudioStatus(false));
    await tick(1000);
    expect(result.current).toBeNull();
    expect(studioApi.status).not.toHaveBeenCalled();
  });

  it("polls while enabled and returns the latest status", async () => {
    const { result } = renderHook(() => useStudioStatus(true, 200));
    await tick(0);
    expect(result.current).toEqual(STATUS);
    await tick(650);
    expect(vi.mocked(studioApi.status).mock.calls.length).toBeGreaterThanOrEqual(3);
  });

  it("stops polling and clears when disabled or unmounted", async () => {
    const { result, rerender, unmount } = renderHook(({ on }) => useStudioStatus(on, 200), { initialProps: { on: true } });
    await tick(250);
    rerender({ on: false });
    expect(result.current).toBeNull();
    const calls = vi.mocked(studioApi.status).mock.calls.length;
    await tick(1000);
    expect(studioApi.status).toHaveBeenCalledTimes(calls);
    unmount();
  });

  it("keeps the last status and keeps polling through a failed poll", async () => {
    const { result } = renderHook(() => useStudioStatus(true, 200));
    await tick(0);
    vi.mocked(studioApi.status).mockRejectedValueOnce(new Error("blip"));
    await tick(200);
    expect(result.current).toEqual(STATUS);
    await tick(400);
    expect(vi.mocked(studioApi.status).mock.calls.length).toBeGreaterThanOrEqual(3);
  });

  it("does not overlap requests when the backend is slow", async () => {
    let release: (v: typeof STATUS) => void = () => {};
    vi.mocked(studioApi.status).mockReturnValue(new Promise((r) => { release = r; }));
    renderHook(() => useStudioStatus(true, 200));
    await tick(1000);
    expect(studioApi.status).toHaveBeenCalledTimes(1);
    release(STATUS);
  });
});
