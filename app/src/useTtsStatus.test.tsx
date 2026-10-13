import { act, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { TtsStatus } from "./ttsApi";
import { useTtsStatus } from "./useTtsStatus";

vi.mock("./ttsApi", async (importOriginal) => ({
  ...(await importOriginal<typeof import("./ttsApi")>()),
  ttsApi: { status: vi.fn() },
}));
import { ttsApi } from "./ttsApi";

const make = (over: Partial<TtsStatus> = {}): TtsStatus => ({
  engine: "ready", state: "idle", voice_id: null, preset: null, piece: 0, pieces: 0, error: null,
  saved_path: null, started: null, ...over,
});
const SPEAKING = make({ state: "speaking", piece: 1, pieces: 3 });
const tick = (ms: number) => act(async () => { await vi.advanceTimersByTimeAsync(ms); });
const calls = () => vi.mocked(ttsApi.status).mock.calls.length;

beforeEach(() => {
  vi.useFakeTimers();
  vi.mocked(ttsApi.status).mockReset().mockResolvedValue(make());
});
afterEach(() => vi.useRealTimers());

describe("useTtsStatus", () => {
  it("asks once at mount and does not poll while idle", async () => {
    const { result } = renderHook(() => useTtsStatus(false));
    await tick(10_000);
    expect(calls()).toBe(1);
    expect(result.current.status?.state).toBe("idle");
  });

  it("polls twice a second while a request is active, even when the tab is hidden, then stops", async () => {
    vi.mocked(ttsApi.status).mockResolvedValue(SPEAKING);
    const { result } = renderHook(() => useTtsStatus(false));
    await tick(0);
    expect(result.current.status?.piece).toBe(1);
    await tick(1500);
    expect(calls()).toBe(4); // at once, 0.5 s, 1 s, 1.5 s
    vi.mocked(ttsApi.status).mockResolvedValue(make());
    await tick(500);
    const settled = calls();
    await tick(10_000);
    expect(calls()).toBe(settled);
  });

  it("treats a starting engine as active", async () => {
    vi.mocked(ttsApi.status).mockResolvedValue(make({ engine: "starting", state: "preparing" }));
    renderHook(() => useTtsStatus(true));
    await tick(1000);
    expect(calls()).toBe(3);
  });

  it("asks once more each time the tab becomes visible", async () => {
    const { rerender } = renderHook(({ v }) => useTtsStatus(v), { initialProps: { v: false } });
    await tick(0);
    expect(calls()).toBe(1);
    rerender({ v: true });
    await tick(0);
    expect(calls()).toBe(2);
    rerender({ v: false });
    await tick(0);
    expect(calls()).toBe(2);
  });

  it("never overlaps requests when the backend is slow", async () => {
    let release: (s: TtsStatus) => void = () => {};
    vi.mocked(ttsApi.status).mockReturnValue(new Promise((r) => { release = r; }));
    renderHook(() => useTtsStatus(true));
    await tick(5000);
    expect(calls()).toBe(1);
    release(SPEAKING);
  });

  it("a refresh during a slow request waits for it, then asks again", async () => {
    let release: (s: TtsStatus) => void = () => {};
    vi.mocked(ttsApi.status).mockReturnValueOnce(new Promise((r) => { release = r; }));
    const { result } = renderHook(() => useTtsStatus(true));
    await tick(0);
    act(() => result.current.refresh());
    expect(calls()).toBe(1);
    await act(async () => { release(make()); });
    await tick(0);
    expect(calls()).toBe(2);
  });

  it("reports unreachable after a failed poll, keeps the last status, retries every 3 seconds and recovers", async () => {
    vi.mocked(ttsApi.status).mockResolvedValueOnce(SPEAKING).mockRejectedValue(new Error("down"));
    const { result } = renderHook(() => useTtsStatus(true));
    await tick(500);
    expect(result.current.unreachable).toBe(true);
    expect(result.current.status?.state).toBe("speaking");
    const after = calls();
    await tick(3000);
    expect(calls()).toBe(after + 1);
    vi.mocked(ttsApi.status).mockResolvedValue(make());
    await tick(3000);
    expect(result.current.unreachable).toBe(false);
    expect(result.current.status?.state).toBe("idle");
    const settled = calls();
    await tick(10_000);
    expect(calls()).toBe(settled);
  });

  it("apply shows an answer at once and starts polling when it is active", async () => {
    const { result } = renderHook(() => useTtsStatus(true));
    await tick(0);
    act(() => result.current.apply(SPEAKING));
    expect(result.current.status?.state).toBe("speaking");
    await tick(500);
    expect(calls()).toBeGreaterThan(1);
  });

  it("stops and makes no state update after unmount", async () => {
    vi.mocked(ttsApi.status).mockResolvedValue(SPEAKING);
    const { unmount } = renderHook(() => useTtsStatus(true));
    await tick(0);
    unmount();
    const settled = calls();
    await tick(5000);
    expect(calls()).toBe(settled);
  });

  it("ignores an answer that arrives after unmount", async () => {
    let release: (s: TtsStatus) => void = () => {};
    vi.mocked(ttsApi.status).mockReturnValue(new Promise((r) => { release = r; }));
    const { unmount } = renderHook(() => useTtsStatus(true));
    unmount();
    await act(async () => { release(SPEAKING); });
    await tick(5000);
    expect(calls()).toBe(1);
  });
});
