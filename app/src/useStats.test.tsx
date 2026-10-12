import { act, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { api } from "./api";
import { IDLE, useStats } from "./useStats";

vi.mock("./api", () => ({ api: { wsUrl: vi.fn() } }));

class FakeSocket {
  static instances: FakeSocket[] = [];
  onmessage: ((e: { data: string }) => void) | null = null;
  onclose: (() => void) | null = null;
  closed = false;
  constructor(public url: string) {
    FakeSocket.instances.push(this);
  }
  close() {
    this.closed = true;
  }
}

const flush = () => act(async () => { await vi.advanceTimersByTimeAsync(0); });

beforeEach(() => {
  vi.useFakeTimers();
  FakeSocket.instances = [];
  vi.stubGlobal("WebSocket", FakeSocket);
  vi.mocked(api.wsUrl).mockReset().mockResolvedValue("ws://x/ws");
});
afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

describe("useStats", () => {
  it("updates from messages", async () => {
    const { result } = renderHook(() => useStats());
    await flush();
    act(() => FakeSocket.instances[0].onmessage?.({ data: JSON.stringify({ ...IDLE, running: true, mode: "converted" }) }));
    expect(result.current.running).toBe(true);
  });

  it("resets to idle on close but keeps the last error, then reconnects", async () => {
    const { result } = renderHook(() => useStats());
    await flush();
    act(() => FakeSocket.instances[0].onmessage?.({ data: JSON.stringify({ ...IDLE, running: true, error: "bad" }) }));
    act(() => FakeSocket.instances[0].onclose?.());
    expect(result.current).toEqual({ ...IDLE, error: "bad" });
    await act(async () => { await vi.advanceTimersByTimeAsync(1000); });
    expect(FakeSocket.instances).toHaveLength(2);
  });

  it("does not leave a socket open when unmounted during the pending url lookup", async () => {
    let resolve!: (u: string) => void;
    vi.mocked(api.wsUrl).mockReturnValue(new Promise((r) => { resolve = r; }));
    const { unmount } = renderHook(() => useStats());
    unmount();
    resolve("ws://x/ws");
    await flush();
    expect(FakeSocket.instances.every((s) => s.closed)).toBe(true);
  });

  it("retries when the url lookup fails", async () => {
    vi.mocked(api.wsUrl).mockRejectedValueOnce(new Error("no port"));
    renderHook(() => useStats());
    await flush();
    expect(FakeSocket.instances).toHaveLength(0);
    await act(async () => { await vi.advanceTimersByTimeAsync(1000); });
    expect(FakeSocket.instances).toHaveLength(1);
  });

  it("ignores malformed frames", async () => {
    const { result } = renderHook(() => useStats());
    await flush();
    expect(() => act(() => FakeSocket.instances[0].onmessage?.({ data: "{not json" }))).not.toThrow();
    expect(result.current).toEqual(IDLE);
  });
});
