import { renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { useLiveShortcuts } from "./useLiveShortcuts";
import { IDLE } from "./useStats";

const handlers: Record<string, (event: { payload: unknown }) => void> = {};
vi.mock("@tauri-apps/api/event", () => ({
  listen: vi.fn(async (name: string, cb: (event: { payload: unknown }) => void) => {
    handlers[name] = cb;
    return () => undefined;
  }),
}));
vi.mock("./api", () => ({ api: { mode: vi.fn(), stop: vi.fn() } }));
import { api } from "./api";

const running = (mode: "converted" | "muted") => ({ current: { ...IDLE, running: true, mode } });

async function setup(stats: { current: typeof IDLE }) {
  const onError = vi.fn();
  renderHook(() => useLiveShortcuts(stats, onError));
  await vi.waitFor(() => expect(handlers["live-toggle"]).toBeDefined());
  await vi.waitFor(() => expect(handlers["live-mode"]).toBeDefined());
  return onError;
}

beforeEach(() => {
  vi.resetAllMocks();
  for (const k of Object.keys(handlers)) delete handlers[k];
  vi.mocked(api.mode).mockResolvedValue(IDLE);
  vi.mocked(api.stop).mockResolvedValue(IDLE);
});

describe("useLiveShortcuts", () => {
  it("mutes when toggled while converting", async () => {
    const onError = await setup(running("converted"));
    handlers["live-toggle"]({ payload: null });
    await vi.waitFor(() => expect(api.mode).toHaveBeenCalledWith("muted"));
    await vi.waitFor(() => expect(onError).toHaveBeenCalledWith(null));
  });

  it("unmutes when toggled while muted", async () => {
    await setup(running("muted"));
    handlers["live-toggle"]({ payload: null });
    await vi.waitFor(() => expect(api.mode).toHaveBeenCalledWith("converted"));
  });

  it("does nothing when toggled while not running", async () => {
    await setup({ current: IDLE });
    handlers["live-toggle"]({ payload: null });
    await new Promise((r) => setTimeout(r, 20));
    expect(api.mode).not.toHaveBeenCalled();
  });

  it("stops from the tray's off item", async () => {
    await setup({ current: IDLE });
    handlers["live-mode"]({ payload: "off" });
    await vi.waitFor(() => expect(api.stop).toHaveBeenCalled());
  });

  it("sets the mode from the tray while running", async () => {
    await setup(running("converted"));
    handlers["live-mode"]({ payload: "muted" });
    await vi.waitFor(() => expect(api.mode).toHaveBeenCalledWith("muted"));
  });

  it("reports a failed mute", async () => {
    vi.mocked(api.mode).mockRejectedValue(new Error("boom"));
    const onError = await setup(running("converted"));
    handlers["live-toggle"]({ payload: null });
    await vi.waitFor(() => expect(onError).toHaveBeenCalledWith("Could not mute: boom"));
  });

  it("reports a failed stop", async () => {
    vi.mocked(api.stop).mockRejectedValue(new Error("nope"));
    const onError = await setup({ current: IDLE });
    handlers["live-mode"]({ payload: "off" });
    await vi.waitFor(() => expect(onError).toHaveBeenCalledWith("Could not stop: nope"));
  });
});
