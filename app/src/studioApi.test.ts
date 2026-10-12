import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn(async () => 4242) }));
import { studioApi } from "./studioApi";

const fetchMock = vi.fn();

beforeEach(() => {
  fetchMock.mockReset();
  vi.stubGlobal("fetch", fetchMock);
  URL.createObjectURL = vi.fn(() => "blob:clip-1");
});
afterEach(() => vi.unstubAllGlobals());

describe("studioApi.clipAudioUrl", () => {
  it("fetches the audio route and returns an object URL for the body", async () => {
    const blob = new Blob(["wav"], { type: "audio/wav" });
    fetchMock.mockResolvedValue({ ok: true, blob: async () => blob });
    await expect(studioApi.clipAudioUrl("v 1", 3)).resolves.toBe("blob:clip-1");
    expect(fetchMock).toHaveBeenCalledWith("http://127.0.0.1:4242/library/voices/v%201/clips/3/audio");
    expect(URL.createObjectURL).toHaveBeenCalledWith(blob);
  });

  it("throws the backend's message and creates no object URL when refused", async () => {
    fetchMock.mockResolvedValue({ ok: false, status: 403, json: async () => ({ detail: "Origin not allowed" }) });
    await expect(studioApi.clipAudioUrl("v-1", 3)).rejects.toThrow("Origin not allowed");
    expect(URL.createObjectURL).not.toHaveBeenCalled();
  });

  it("falls back to a generic message when the error has no detail", async () => {
    fetchMock.mockResolvedValue({ ok: false, status: 500, json: async () => { throw new Error("not json"); } });
    await expect(studioApi.clipAudioUrl("v-1", 3)).rejects.toThrow("Request failed (500)");
  });
});
