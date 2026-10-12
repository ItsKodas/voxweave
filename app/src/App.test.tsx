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
  LivePanel: ({ onDevicesRefreshed }: { onDevicesRefreshed: () => void }) => (
    <button onClick={onDevicesRefreshed}>live panel</button>
  ),
}));
import { api } from "./api";

const advance = (ms: number) => act(async () => { await vi.advanceTimersByTimeAsync(ms); });

beforeEach(() => {
  vi.useFakeTimers();
  vi.mocked(api.health).mockReset();
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
});
