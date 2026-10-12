import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { SetupCheck } from "./SetupCheck";

vi.mock("./api", () => ({ api: { setupCheck: vi.fn() } }));
import { api } from "./api";

describe("SetupCheck", () => {
  it("shows nothing when every check passes", async () => {
    vi.mocked(api.setupCheck).mockResolvedValue([{ id: "directml", ok: true, message: "fine" }]);
    const { container } = render(<SetupCheck />);
    await vi.waitFor(() => expect(api.setupCheck).toHaveBeenCalled());
    expect(container).toBeEmptyDOMElement();
  });

  it("lists only the failing checks with their messages", async () => {
    vi.mocked(api.setupCheck).mockResolvedValue([
      { id: "directml", ok: true, message: "fine" },
      { id: "voice", ok: false, message: "No voice models installed." },
    ]);
    render(<SetupCheck />);
    expect(await screen.findByText("No voice models installed.")).toBeInTheDocument();
    expect(screen.queryByText("fine")).not.toBeInTheDocument();
  });

  it("runs the checks again when refreshKey changes", async () => {
    vi.mocked(api.setupCheck).mockReset().mockResolvedValueOnce([{ id: "virtual_mic", ok: false, message: "No virtual mic found." }]);
    const { rerender } = render(<SetupCheck refreshKey={0} />);
    expect(await screen.findByText("No virtual mic found.")).toBeInTheDocument();
    vi.mocked(api.setupCheck).mockResolvedValueOnce([{ id: "virtual_mic", ok: true, message: "found" }]);
    rerender(<SetupCheck refreshKey={1} />);
    await vi.waitFor(() => expect(screen.queryByText("No virtual mic found.")).not.toBeInTheDocument());
    expect(api.setupCheck).toHaveBeenCalledTimes(2);
  });
});
