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
});
