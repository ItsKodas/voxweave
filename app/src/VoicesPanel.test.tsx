import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { VoicesPanel } from "./VoicesPanel";

vi.mock("./studioApi", () => ({
  studioApi: { voices: vi.fn(), createVoice: vi.fn(), deleteVoice: vi.fn() },
}));
vi.mock("./VoiceStudio", () => ({
  VoiceStudio: ({ voice, onChanged, devicesVersion, onModelReady }: { voice: { name: string }; onChanged: () => void; devicesVersion?: number; onModelReady?: () => void }) => (
    <>
      <button onClick={onModelReady}>model ready for {voice.name}</button>
      <button onClick={onChanged}>studio for {voice.name}</button>
      <p>studio devices {devicesVersion}</p>
    </>
  ),
}));
import { studioApi } from "./studioApi";

const A = { id: "a-111111", name: "Alpha", created: "1", minutes: 4.5, clips: 30, models: [] };
const B = { id: "b-222222", name: "Beta", created: "2", minutes: 0, clips: 0, models: [] };

beforeEach(() => {
  vi.resetAllMocks();
  vi.mocked(studioApi.voices).mockResolvedValue([A, B]);
  vi.mocked(studioApi.createVoice).mockResolvedValue({ id: "c-333333", name: "Gamma", created: "3", minutes: 0, clips: 0, models: [] });
  vi.mocked(studioApi.deleteVoice).mockResolvedValue({ deleted: true });
});

describe("VoicesPanel", () => {
  it("lists voices with their recorded minutes and opens the first one", async () => {
    render(<VoicesPanel active />);
    const list = await screen.findByRole("list", { name: "Voices" });
    expect(within(list).getByText("Alpha")).toBeInTheDocument();
    expect(within(list).getByText(/4\.5 min/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "studio for Alpha" })).toBeInTheDocument();
  });

  it("switches the studio when another voice is chosen", async () => {
    render(<VoicesPanel active />);
    await userEvent.click(await screen.findByRole("button", { name: /^Beta/ }));
    expect(screen.getByRole("button", { name: "studio for Beta" })).toBeInTheDocument();
  });

  it("explains what to do when there are no voices", async () => {
    vi.mocked(studioApi.voices).mockResolvedValue([]);
    render(<VoicesPanel active />);
    expect(await screen.findByText(/create a voice to start recording/i)).toBeInTheDocument();
  });

  it("creates a voice, selects it and clears the name field", async () => {
    render(<VoicesPanel active />);
    await screen.findByRole("list", { name: "Voices" });
    vi.mocked(studioApi.voices).mockResolvedValue([A, B, { id: "c-333333", name: "Gamma", created: "3", minutes: 0, clips: 0, models: [] }]);
    const field = screen.getByLabelText("New voice name");
    await userEvent.type(field, "Gamma");
    await userEvent.click(screen.getByRole("button", { name: "Create voice" }));
    expect(studioApi.createVoice).toHaveBeenCalledWith("Gamma");
    expect(await screen.findByRole("button", { name: "studio for Gamma" })).toBeInTheDocument();
    expect(field).toHaveValue("");
  });

  it("cannot create a voice with a blank name", async () => {
    render(<VoicesPanel active />);
    await screen.findByRole("list", { name: "Voices" });
    expect(screen.getByRole("button", { name: "Create voice" })).toBeDisabled();
    await userEvent.type(screen.getByLabelText("New voice name"), "   ");
    expect(screen.getByRole("button", { name: "Create voice" })).toBeDisabled();
  });

  it("shows the backend's message when creating fails", async () => {
    vi.mocked(studioApi.createVoice).mockRejectedValue(new Error("A voice name must be 1 to 60 characters on one line"));
    render(<VoicesPanel active />);
    await screen.findByRole("list", { name: "Voices" });
    await userEvent.type(screen.getByLabelText("New voice name"), "x");
    await userEvent.click(screen.getByRole("button", { name: "Create voice" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("1 to 60 characters");
  });

  it("deletes a voice only after confirmation", async () => {
    const confirm = vi.spyOn(window, "confirm").mockReturnValue(false);
    render(<VoicesPanel active />);
    await userEvent.click(await screen.findByRole("button", { name: "Delete voice Alpha" }));
    expect(confirm).toHaveBeenCalled();
    expect(confirm.mock.calls[0][0]).toMatch(/Alpha/);
    expect(confirm.mock.calls[0][0]).toMatch(/trash/i);
    expect(studioApi.deleteVoice).not.toHaveBeenCalled();
    confirm.mockReturnValue(true);
    vi.mocked(studioApi.voices).mockResolvedValue([B]);
    await userEvent.click(screen.getByRole("button", { name: "Delete voice Alpha" }));
    expect(studioApi.deleteVoice).toHaveBeenCalledWith("a-111111");
    expect(await screen.findByRole("button", { name: "studio for Beta" })).toBeInTheDocument();
    confirm.mockRestore();
  });

  it("reloads the list when the studio reports a change", async () => {
    render(<VoicesPanel active />);
    await userEvent.click(await screen.findByRole("button", { name: "studio for Alpha" }));
    await vi.waitFor(() => expect(studioApi.voices).toHaveBeenCalledTimes(2));
  });

  it("shows a load error with a retry", async () => {
    vi.mocked(studioApi.voices).mockRejectedValueOnce(new Error("backend unreachable"));
    render(<VoicesPanel active />);
    expect(await screen.findByRole("alert")).toHaveTextContent("backend unreachable");
    await userEvent.click(screen.getByRole("button", { name: "Retry" }));
    expect(await screen.findByRole("list", { name: "Voices" })).toBeInTheDocument();
  });

  it("hands the device list version to the open voice", async () => {
    const { rerender } = render(<VoicesPanel active devicesVersion={1} />);
    expect(await screen.findByText("studio devices 1")).toBeInTheDocument();
    rerender(<VoicesPanel active devicesVersion={2} />);
    expect(screen.getByText("studio devices 2")).toBeInTheDocument();
  });

  it("does not load until the tab is first shown", async () => {
    const { rerender } = render(<VoicesPanel active={false} />);
    expect(studioApi.voices).not.toHaveBeenCalled();
    rerender(<VoicesPanel active />);
    await vi.waitFor(() => expect(studioApi.voices).toHaveBeenCalledTimes(1));
  });

  it("reloads the voices and tells the app when a voice's training has finished", async () => {
    const onModelReady = vi.fn();
    render(<VoicesPanel active onModelReady={onModelReady} />);
    await userEvent.click(await screen.findByRole("button", { name: "model ready for Alpha" }));
    expect(onModelReady).toHaveBeenCalledTimes(1);
    await vi.waitFor(() => expect(studioApi.voices).toHaveBeenCalledTimes(2));
  });
});
