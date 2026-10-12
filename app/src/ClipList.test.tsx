import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { ClipList } from "./ClipList";

vi.mock("./studioApi", () => ({
  studioApi: { clips: vi.fn(), editClip: vi.fn(), deleteClip: vi.fn(), clipAudioUrl: vi.fn() },
}));
import { studioApi } from "./studioApi";

const C1 = { index: 1, text: "First one.", seconds: 2.5, source: "recorded" as const, prompt: 0, origin: null };
const C2 = { index: 2, text: "Second won.", seconds: 4.0, source: "imported" as const, prompt: null, origin: "talk.mp3" };

beforeEach(() => {
  vi.resetAllMocks();
  vi.mocked(studioApi.clips).mockResolvedValue([C1, C2]);
  vi.mocked(studioApi.editClip).mockImplementation(async (_id, index, text) => ({ ...(index === 1 ? C1 : C2), text }));
  vi.mocked(studioApi.deleteClip).mockResolvedValue({ deleted: true });
  vi.mocked(studioApi.clipAudioUrl).mockImplementation(async (_id, index) => `blob:x/${index}`);
  URL.revokeObjectURL = vi.fn();
});

const rows = async () => within(await screen.findByRole("list", { name: "Clips" })).getAllByRole("listitem");

describe("ClipList", () => {
  it("lists clips newest first with source and length", async () => {
    render(<ClipList voiceId="v-1" version={0} onChanged={() => {}} />);
    const items = await rows();
    expect(items).toHaveLength(2);
    expect(within(items[0]).getByDisplayValue("Second won.")).toBeInTheDocument();
    expect(items[0]).toHaveTextContent("talk.mp3");
    expect(items[0]).toHaveTextContent("4.0 s");
    expect(items[1]).toHaveTextContent(/recorded/i);
  });

  it("says when there are no clips yet", async () => {
    vi.mocked(studioApi.clips).mockResolvedValue([]);
    render(<ClipList voiceId="v-1" version={0} onChanged={() => {}} />);
    expect(await screen.findByText(/no clips yet/i)).toBeInTheDocument();
  });

  it("saves an edited transcript when the field loses focus", async () => {
    render(<ClipList voiceId="v-1" version={0} onChanged={() => {}} />);
    const field = within((await rows())[0]).getByLabelText("Transcript of clip 2");
    await userEvent.clear(field);
    await userEvent.type(field, "Second one.");
    await userEvent.tab();
    expect(studioApi.editClip).toHaveBeenCalledWith("v-1", 2, "Second one.");
  });

  it("does not save when the text is unchanged", async () => {
    render(<ClipList voiceId="v-1" version={0} onChanged={() => {}} />);
    await userEvent.click(within((await rows())[0]).getByLabelText("Transcript of clip 2"));
    await userEvent.tab();
    expect(studioApi.editClip).not.toHaveBeenCalled();
  });

  it("restores the text and shows the error when saving fails", async () => {
    vi.mocked(studioApi.editClip).mockRejectedValue(new Error("A transcript can be at most 500 characters"));
    render(<ClipList voiceId="v-1" version={0} onChanged={() => {}} />);
    const field = within((await rows())[0]).getByLabelText("Transcript of clip 2");
    await userEvent.type(field, " extra");
    await userEvent.tab();
    expect(await screen.findByRole("alert")).toHaveTextContent("at most 500 characters");
    expect(field).toHaveValue("Second won.");
  });

  it("plays the chosen clip in the shared player", async () => {
    render(<ClipList voiceId="v-1" version={0} onChanged={() => {}} />);
    await userEvent.click(within((await rows())[1]).getByRole("button", { name: "Play clip 1" }));
    expect(await screen.findByLabelText("Clip player")).toHaveAttribute("src", "blob:x/1");
  });

  it("deletes a clip, removes it from the list and tells the parent", async () => {
    const onChanged = vi.fn();
    render(<ClipList voiceId="v-1" version={0} onChanged={onChanged} />);
    await userEvent.click(within((await rows())[0]).getByRole("button", { name: "Delete clip 2" }));
    expect(studioApi.deleteClip).toHaveBeenCalledWith("v-1", 2);
    await vi.waitFor(async () => expect(await rows()).toHaveLength(1));
    expect(onChanged).toHaveBeenCalledTimes(1);
  });

  it("reloads when the version changes", async () => {
    const { rerender } = render(<ClipList voiceId="v-1" version={0} onChanged={() => {}} />);
    await rows();
    rerender(<ClipList voiceId="v-1" version={1} onChanged={() => {}} />);
    await vi.waitFor(() => expect(studioApi.clips).toHaveBeenCalledTimes(2));
  });

  it("shows a load error", async () => {
    vi.mocked(studioApi.clips).mockRejectedValue(new Error("Unknown voice"));
    render(<ClipList voiceId="v-1" version={0} onChanged={() => {}} />);
    expect(await screen.findByRole("alert")).toHaveTextContent("Unknown voice");
  });

  it("releases the previous clip's object URL when another is played, and the last on unmount", async () => {
    const { unmount } = render(<ClipList voiceId="v-1" version={0} onChanged={() => {}} />);
    await userEvent.click(within((await rows())[1]).getByRole("button", { name: "Play clip 1" }));
    await screen.findByLabelText("Clip player");
    expect(URL.revokeObjectURL).not.toHaveBeenCalled();
    await userEvent.click(within((await rows())[0]).getByRole("button", { name: "Play clip 2" }));
    await vi.waitFor(() => expect(screen.getByLabelText("Clip player")).toHaveAttribute("src", "blob:x/2"));
    expect(URL.revokeObjectURL).toHaveBeenCalledTimes(1);
    expect(URL.revokeObjectURL).toHaveBeenCalledWith("blob:x/1");
    unmount();
    expect(URL.revokeObjectURL).toHaveBeenCalledTimes(2);
    expect(URL.revokeObjectURL).toHaveBeenLastCalledWith("blob:x/2");
  });

  it("shows the error when a clip cannot be fetched for playback", async () => {
    vi.mocked(studioApi.clipAudioUrl).mockRejectedValue(new Error("Origin not allowed"));
    render(<ClipList voiceId="v-1" version={0} onChanged={() => {}} />);
    await userEvent.click(within((await rows())[1]).getByRole("button", { name: "Play clip 1" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Origin not allowed");
  });
});
