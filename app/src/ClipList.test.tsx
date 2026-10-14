import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { ClipList } from "./ClipList";

vi.mock("./studioApi", async (original) => ({
  ...(await original<typeof import("./studioApi")>()),
  studioApi: { clips: vi.fn(), editClip: vi.fn(), deleteClip: vi.fn(), clipAudioUrl: vi.fn(), setClipKind: vi.fn(), labelSounds: vi.fn() },
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
  vi.spyOn(window, "confirm").mockReturnValue(true);
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

  it("asks before deleting and deletes nothing when the answer is no", async () => {
    vi.mocked(window.confirm).mockReturnValue(false);
    const onChanged = vi.fn();
    render(<ClipList voiceId="v-1" version={0} onChanged={onChanged} />);
    await userEvent.click(within((await rows())[0]).getByRole("button", { name: "Delete clip 2" }));
    expect(window.confirm).toHaveBeenCalledTimes(1);
    const question = vi.mocked(window.confirm).mock.calls[0][0] as string;
    expect(question).toMatch(/clip 2/i);
    expect(question).toMatch(/moved to the data\/trash folder, not erased/);
    expect(studioApi.deleteClip).not.toHaveBeenCalled();
    expect(await rows()).toHaveLength(2);
    expect(onChanged).not.toHaveBeenCalled();
  });

  it("deletes the clip once the question is answered yes", async () => {
    render(<ClipList voiceId="v-1" version={0} onChanged={() => {}} />);
    await userEvent.click(within((await rows())[1]).getByRole("button", { name: "Delete clip 1" }));
    expect(window.confirm).toHaveBeenCalledTimes(1);
    expect(studioApi.deleteClip).toHaveBeenCalledWith("v-1", 1);
    await vi.waitFor(async () => expect(await rows()).toHaveLength(1));
  });

  it("does not save an emptied transcript: it puts the text back and says why", async () => {
    render(<ClipList voiceId="v-1" version={0} onChanged={() => {}} />);
    const field = within((await rows())[0]).getByLabelText("Transcript of clip 2");
    await userEvent.clear(field);
    await userEvent.type(field, "   ");
    await userEvent.tab();
    expect(await screen.findByRole("alert")).toHaveTextContent(
      "A transcript cannot be empty. Delete the clip if it is not usable.",
    );
    expect(field).toHaveValue("Second won.");
    expect(studioApi.editClip).not.toHaveBeenCalled();
  });

  it("clears the empty-transcript message after the next successful save", async () => {
    render(<ClipList voiceId="v-1" version={0} onChanged={() => {}} />);
    const field = within((await rows())[0]).getByLabelText("Transcript of clip 2");
    await userEvent.clear(field);
    await userEvent.tab();
    expect(await screen.findByRole("alert")).toBeInTheDocument();
    await userEvent.type(field, " Now.");
    await userEvent.tab();
    expect(studioApi.editClip).toHaveBeenCalledWith("v-1", 2, "Second won. Now.");
    await vi.waitFor(() => expect(screen.queryByRole("alert")).not.toBeInTheDocument());
  });

  it("leaves a clip whose transcript was already empty alone when nothing is typed", async () => {
    vi.mocked(studioApi.clips).mockResolvedValue([{ ...C2, text: "" }]);
    render(<ClipList voiceId="v-1" version={0} onChanged={() => {}} />);
    await userEvent.click(within((await rows())[0]).getByLabelText("Transcript of clip 2"));
    await userEvent.tab();
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
    expect(studioApi.editClip).not.toHaveBeenCalled();
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

  it("releases audio from a superseded play request, and shows only the latest clip", async () => {
    let late: (url: string) => void = () => {};
    vi.mocked(studioApi.clipAudioUrl).mockImplementation((_id, index) =>
      index === 1 ? new Promise((r) => { late = r; }) : Promise.resolve("blob:x/2"));
    render(<ClipList voiceId="v-1" version={0} onChanged={() => {}} />);
    await userEvent.click(within((await rows())[1]).getByRole("button", { name: "Play clip 1" }));
    await userEvent.click(within((await rows())[0]).getByRole("button", { name: "Play clip 2" }));
    await vi.waitFor(() => expect(screen.getByLabelText("Clip player")).toHaveAttribute("src", "blob:x/2"));
    late("blob:late-1");
    await vi.waitFor(() => expect(URL.revokeObjectURL).toHaveBeenCalledWith("blob:late-1"));
    expect(screen.getByLabelText("Clip player")).toHaveAttribute("src", "blob:x/2");
  });

  it("releases audio that arrives after the list was unmounted", async () => {
    let late: (url: string) => void = () => {};
    vi.mocked(studioApi.clipAudioUrl).mockReturnValue(new Promise((r) => { late = r; }));
    const { unmount } = render(<ClipList voiceId="v-1" version={0} onChanged={() => {}} />);
    await userEvent.click(within((await rows())[1]).getByRole("button", { name: "Play clip 1" }));
    unmount();
    late("blob:late");
    await vi.waitFor(() => expect(URL.revokeObjectURL).toHaveBeenCalledWith("blob:late"));
  });

  it("stops and releases the player when the playing clip is deleted", async () => {
    render(<ClipList voiceId="v-1" version={0} onChanged={() => {}} />);
    await userEvent.click(within((await rows())[0]).getByRole("button", { name: "Play clip 2" }));
    await screen.findByLabelText("Clip player");
    await userEvent.click(within((await rows())[1]).getByRole("button", { name: "Delete clip 1" }));
    await vi.waitFor(async () => expect(await rows()).toHaveLength(1));
    expect(screen.getByLabelText("Clip player")).toBeInTheDocument(); // another clip's delete leaves it playing
    await userEvent.click(within((await rows())[0]).getByRole("button", { name: "Delete clip 2" }));
    await vi.waitFor(() => expect(screen.queryByLabelText("Clip player")).not.toBeInTheDocument());
    expect(URL.revokeObjectURL).toHaveBeenCalledWith("blob:x/2");
  });

  it("stops and releases the player when the voice changes", async () => {
    const { rerender } = render(<ClipList voiceId="v-1" version={0} onChanged={() => {}} />);
    await userEvent.click(within((await rows())[0]).getByRole("button", { name: "Play clip 2" }));
    await screen.findByLabelText("Clip player");
    rerender(<ClipList voiceId="v-2" version={0} onChanged={() => {}} />);
    await vi.waitFor(() => expect(screen.queryByLabelText("Clip player")).not.toBeInTheDocument());
    expect(URL.revokeObjectURL).toHaveBeenCalledWith("blob:x/2");
  });
});

describe("kinds of sound", () => {
  it("shows each clip's kind and saves a correction", async () => {
    vi.mocked(studioApi.clips).mockResolvedValue([
      { index: 2, text: "[laugh] hahaha", seconds: 1.5, source: "imported", prompt: null, origin: "stream.mp3", kind: "laugh" },
      { index: 1, text: "Hello there.", seconds: 2, source: "recorded", prompt: 0, origin: null },
    ]);
    vi.mocked(studioApi.setClipKind).mockResolvedValue({} as never);
    render(<ClipList voiceId="v-1" version={0} onChanged={() => {}} />);
    const laugh = await screen.findByLabelText("Kind of sound in clip 2");
    expect(laugh).toHaveValue("laugh");
    expect(screen.getByLabelText("Kind of sound in clip 1")).toHaveValue("speech"); // clips from before kinds existed
    fireEvent.change(laugh, { target: { value: "cough" } });
    await waitFor(() => expect(studioApi.setClipKind).toHaveBeenCalledWith("v-1", 2, "cough"));
  });

  it("labels existing clips on request and reloads the list", async () => {
    vi.mocked(studioApi.clips).mockResolvedValue([
      { index: 1, text: "Hahaha", seconds: 1, source: "imported", prompt: null, origin: "a.mp3", kind: "speech" },
    ]);
    vi.mocked(studioApi.labelSounds).mockResolvedValue({ labelled: { laugh: 1 }, clips: 1 });
    render(<ClipList voiceId="v-1" version={0} onChanged={() => {}} />);
    fireEvent.click(await screen.findByRole("button", { name: "Find laughs and other sounds" }));
    expect(await screen.findByText("Labelled 1 laugh.")).toBeInTheDocument();
    await waitFor(() => expect(studioApi.clips).toHaveBeenCalledTimes(2));
  });
});
