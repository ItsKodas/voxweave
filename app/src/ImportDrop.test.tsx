import { act, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { ImportDrop } from "./ImportDrop";

const drop = vi.hoisted(() => ({ handler: null as null | ((paths: string[]) => void), unsubscribed: 0 }));
vi.mock("./dropEvents", () => ({
  onFilesDropped: vi.fn(async (handler: (paths: string[]) => void) => {
    drop.handler = handler;
    return () => { drop.unsubscribed += 1; drop.handler = null; };
  }),
}));
vi.mock("./studioApi", () => ({ studioApi: { importFiles: vi.fn(), importCancel: vi.fn(), status: vi.fn() } }));
const status = vi.hoisted(() => ({ current: null as unknown }));
vi.mock("./useStudioStatus", () => ({ useStudioStatus: () => status.current }));
import { studioApi } from "./studioApi";

const imp = (over: object) => ({
  recording: false, level: 0, seconds: 0, full: false,
  import: {
    active: false, finished: false, total_files: 0, done_files: 0, current: null, clips_added: 0, skipped: 0,
    errors: [], voice_id: "v-1", job: 1, ...over,
  },
});

beforeEach(() => {
  vi.resetAllMocks();
  drop.handler = null;
  drop.unsubscribed = 0;
  status.current = null;
  vi.mocked(studioApi.importFiles).mockResolvedValue(imp({ active: true, total_files: 2, job: 1 }) as never);
  vi.mocked(studioApi.importCancel).mockResolvedValue(imp({}) as never);
  vi.mocked(studioApi.status).mockResolvedValue(imp({}) as never);
});

const dropFiles = (paths: string[]) => act(async () => { drop.handler?.(paths); });

describe("ImportDrop", () => {
  it("explains what to do and what the first import downloads", async () => {
    render(<ImportDrop voiceId="v-1" active onImported={() => {}} />);
    expect(await screen.findByText(/drop audio files/i)).toBeInTheDocument();
    expect(screen.getByText(/500 MB/)).toBeInTheDocument();
    expect(screen.getByText(/English/)).toBeInTheDocument();
  });

  it("starts an import for this voice when files are dropped", async () => {
    render(<ImportDrop voiceId="v-1" active onImported={() => {}} />);
    await vi.waitFor(() => expect(drop.handler).not.toBeNull());
    await dropFiles(["C:\\a.mp3", "C:\\b.wav"]);
    expect(studioApi.importFiles).toHaveBeenCalledWith("v-1", ["C:\\a.mp3", "C:\\b.wav"]);
  });

  it("shows progress and a cancel button while importing", async () => {
    const { rerender } = render(<ImportDrop voiceId="v-1" active onImported={() => {}} />);
    await vi.waitFor(() => expect(drop.handler).not.toBeNull());
    await dropFiles(["C:\\a.mp3", "C:\\b.wav"]);
    status.current = imp({ active: true, total_files: 2, done_files: 1, current: "b.wav", clips_added: 4 });
    rerender(<ImportDrop voiceId="v-1" active onImported={() => {}} />);
    expect(screen.getByText(/Importing file 2 of 2: b\.wav/)).toBeInTheDocument();
    expect(screen.getByText(/4 clips so far/)).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Cancel import" }));
    expect(studioApi.importCancel).toHaveBeenCalled();
  });

  it("reports the result and errors once, and tells the parent once even when its callback changes", async () => {
    const onImported = vi.fn();
    const { rerender } = render(<ImportDrop voiceId="v-1" active onImported={() => onImported()} />);
    await vi.waitFor(() => expect(drop.handler).not.toBeNull());
    await dropFiles(["C:\\a.mp3", "C:\\notes.mp3"]);
    status.current = imp({ finished: true, total_files: 2, done_files: 2, clips_added: 6, skipped: 1, errors: ["Could not read notes.mp3: invalid data"] });
    rerender(<ImportDrop voiceId="v-1" active onImported={() => onImported()} />);
    expect(await screen.findByText(/Added 6 clips/)).toBeInTheDocument();
    expect(screen.getByText(/1 skipped as too quiet/)).toBeInTheDocument();
    expect(screen.getByText(/Added 6 clips/)).toHaveTextContent(
      "Added 6 clips. 1 skipped as too quiet. Quiet recordings can be made louder in an audio editor and imported again.",
    );
    expect(screen.getByText("Could not read notes.mp3: invalid data")).toBeInTheDocument();
    rerender(<ImportDrop voiceId="v-1" active onImported={() => onImported()} />);
    rerender(<ImportDrop voiceId="v-1" active onImported={() => onImported()} />);
    expect(onImported).toHaveBeenCalledTimes(1);
  });

  it("gives no hint about quiet recordings when nothing was skipped", async () => {
    const { rerender } = render(<ImportDrop voiceId="v-1" active onImported={() => {}} />);
    await vi.waitFor(() => expect(drop.handler).not.toBeNull());
    await dropFiles(["C:\\a.mp3"]);
    status.current = imp({ finished: true, total_files: 1, done_files: 1, clips_added: 6, skipped: 0 });
    rerender(<ImportDrop voiceId="v-1" active onImported={() => {}} />);
    expect(await screen.findByText(/Added 6 clips/)).toHaveTextContent(/^Added 6 clips\.$/);
    expect(screen.queryByText(/audio editor/)).not.toBeInTheDocument();
  });

  it("ignores a drop while an import is running and says so", async () => {
    const { rerender } = render(<ImportDrop voiceId="v-1" active onImported={() => {}} />);
    await vi.waitFor(() => expect(drop.handler).not.toBeNull());
    await dropFiles(["C:\\a.mp3"]);
    status.current = imp({ active: true, total_files: 1, current: "a.mp3" });
    rerender(<ImportDrop voiceId="v-1" active onImported={() => {}} />);
    await dropFiles(["C:\\b.mp3"]);
    expect(studioApi.importFiles).toHaveBeenCalledTimes(1);
    expect(screen.getByText(/wait for the current import/i)).toBeInTheDocument();
  });

  it("shows the backend's message when the import cannot start", async () => {
    vi.mocked(studioApi.importFiles).mockRejectedValue(new Error("An import is already running"));
    render(<ImportDrop voiceId="v-1" active onImported={() => {}} />);
    await vi.waitFor(() => expect(drop.handler).not.toBeNull());
    await dropFiles(["C:\\a.mp3"]);
    expect(await screen.findByRole("alert")).toHaveTextContent("An import is already running");
  });

  it("does not listen for drops while its tab is hidden, and unsubscribes on unmount", async () => {
    const { rerender, unmount } = render(<ImportDrop voiceId="v-1" active={false} onImported={() => {}} />);
    await act(async () => {});
    expect(drop.handler).toBeNull();
    rerender(<ImportDrop voiceId="v-1" active onImported={() => {}} />);
    await vi.waitFor(() => expect(drop.handler).not.toBeNull());
    unmount();
    expect(drop.unsubscribed).toBe(1);
  });

  it("shows my own import that is already running when mounted, and reports it once when done", async () => {
    const running = imp({ active: true, total_files: 3, done_files: 1, current: "b.mp3", clips_added: 2, job: 4 });
    vi.mocked(studioApi.status).mockResolvedValue(running as never);
    status.current = running;
    const onImported = vi.fn();
    const { rerender } = render(<ImportDrop voiceId="v-1" active onImported={() => onImported()} />);
    expect(await screen.findByText(/Importing file 2 of 3: b\.mp3/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Cancel import" })).toBeInTheDocument();
    status.current = imp({ finished: true, total_files: 3, done_files: 3, clips_added: 5, job: 4 });
    rerender(<ImportDrop voiceId="v-1" active onImported={() => onImported()} />);
    expect(await screen.findByText(/Added 5 clips/)).toBeInTheDocument();
    rerender(<ImportDrop voiceId="v-1" active onImported={() => onImported()} />);
    expect(onImported).toHaveBeenCalledTimes(1);
  });

  it("only shows a neutral notice for another voice's import, refuses drops, and never reports it", async () => {
    const theirs = imp({ active: true, total_files: 3, done_files: 1, current: "secret.mp3", clips_added: 2, voice_id: "v-2", job: 4 });
    vi.mocked(studioApi.status).mockResolvedValue(theirs as never);
    status.current = theirs;
    const onImported = vi.fn();
    const { rerender } = render(<ImportDrop voiceId="v-1" active onImported={() => onImported()} />);
    expect(await screen.findByText("An import into another voice is running. Wait for it to finish before importing here.")).toBeInTheDocument();
    expect(screen.queryByText(/secret\.mp3/)).not.toBeInTheDocument();
    expect(screen.queryByText(/clips so far/)).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Cancel import" })).not.toBeInTheDocument();
    await vi.waitFor(() => expect(drop.handler).not.toBeNull());
    await dropFiles(["C:\\a.mp3"]);
    expect(studioApi.importFiles).not.toHaveBeenCalled();
    expect(screen.getAllByText(/another voice is running/)).toHaveLength(1);
    status.current = imp({ finished: true, total_files: 3, done_files: 3, clips_added: 5, voice_id: "v-2", job: 4 });
    rerender(<ImportDrop voiceId="v-1" active onImported={() => onImported()} />);
    await act(async () => {});
    expect(screen.queryByText(/another voice is running/)).not.toBeInTheDocument();
    expect(screen.queryByText(/Added 5 clips/)).not.toBeInTheDocument();
    expect(onImported).not.toHaveBeenCalled();
  });

  it("does not announce a job that was already finished when mounted", async () => {
    const old = imp({ finished: true, total_files: 1, done_files: 1, clips_added: 9, job: 3 });
    vi.mocked(studioApi.status).mockResolvedValue(old as never);
    status.current = old;
    const onImported = vi.fn();
    render(<ImportDrop voiceId="v-1" active onImported={onImported} />);
    await act(async () => {});
    expect(screen.getByText(/drop audio files/i)).toBeInTheDocument();
    expect(screen.queryByText(/Added 9 clips/)).not.toBeInTheDocument();
    expect(onImported).not.toHaveBeenCalled();
  });

  it("a second import in the same mount does not report, or show, the previous job's result", async () => {
    const onImported = vi.fn();
    const ui = () => <ImportDrop voiceId="v-1" active onImported={() => onImported()} />;
    const { rerender } = render(ui());
    await vi.waitFor(() => expect(drop.handler).not.toBeNull());
    await dropFiles(["C:\\a.mp3"]);
    status.current = imp({ finished: true, total_files: 1, done_files: 1, clips_added: 3, job: 1 });
    rerender(ui());
    expect(await screen.findByText(/Added 3 clips/)).toBeInTheDocument();
    expect(onImported).toHaveBeenCalledTimes(1);

    vi.mocked(studioApi.importFiles).mockResolvedValue(imp({ active: true, total_files: 1, job: 2 }) as never);
    await dropFiles(["C:\\b.mp3"]);  // the poll still returns job 1's finished status
    rerender(ui());
    rerender(ui());
    expect(studioApi.importFiles).toHaveBeenCalledTimes(2);
    expect(screen.queryByText(/Added 3 clips/)).not.toBeInTheDocument();
    expect(onImported).toHaveBeenCalledTimes(1);

    status.current = imp({ finished: true, total_files: 1, done_files: 1, clips_added: 7, job: 2 });
    rerender(ui());
    expect(await screen.findByText(/Added 7 clips/)).toBeInTheDocument();
    rerender(ui());
    expect(onImported).toHaveBeenCalledTimes(2);
  });

  it("ignores a drop that arrives after the drop listener was torn down", async () => {
    const { rerender } = render(<ImportDrop voiceId="v-1" active onImported={() => {}} />);
    await vi.waitFor(() => expect(drop.handler).not.toBeNull());
    const stale = drop.handler!;
    rerender(<ImportDrop voiceId="v-1" active={false} onImported={() => {}} />);
    await act(async () => { stale(["C:\\a.mp3"]); });
    expect(studioApi.importFiles).not.toHaveBeenCalled();
  });

  it("shows an error when the drop listener cannot be set up", async () => {
    const { onFilesDropped } = await import("./dropEvents");
    vi.mocked(onFilesDropped).mockRejectedValueOnce(new Error("Drag and drop is unavailable"));
    render(<ImportDrop voiceId="v-1" active onImported={() => {}} />);
    expect(await screen.findByRole("alert")).toHaveTextContent("Drag and drop is unavailable");
  });
});
