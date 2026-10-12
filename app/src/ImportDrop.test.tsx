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
  import: { active: false, finished: false, total_files: 0, done_files: 0, current: null, clips_added: 0, skipped: 0, errors: [], ...over },
});

beforeEach(() => {
  vi.resetAllMocks();
  drop.handler = null;
  drop.unsubscribed = 0;
  status.current = null;
  vi.mocked(studioApi.importFiles).mockResolvedValue(imp({ active: true, total_files: 2 }) as never);
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

  it("reports the result and errors once, and tells the parent once", async () => {
    const onImported = vi.fn();
    const { rerender } = render(<ImportDrop voiceId="v-1" active onImported={onImported} />);
    await vi.waitFor(() => expect(drop.handler).not.toBeNull());
    await dropFiles(["C:\\a.mp3", "C:\\notes.mp3"]);
    status.current = imp({ finished: true, total_files: 2, done_files: 2, clips_added: 6, skipped: 1, errors: ["Could not read notes.mp3: invalid data"] });
    rerender(<ImportDrop voiceId="v-1" active onImported={onImported} />);
    expect(await screen.findByText(/Added 6 clips/)).toBeInTheDocument();
    expect(screen.getByText(/1 skipped as too quiet/)).toBeInTheDocument();
    expect(screen.getByText("Could not read notes.mp3: invalid data")).toBeInTheDocument();
    rerender(<ImportDrop voiceId="v-1" active onImported={onImported} />);
    expect(onImported).toHaveBeenCalledTimes(1);
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

  it("shows a running import when mounted while a job is active, and reports it once when done", async () => {
    vi.mocked(studioApi.status).mockResolvedValue(imp({ active: true, total_files: 3, done_files: 1, current: "b.mp3", clips_added: 2 }) as never);
    status.current = imp({ active: true, total_files: 3, done_files: 1, current: "b.mp3", clips_added: 2 });
    const onImported = vi.fn();
    const { rerender } = render(<ImportDrop voiceId="v-1" active onImported={onImported} />);
    expect(await screen.findByText(/Importing file 2 of 3: b\.mp3/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Cancel import" })).toBeInTheDocument();
    status.current = imp({ finished: true, total_files: 3, done_files: 3, clips_added: 5 });
    rerender(<ImportDrop voiceId="v-1" active onImported={onImported} />);
    expect(await screen.findByText(/Added 5 clips/)).toBeInTheDocument();
    expect(onImported).toHaveBeenCalledTimes(1);
  });

  it("does not adopt an old finished job when mounted with nothing running", async () => {
    vi.mocked(studioApi.status).mockResolvedValue(imp({ finished: true, total_files: 1, done_files: 1, clips_added: 9 }) as never);
    status.current = imp({ finished: true, total_files: 1, done_files: 1, clips_added: 9 });
    const onImported = vi.fn();
    render(<ImportDrop voiceId="v-1" active onImported={onImported} />);
    await act(async () => {});
    expect(screen.getByText(/drop audio files/i)).toBeInTheDocument();
    expect(screen.queryByText(/Added 9 clips/)).not.toBeInTheDocument();
    expect(onImported).not.toHaveBeenCalled();
  });
});
