import { act, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ImportDrop } from "./ImportDrop";

// Unlike ImportDrop.test.tsx, the real status poller runs here (on fake timers), so these tests count requests.
const drop = vi.hoisted(() => ({ handler: null as null | ((paths: string[]) => void) }));
vi.mock("./dropEvents", () => ({
  onFilesDropped: vi.fn(async (handler: (paths: string[]) => void) => {
    drop.handler = handler;
    return () => { drop.handler = null; };
  }),
}));
vi.mock("./studioApi", () => ({ studioApi: { importFiles: vi.fn(), importCancel: vi.fn(), status: vi.fn() } }));
import { studioApi } from "./studioApi";

const imp = (over: object) => ({
  recording: false, level: 0, seconds: 0, full: false,
  import: {
    active: false, finished: false, total_files: 0, done_files: 0, current: null, clips_added: 0, skipped: 0,
    errors: [], voice_id: "v-1", job: 0, ...over,
  },
});
const IDLE = imp({ voice_id: null });
const RUNNING = imp({ active: true, total_files: 1, current: "a.mp3", job: 1 });
const DONE = imp({ finished: true, total_files: 1, done_files: 1, clips_added: 4, job: 1 });

const tick = (ms: number) => act(async () => { await vi.advanceTimersByTimeAsync(ms); });
const dropFiles = (paths: string[]) => act(async () => { drop.handler?.(paths); });
const polls = () => vi.mocked(studioApi.status).mock.calls.length;

beforeEach(() => {
  vi.useFakeTimers();
  vi.resetAllMocks();
  drop.handler = null;
  vi.mocked(studioApi.status).mockResolvedValue(IDLE as never);
  vi.mocked(studioApi.importFiles).mockResolvedValue(RUNNING as never);
});
afterEach(() => vi.useRealTimers());

describe("ImportDrop status polling", () => {
  it("does not poll while nothing is being imported", async () => {
    render(<ImportDrop voiceId="v-1" active onImported={() => {}} />);
    await tick(0);
    expect(polls()).toBe(1); // the one look at mount
    await tick(5000);
    expect(polls()).toBe(1);
  });

  it("stops asking for the status once the import is finished, and keeps the result on screen", async () => {
    const onImported = vi.fn();
    render(<ImportDrop voiceId="v-1" active onImported={onImported} />);
    await tick(0);
    vi.mocked(studioApi.status).mockResolvedValue(RUNNING as never);
    await dropFiles(["C:\\a.mp3"]);
    await tick(650);
    expect(screen.getByText(/Importing file 1 of 1: a\.mp3/)).toBeInTheDocument();
    const whileRunning = polls();
    expect(whileRunning).toBeGreaterThanOrEqual(3);

    vi.mocked(studioApi.status).mockResolvedValue(DONE as never);
    await tick(400);
    expect(screen.getByText(/Added 4 clips/)).toBeInTheDocument();
    expect(onImported).toHaveBeenCalledTimes(1);
    const whenDone = polls();
    await tick(10_000);
    expect(polls()).toBe(whenDone); // not one more request
    expect(screen.getByText(/Added 4 clips/)).toBeInTheDocument();
    expect(onImported).toHaveBeenCalledTimes(1);
  });

  it("polls again for the next drop, and shows that import's own result", async () => {
    const onImported = vi.fn();
    render(<ImportDrop voiceId="v-1" active onImported={onImported} />);
    await tick(0);
    vi.mocked(studioApi.status).mockResolvedValue(DONE as never);
    await dropFiles(["C:\\a.mp3"]);
    await tick(400);
    expect(screen.getByText(/Added 4 clips/)).toBeInTheDocument();
    const afterFirst = polls();

    const running2 = imp({ active: true, total_files: 1, current: "b.mp3", job: 2 });
    vi.mocked(studioApi.importFiles).mockResolvedValue(running2 as never);
    vi.mocked(studioApi.status).mockResolvedValue(running2 as never);
    await dropFiles(["C:\\b.mp3"]);
    expect(screen.queryByText(/Added 4 clips/)).not.toBeInTheDocument();
    await tick(650);
    expect(polls()).toBeGreaterThanOrEqual(afterFirst + 3);
    expect(screen.getByText(/Importing file 1 of 1: b\.mp3/)).toBeInTheDocument();

    vi.mocked(studioApi.status).mockResolvedValue(imp({ finished: true, total_files: 1, done_files: 1, clips_added: 9, job: 2 }) as never);
    await tick(400);
    expect(screen.getByText(/Added 9 clips/)).toBeInTheDocument();
    expect(onImported).toHaveBeenCalledTimes(2);
    const afterSecond = polls();
    await tick(5000);
    expect(polls()).toBe(afterSecond);
  });

  it("stops polling when another voice's import ends, without showing its result", async () => {
    const theirs = imp({ active: true, total_files: 1, current: "x.mp3", voice_id: "v-2", job: 7 });
    vi.mocked(studioApi.status).mockResolvedValue(theirs as never);
    const onImported = vi.fn();
    render(<ImportDrop voiceId="v-1" active onImported={onImported} />);
    await tick(450);
    expect(screen.getByText(/another voice is running/)).toBeInTheDocument();
    vi.mocked(studioApi.status).mockResolvedValue(imp({ finished: true, clips_added: 3, voice_id: "v-2", job: 7 }) as never);
    await tick(400);
    expect(screen.queryByText(/another voice is running/)).not.toBeInTheDocument();
    const settled = polls();
    await tick(5000);
    expect(polls()).toBe(settled);
    expect(screen.queryByText(/Added 3 clips/)).not.toBeInTheDocument();
    expect(onImported).not.toHaveBeenCalled();
  });

  it("keeps the finished result on screen when the tab is hidden and shown again", async () => {
    const ui = (active: boolean) => <ImportDrop voiceId="v-1" active={active} onImported={() => {}} />;
    const { rerender } = render(ui(true));
    await tick(0);
    vi.mocked(studioApi.status).mockResolvedValue(DONE as never);
    await dropFiles(["C:\\a.mp3"]);
    await tick(400);
    expect(screen.getByText(/Added 4 clips/)).toBeInTheDocument();
    rerender(ui(false));
    await tick(1000);
    rerender(ui(true));
    await tick(1000);
    expect(screen.getByText(/Added 4 clips/)).toBeInTheDocument();
    const settled = polls();
    await tick(5000);
    expect(polls()).toBe(settled);
  });
});
