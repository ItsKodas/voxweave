import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { PageReader } from "./PageReader";

vi.mock("./studioApi", () => ({
  studioApi: { pages: vi.fn(), recordPage: vi.fn(), recordStop: vi.fn(), recordCancel: vi.fn(), status: vi.fn() },
}));
const status = vi.hoisted(() => ({ current: null as unknown }));
vi.mock("./useStudioStatus", () => ({ useStudioStatus: () => status.current }));
import { studioApi } from "./studioApi";

const PAGES = [
  { index: 0, title: "The Lighthouse Keeper", text: "For thirty-one years, Margaret kept the light.\n\nMost nights were quiet.", words: 420, minutes: 2.8 },
  { index: 1, title: "Game Night", text: "Alright, everybody, settle down!", words: 380, minutes: 2.5 },
];
const IMPORT = { active: false, finished: false, total_files: 0, done_files: 0, current: null, clips_added: 0, skipped: 0, errors: [], voice_id: null, job: 4 };
const IDLE = { recording: false, level: 0, seconds: 0, full: false, import: IMPORT };

beforeEach(() => {
  vi.resetAllMocks();
  status.current = null;
  vi.mocked(studioApi.pages).mockResolvedValue(PAGES);
  vi.mocked(studioApi.recordPage).mockResolvedValue({ ...IDLE, recording: true });
  vi.mocked(studioApi.recordCancel).mockResolvedValue(IDLE);
  vi.mocked(studioApi.recordStop).mockResolvedValue({ accepted: true, imported: true, seconds: 162.4, reason: null, message: null, clip: null });
  vi.mocked(studioApi.status).mockResolvedValue({ ...IDLE, import: { ...IMPORT, active: true, job: 5, voice_id: "v-1", total_files: 1 } });
});

async function openFirstPage() {
  await userEvent.click(await screen.findByRole("button", { name: /The Lighthouse Keeper/ }));
  return screen.findByRole("dialog", { name: "The Lighthouse Keeper" });
}

describe("PageReader", () => {
  it("offers each page with its reading time", async () => {
    render(<PageReader voiceId="v-1" input={4} active onSaved={() => {}} />);
    expect(await screen.findByRole("button", { name: /The Lighthouse Keeper.*about 3 minutes/ })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Game Night/ })).toBeInTheDocument();
  });

  it("opens the whole page full screen to read from, paragraph by paragraph", async () => {
    render(<PageReader voiceId="v-1" input={4} active onSaved={() => {}} />);
    const page = await openFirstPage();
    expect(page).toHaveTextContent("For thirty-one years, Margaret kept the light.");
    expect(page).toHaveTextContent("Most nights were quiet.");
    expect(screen.getByRole("button", { name: "Start reading" })).toBeEnabled();
    await userEvent.click(screen.getByRole("button", { name: "Close" }));
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });

  it("records the reading, then adds it to the voice as clips and says how many", async () => {
    const onSaved = vi.fn();
    const { rerender } = render(<PageReader voiceId="v-1" input={4} active onSaved={onSaved} />);
    await openFirstPage();
    await userEvent.click(screen.getByRole("button", { name: "Start reading" }));
    expect(studioApi.recordPage).toHaveBeenCalledWith("v-1", 4, 0);
    status.current = { ...IDLE, recording: true, level: 0.4, seconds: 75.2 };
    rerender(<PageReader voiceId="v-1" input={4} active onSaved={onSaved} />);
    expect(screen.getByText(/1:15/)).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Stop and add to the voice" }));
    expect(await screen.findByText(/Adding your reading to the voice/)).toBeInTheDocument();
    status.current = { ...IDLE, import: { ...IMPORT, job: 5, voice_id: "v-1", finished: true, clips_added: 23, total_files: 1, done_files: 1 } };
    rerender(<PageReader voiceId="v-1" input={4} active onSaved={onSaved} />);
    expect(await screen.findByText(/Added 23 clips from 2:42 of reading/)).toBeInTheDocument();
    expect(onSaved).toHaveBeenCalled();
  });

  it("shows why a reading was not used", async () => {
    vi.mocked(studioApi.recordStop).mockResolvedValue({
      accepted: false, reason: "too_short", message: "That was too short to be a page.", clip: null,
    });
    render(<PageReader voiceId="v-1" input={4} active onSaved={() => {}} />);
    await openFirstPage();
    await userEvent.click(screen.getByRole("button", { name: "Start reading" }));
    await userEvent.click(screen.getByRole("button", { name: "Stop and add to the voice" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("That was too short to be a page.");
    expect(screen.getByRole("button", { name: "Start reading" })).toBeEnabled();
  });

  it("Cancel throws the reading away", async () => {
    render(<PageReader voiceId="v-1" input={4} active onSaved={() => {}} />);
    await openFirstPage();
    await userEvent.click(screen.getByRole("button", { name: "Start reading" }));
    await userEvent.click(screen.getByRole("button", { name: "Cancel" }));
    expect(studioApi.recordCancel).toHaveBeenCalled();
    expect(studioApi.recordStop).not.toHaveBeenCalled();
    expect(screen.getByRole("button", { name: "Start reading" })).toBeEnabled();
  });

  it("stops the microphone when the tab is left mid-reading", async () => {
    const { rerender } = render(<PageReader voiceId="v-1" input={4} active onSaved={() => {}} />);
    await openFirstPage();
    await userEvent.click(screen.getByRole("button", { name: "Start reading" }));
    rerender(<PageReader voiceId="v-1" input={4} active={false} onSaved={() => {}} />);
    await waitFor(() => expect(studioApi.recordCancel).toHaveBeenCalled());
  });

  it("cannot start without a microphone", async () => {
    render(<PageReader voiceId="v-1" input={null} active onSaved={() => {}} />);
    await openFirstPage();
    expect(screen.getByRole("button", { name: "Start reading" })).toBeDisabled();
  });
});
