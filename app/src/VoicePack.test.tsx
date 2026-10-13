import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { onFilesDropped } from "./dropEvents";
import { LibraryVoice, studioApi } from "./studioApi";
import { isPack, VoicePackDrop, VoicePackExport } from "./VoicePack";

vi.mock("./studioApi", () => ({ studioApi: { exportPack: vi.fn(), importPack: vi.fn() } }));
vi.mock("./dropEvents", () => ({ onFilesDropped: vi.fn() }));

const TRAINED: LibraryVoice = { id: "ada-1", name: "Ada", created: "", minutes: 5, clips: 40, models: ["rvc"] };
const UNTRAINED: LibraryVoice = { ...TRAINED, id: "new-1", name: "New", models: [] };

let drop: (paths: string[]) => void = () => {};
const off = vi.fn();

beforeEach(() => {
  vi.resetAllMocks();
  vi.mocked(onFilesDropped).mockImplementation(async (handler) => {
    drop = handler;
    return off;
  });
});

describe("isPack", () => {
  it("knows a voice pack by its extension, whatever the case", () => {
    expect(isPack("C:\Users\me\Downloads\ada.voxweave")).toBe(true);
    expect(isPack("D:/x/ADA.VOXWEAVE")).toBe(true);
    expect(isPack("C:/x/ada.voxweave.mp3")).toBe(false);
    expect(isPack("C:/x/song.wav")).toBe(false);
  });
});

describe("VoicePackExport", () => {
  it("asks for training first when the voice has no model", () => {
    render(<VoicePackExport voice={UNTRAINED} />);
    expect(screen.getByText(/Train this voice first/)).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Export voice" })).not.toBeInTheDocument();
  });

  it("exports with the clip by default and says where the file went", async () => {
    vi.mocked(studioApi.exportPack).mockResolvedValue({ path: "C:\Users\me\Downloads\ada.voxweave", bytes: 225_000_000, has_reference: true });
    render(<VoicePackExport voice={TRAINED} />);
    fireEvent.click(screen.getByRole("button", { name: "Export voice" }));
    expect(studioApi.exportPack).toHaveBeenCalledWith("ada-1", true);
    expect(await screen.findByText(/Saved 215 MB to/)).toHaveTextContent("C:\Users\me\Downloads\ada.voxweave.");
  });

  it("can leave the clip out, and says when no clip was suitable", async () => {
    vi.mocked(studioApi.exportPack).mockResolvedValue({ path: "C:/d/ada.voxweave", bytes: 1000, has_reference: false });
    render(<VoicePackExport voice={TRAINED} />);
    fireEvent.click(screen.getByRole("checkbox", { name: /Include one short clip/ }));
    fireEvent.click(screen.getByRole("button", { name: "Export voice" }));
    expect(studioApi.exportPack).toHaveBeenCalledWith("ada-1", false);
    await screen.findByText(/Saved 1 MB/);
    expect(screen.queryByText(/No clip was suitable/)).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("checkbox", { name: /Include one short clip/ }));
    fireEvent.click(screen.getByRole("button", { name: "Export voice" }));
    expect(await screen.findByText(/No clip was suitable to include/)).toBeInTheDocument();
  });

  it("shows why an export was refused", async () => {
    vi.mocked(studioApi.exportPack).mockRejectedValue(new Error("Training is running. Wait for it to finish."));
    render(<VoicePackExport voice={TRAINED} />);
    fireEvent.click(screen.getByRole("button", { name: "Export voice" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Training is running");
    expect(screen.getByRole("button", { name: "Export voice" })).toBeEnabled();
  });
});

describe("VoicePackDrop", () => {
  it("imports a dropped voice pack as a new voice and leaves audio files alone", async () => {
    vi.mocked(studioApi.importPack).mockResolvedValue({ ...TRAINED, id: "ada-9" });
    const onImported = vi.fn();
    render(<VoicePackDrop active onImported={onImported} />);
    await waitFor(() => expect(onFilesDropped).toHaveBeenCalled());
    await act(async () => drop(["C:/d/song.wav", "C:/d/ada.voxweave"]));
    expect(studioApi.importPack).toHaveBeenCalledTimes(1);
    expect(studioApi.importPack).toHaveBeenCalledWith("C:/d/ada.voxweave");
    expect(await screen.findByText(/Imported the voice Ada. It is ready on the Live tab/)).toBeInTheDocument();
    expect(onImported).toHaveBeenCalledWith("ada-9");
  });

  it("does nothing for a drop without a voice pack", async () => {
    render(<VoicePackDrop active onImported={() => {}} />);
    await waitFor(() => expect(onFilesDropped).toHaveBeenCalled());
    await act(async () => drop(["C:/d/song.wav"]));
    expect(studioApi.importPack).not.toHaveBeenCalled();
  });

  it("names the file and the reason when a pack cannot be imported, and still imports the others", async () => {
    vi.mocked(studioApi.importPack)
      .mockRejectedValueOnce(new Error("This file is not a voice pack this app can read"))
      .mockResolvedValueOnce({ ...TRAINED, id: "ada-9" });
    const onImported = vi.fn();
    render(<VoicePackDrop active onImported={onImported} />);
    await waitFor(() => expect(onFilesDropped).toHaveBeenCalled());
    await act(async () => drop(["C:\\d\\broken.voxweave", "C:/d/ada.voxweave"]));
    expect(await screen.findByRole("alert")).toHaveTextContent("broken.voxweave: This file is not a voice pack this app can read");
    expect(screen.getByText(/Imported the voice Ada/)).toBeInTheDocument();
    expect(onImported).toHaveBeenCalledTimes(1);
  });

  it("does not listen while the tab is hidden, and stops listening when it goes away", async () => {
    const { rerender, unmount } = render(<VoicePackDrop active={false} onImported={() => {}} />);
    expect(onFilesDropped).not.toHaveBeenCalled();
    rerender(<VoicePackDrop active onImported={() => {}} />);
    await waitFor(() => expect(onFilesDropped).toHaveBeenCalledTimes(1));
    await act(async () => {});
    unmount();
    expect(off).toHaveBeenCalledTimes(1);
  });
});
