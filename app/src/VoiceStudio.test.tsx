import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { MinutesBar, VoiceStudio } from "./VoiceStudio";

vi.mock("./RecordSession", () => ({
  RecordSession: ({ onSaved, devicesVersion }: { onSaved: () => void; devicesVersion?: number }) => (
    <>
      <button onClick={onSaved}>record saved</button>
      <p>record devices {devicesVersion}</p>
    </>
  ),
}));
vi.mock("./ImportDrop", () => ({
  ImportDrop: ({ onImported }: { onImported: () => void }) => <button onClick={onImported}>import done</button>,
}));
vi.mock("./TrainPanel", () => ({
  TrainPanel: ({ voice, active, onModelReady, onStarted }: { voice: { name: string }; active: boolean; onModelReady: () => void; onStarted?: () => void }) => (
    <>
      <button onClick={onModelReady}>train {voice.name} {active ? "active" : "idle"}</button>
      <button onClick={onStarted}>started {voice.name}</button>
    </>
  ),
}));
vi.mock("./ClipList", () => ({
  ClipList: ({ version, onChanged }: { version: number; onChanged: () => void }) => (
    <button onClick={onChanged}>clips v{version}</button>
  ),
}));

const VOICE = { id: "v-1", name: "Alpha", created: "1", minutes: 12.3, clips: 80, models: [] };

describe("MinutesBar", () => {
  it("shows minutes against the usable and good marks", () => {
    render(<MinutesBar minutes={4.25} />);
    expect(screen.getByLabelText("Recorded minutes")).toHaveValue(4.25);
    expect(screen.getByText(/4\.3 min of clean audio/)).toBeInTheDocument();
    expect(screen.queryByText(/usable voice\./)).not.toBeInTheDocument();
  });

  it("says when there is enough for a usable voice and for a good one", () => {
    const { rerender } = render(<MinutesBar minutes={10} />);
    expect(screen.getByText(/Enough for a usable voice\./)).toBeInTheDocument();
    rerender(<MinutesBar minutes={45} />);
    expect(screen.getByText(/Plenty for a good voice\./)).toBeInTheDocument();
    expect(screen.getByLabelText("Recorded minutes")).toHaveValue(30);
  });
});

describe("VoiceStudio", () => {
  it("refreshes the clip list and the parent after a saved take, an import, or a clip change", async () => {
    const onChanged = vi.fn();
    render(<VoiceStudio voice={VOICE} active onChanged={onChanged} />);
    expect(screen.getByRole("heading", { name: "Alpha" })).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "record saved" }));
    expect(screen.getByRole("button", { name: "clips v1" })).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "import done" }));
    expect(screen.getByRole("button", { name: "clips v2" })).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "clips v2" }));
    expect(onChanged).toHaveBeenCalledTimes(3);
  });

  it("hands the device list version to the recording screen", () => {
    const { rerender } = render(<VoiceStudio voice={VOICE} active devicesVersion={2} onChanged={() => {}} />);
    expect(screen.getByText("record devices 2")).toBeInTheDocument();
    rerender(<VoiceStudio voice={VOICE} active devicesVersion={3} onChanged={() => {}} />);
    expect(screen.getByText("record devices 3")).toBeInTheDocument();
  });

  it("shows the training panel for this voice and passes on a finished model", async () => {
    const onModelReady = vi.fn();
    const { rerender } = render(<VoiceStudio voice={VOICE} active onChanged={() => {}} onModelReady={onModelReady} />);
    expect(screen.getByRole("heading", { name: "Train" })).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "train Alpha active" }));
    expect(onModelReady).toHaveBeenCalledTimes(1);
    rerender(<VoiceStudio voice={VOICE} active={false} onChanged={() => {}} onModelReady={onModelReady} />);
    expect(screen.getByRole("button", { name: "train Alpha idle" })).toBeInTheDocument();
  });

  it("passes on that a training run was started", async () => {
    const onTrainingStarted = vi.fn();
    const onChanged = vi.fn();
    render(<VoiceStudio voice={VOICE} active onChanged={onChanged} onTrainingStarted={onTrainingStarted} />);
    await userEvent.click(screen.getByRole("button", { name: "started Alpha" }));
    expect(onTrainingStarted).toHaveBeenCalledTimes(1);
    expect(onChanged).not.toHaveBeenCalled();
  });
});
