import { useCallback, useState } from "react";
import { ClipList } from "./ClipList";
import { ImportDrop } from "./ImportDrop";
import { RecordSession } from "./RecordSession";
import { LibraryVoice } from "./studioApi";
import { TrainPanel } from "./TrainPanel";
import { VoicePackExport } from "./VoicePack";

export const USABLE_MINUTES = 10;
export const GOOD_MINUTES = 30;

export function MinutesBar({ minutes }: { minutes: number }) {
  const state = minutes >= GOOD_MINUTES ? "Plenty for a good voice." : minutes >= USABLE_MINUTES ? "Enough for a usable voice." : "";
  return (
    <div className="minutes">
      <progress max={GOOD_MINUTES} value={Math.min(minutes, GOOD_MINUTES)} aria-label="Recorded minutes" />
      <p>
        {minutes.toFixed(1)} min of clean audio. {USABLE_MINUTES} min is usable, {GOOD_MINUTES} min is good. {state}
      </p>
    </div>
  );
}

const SECTIONS = [
  { id: "record", label: "Record" },
  { id: "import", label: "Import" },
  { id: "train", label: "Train" },
  { id: "share", label: "Share" },
] as const;
type Section = (typeof SECTIONS)[number]["id"];

/**
 * One voice: its header, Record / Import / Train / Share as tabs, and its clips in a column of their own. Returns
 * two grid cells (the studio and the clips) for the Voices tab's layout. Every section stays mounted, so switching
 * between them never interrupts a take or an import.
 */
export function VoiceStudio({ voice, active, devicesVersion = 0, onChanged, onModelReady = () => {}, onTrainingStarted }: {
  voice: LibraryVoice;
  active: boolean;
  devicesVersion?: number;
  onChanged: () => void;
  /** A training run finished: the voice list and the Live tab need to reload. */
  onModelReady?: () => void;
  /** A training run was started or resumed here. */
  onTrainingStarted?: () => void;
}) {
  const [version, setVersion] = useState(0);
  const [section, setSection] = useState<Section>("record");
  const changed = useCallback(() => {
    setVersion((v) => v + 1);
    onChanged();
  }, [onChanged]);

  return (
    <>
      <section className="studio" aria-label={`Voice ${voice.name}`}>
        <div className="card studio-head">
          <span className="avatar" aria-hidden="true">{voice.name.slice(0, 1).toUpperCase()}</span>
          <div className="who">
            <h2>{voice.name}</h2>
            <MinutesBar minutes={voice.minutes} />
          </div>
          <div role="tablist" aria-label="Voice sections" className="studio-tabs">
            {SECTIONS.map((s) => (
              <button key={s.id} role="tab" aria-selected={section === s.id} onClick={() => setSection(s.id)}>{s.label}</button>
            ))}
          </div>
        </div>
        <div className="card studio-body">
          <div hidden={section !== "record"}>
            <RecordSession voiceId={voice.id} active={active} devicesVersion={devicesVersion} onSaved={changed} />
          </div>
          <div hidden={section !== "import"}>
            <ImportDrop voiceId={voice.id} active={active} onImported={changed} />
          </div>
          <div hidden={section !== "train"}>
            <TrainPanel voice={voice} active={active} onModelReady={onModelReady} onStarted={onTrainingStarted} />
          </div>
          <div hidden={section !== "share"}>
            <VoicePackExport voice={voice} />
          </div>
        </div>
      </section>
      <aside className="card clips" aria-label="Clips">
        <h3 className="card-title">Clips</h3>
        <ClipList voiceId={voice.id} version={version} onChanged={onChanged} />
      </aside>
    </>
  );
}
