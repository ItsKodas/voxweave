import { RecordSession } from "./RecordSession";
import { LibraryVoice } from "./studioApi";

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

export function VoiceStudio({ voice, active, onChanged }: { voice: LibraryVoice; active: boolean; onChanged: () => void }) {
  return (
    <section className="studio" aria-label={`Voice ${voice.name}`}>
      <h2>{voice.name}</h2>
      <MinutesBar minutes={voice.minutes} />
      <h3>Record</h3>
      <RecordSession voiceId={voice.id} active={active} onSaved={onChanged} />
    </section>
  );
}
