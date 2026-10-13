import { useEffect, useRef, useState } from "react";
import { onFilesDropped } from "./dropEvents";
import { LibraryVoice, studioApi } from "./studioApi";

export const PACK_EXTENSION = ".voxweave";
export const isPack = (path: string) => path.toLowerCase().endsWith(PACK_EXTENSION);

const megabytes = (bytes: number) => `${Math.max(1, Math.round(bytes / (1024 * 1024)))} MB`;

/** Export of one trained voice as a single file that another computer running the app can import. */
export function VoicePackExport({ voice }: { voice: LibraryVoice }) {
  const [withClip, setWithClip] = useState(true);
  const [busy, setBusy] = useState(false);
  const [saved, setSaved] = useState<{ path: string; bytes: number; has_reference: boolean } | null>(null);
  const [error, setError] = useState<string | null>(null);

  if (!voice.models.includes("rvc")) {
    return <p className="hint">Train this voice first. A trained voice can be saved as one file and used on another computer.</p>;
  }

  const save = async () => {
    setBusy(true);
    setError(null);
    setSaved(null);
    try {
      setSaved(await studioApi.exportPack(voice.id, withClip));
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="pack">
      {error && <p role="alert" className="error">{error}</p>}
      <p className="hint">
        Saves the trained voice as one <code>{PACK_EXTENSION}</code> file in your Downloads folder. On another computer,
        drop that file onto the Voices tab. Your other recordings are not included.
      </p>
      <label className="row">
        <input type="checkbox" checked={withClip} disabled={busy} onChange={(e) => setWithClip(e.target.checked)} />
        Include one short clip, so text to speech works there too
      </label>
      <button className="secondary" disabled={busy} onClick={save}>{busy ? "Saving..." : "Export voice"}</button>
      {saved && (
        <p className="hint" aria-live="polite">
          Saved {megabytes(saved.bytes)} to <code>{saved.path}</code>
          {withClip && !saved.has_reference ? ". No clip was suitable to include, so it holds the model only." : "."}
        </p>
      )}
    </div>
  );
}

/**
 * Listens for voice pack files dropped on the window while the Voices tab is shown and imports each as a new voice.
 * Audio files dropped at the same time are left to the audio import of the selected voice.
 */
export function VoicePackDrop({ active, onImported }: { active: boolean; onImported: (voiceId: string) => void }) {
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const onImportedRef = useRef(onImported);
  onImportedRef.current = onImported;
  const working = useRef(false);

  useEffect(() => {
    if (!active) return;
    let gone = false;
    let unsubscribe: (() => void) | null = null;
    const handle = async (paths: string[]) => {
      const packs = paths.filter(isPack);
      if (packs.length === 0 || gone || working.current) return;
      working.current = true;
      setBusy(true);
      setError(null);
      setNotice(null);
      const names: string[] = [];
      const failures: string[] = [];
      let last: string | null = null;
      for (const path of packs) {
        try {
          const voice = await studioApi.importPack(path);
          names.push(voice.name);
          last = voice.id;
        } catch (e) {
          failures.push(`${path.split(/[\/]/).pop()}: ${(e as Error).message}`);
        }
      }
      working.current = false;
      if (gone) return;
      setBusy(false);
      if (names.length > 0) setNotice(`Imported ${names.length === 1 ? `the voice ${names[0]}` : `${names.length} voices: ${names.join(", ")}`}. It is ready on the Live tab.`);
      if (failures.length > 0) setError(failures.join(" "));
      if (last !== null) onImportedRef.current(last);
    };
    onFilesDropped((paths) => void handle(paths))
      .then((off) => {
        if (gone) off();
        else unsubscribe = off;
      })
      .catch(() => undefined);
    return () => {
      gone = true;
      unsubscribe?.();
    };
  }, [active]);

  return (
    <>
      {busy && <p className="hint" aria-live="polite">Importing the voice pack...</p>}
      {notice && <p className="hint" aria-live="polite">{notice}</p>}
      {error && <p role="alert" className="error">{error}</p>}
    </>
  );
}
