import { useCallback, useEffect, useState } from "react";
import { LibraryVoice, studioApi } from "./studioApi";
import { VoiceStudio } from "./VoiceStudio";

export function VoicesPanel({ active, devicesVersion = 0, onModelReady }: {
  active: boolean;
  devicesVersion?: number;
  /** A training run finished (the Live tab's voice list must reload). */
  onModelReady?: () => void;
}) {
  const [voices, setVoices] = useState<LibraryVoice[] | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [name, setName] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [shown, setShown] = useState(active);

  useEffect(() => {
    if (active) setShown(true);
  }, [active]);

  const load = useCallback(async (select?: string) => {
    try {
      const list = await studioApi.voices();
      setVoices(list);
      setError(null);
      setSelected((current) => {
        const wanted = select ?? current;
        return list.some((v) => v.id === wanted) ? wanted! : (list[0]?.id ?? null);
      });
    } catch (e) {
      setError((e as Error).message);
    }
  }, []);

  useEffect(() => {
    if (shown) load();
  }, [shown, load]);

  const create = async () => {
    setBusy(true);
    try {
      const voice = await studioApi.createVoice(name.trim());
      setName("");
      await load(voice.id);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const remove = async (voice: LibraryVoice) => {
    const ok = window.confirm(
      `Delete the voice "${voice.name}" and its ${voice.clips} clips? It is moved to the data/trash folder, not erased.`,
    );
    if (!ok) return;
    try {
      await studioApi.deleteVoice(voice.id);
      await load();
    } catch (e) {
      setError((e as Error).message);
    }
  };

  const modelReady = useCallback(() => {
    void load();
    onModelReady?.();
  }, [load, onModelReady]);

  const current = voices?.find((v) => v.id === selected) ?? null;

  return (
    <section className="voices">
      {error && (
        <p role="alert" className="error">
          {error} {voices === null && <button onClick={() => load()}>Retry</button>}
        </p>
      )}
      <form
        className="row"
        onSubmit={(e) => {
          e.preventDefault();
          create();
        }}
      >
        <label>
          New voice name
          <input value={name} maxLength={60} onChange={(e) => setName(e.target.value)} />
        </label>
        <button type="submit" disabled={busy || name.trim() === ""}>Create voice</button>
      </form>

      {voices !== null && voices.length === 0 && <p className="hint">Create a voice to start recording.</p>}
      {voices !== null && voices.length > 0 && (
        <ul aria-label="Voices" className="voice-list">
          {voices.map((v) => (
            <li key={v.id} className={v.id === selected ? "selected" : ""}>
              <button className="link" aria-pressed={v.id === selected} onClick={() => setSelected(v.id)}>
                <strong>{v.name}</strong> <span>{v.minutes.toFixed(1)} min, {v.clips} clips</span>
              </button>
              <button className="danger" aria-label={`Delete voice ${v.name}`} onClick={() => remove(v)}>Delete</button>
            </li>
          ))}
        </ul>
      )}
      {current && <VoiceStudio key={current.id} voice={current} active={active} devicesVersion={devicesVersion} onChanged={() => load()} onModelReady={modelReady} />}
    </section>
  );
}
