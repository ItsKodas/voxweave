import { useEffect, useRef, useState } from "react";
import { Clip, CLIP_KINDS, ClipKind, studioApi } from "./studioApi";

const KIND_LABELS: Record<ClipKind, string> = {
  speech: "Speech", laugh: "Laugh", cough: "Cough", breath: "Breath", hum: "Hum", filler: "Um / filler", noise: "Other sound",
};

const EMPTY_TRANSCRIPT = "A transcript cannot be empty. Delete the clip if it is not usable.";

function ClipRow({ voiceId, clip, onPlay, onDeleted, onError }: {
  voiceId: string;
  clip: Clip;
  onPlay: (index: number) => void;
  onDeleted: (index: number) => void;
  onError: (message: string | null) => void;
}) {
  const [saved, setSaved] = useState(clip.text);
  const [text, setText] = useState(clip.text);
  const [kind, setKind] = useState<ClipKind>(clip.kind ?? "speech");

  const changeKind = async (next: ClipKind) => {
    const before = kind;
    setKind(next);
    try {
      await studioApi.setClipKind(voiceId, clip.index, next);
      onError(null);
    } catch (e) {
      setKind(before);
      onError((e as Error).message);
    }
  };

  const save = async () => {
    if (text.trim() === saved) return;
    if (text.trim() === "") {
      // An emptied box is almost always a slip, and a clip without words teaches the voice nothing.
      setText(saved);
      onError(EMPTY_TRANSCRIPT);
      return;
    }
    try {
      const next = await studioApi.editClip(voiceId, clip.index, text.trim());
      setSaved(next.text);
      setText(next.text);
      onError(null);
    } catch (e) {
      setText(saved);
      onError((e as Error).message);
    }
  };

  const remove = async () => {
    const ok = window.confirm(
      `Delete clip ${clip.index} (${clip.seconds.toFixed(1)} s)? Its files are moved to the data/trash folder, not erased.`,
    );
    if (!ok) return;
    try {
      await studioApi.deleteClip(voiceId, clip.index);
      onDeleted(clip.index);
    } catch (e) {
      onError((e as Error).message);
    }
  };

  return (
    <li>
      <span className="tag">
        #{clip.index} {clip.source === "imported" ? (clip.origin ?? "imported") : "recorded"} · {clip.seconds.toFixed(1)} s
      </span>
      <input aria-label={`Transcript of clip ${clip.index}`} value={text} maxLength={500}
        onChange={(e) => setText(e.target.value)} onBlur={save} />
      <select aria-label={`Kind of sound in clip ${clip.index}`} className="kind-select" value={kind}
        title="Live mode trains on every kind; text to speech only copies a voice from speech"
        onChange={(e) => changeKind(e.target.value as ClipKind)}>
        {CLIP_KINDS.map((k) => <option key={k} value={k}>{KIND_LABELS[k]}</option>)}
      </select>
      <button className="secondary" aria-label={`Play clip ${clip.index}`} onClick={() => onPlay(clip.index)}>Play</button>
      <button className="danger" aria-label={`Delete clip ${clip.index}`} onClick={remove}>Delete</button>
    </li>
  );
}

export function ClipList({ voiceId, version, onChanged }: { voiceId: string; version: number; onChanged: () => void }) {
  const [clips, setClips] = useState<Clip[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [playing, setPlaying] = useState<{ index: number; url: string } | null>(null);
  const [reload, setReload] = useState(0);
  const [labelling, setLabelling] = useState(false);
  const [labelNote, setLabelNote] = useState<string | null>(null);

  const labelSounds = async () => {
    setLabelling(true);
    setLabelNote(null);
    try {
      const res = await studioApi.labelSounds(voiceId);
      const parts = Object.entries(res.labelled).map(([kind, n]) => `${n} ${kind}`);
      setLabelNote(parts.length ? `Labelled ${parts.join(", ")}.` : "No laughs or other sounds were found among the clips marked as speech.");
      setReload((r) => r + 1);
      setError(null);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setLabelling(false);
    }
  };

  useEffect(() => {
    let gone = false;
    studioApi
      .clips(voiceId)
      .then((list) => {
        if (!gone) {
          setClips(list);
          setError(null);
        }
      })
      .catch((e: Error) => !gone && setError(e.message));
    return () => {
      gone = true;
    };
  }, [voiceId, version, reload]);

  // The player's object URL is released when another clip replaces it and when the list unmounts.
  const playingUrl = playing?.url ?? null;
  useEffect(() => {
    if (playingUrl === null) return;
    return () => URL.revokeObjectURL(playingUrl);
  }, [playingUrl]);

  const mounted = useRef(true);
  const latestPlay = useRef(0);
  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
    };
  }, []);

  // Another voice's list must not keep playing, or accept a clip that was still being fetched.
  useEffect(() => {
    return () => {
      latestPlay.current += 1;
      setPlaying(null);
    };
  }, [voiceId]);

  const play = async (index: number) => {
    const ticket = ++latestPlay.current;
    try {
      const url = await studioApi.clipAudioUrl(voiceId, index);
      if (!mounted.current || ticket !== latestPlay.current) {
        URL.revokeObjectURL(url); // superseded or too late: nothing will play it
        return;
      }
      setPlaying({ index, url });
    } catch (e) {
      if (mounted.current) setError((e as Error).message);
    }
  };

  const deleted = (index: number) => {
    setClips((list) => (list ?? []).filter((c) => c.index !== index));
    setPlaying((p) => (p?.index === index ? null : p));
    onChanged();
  };

  return (
    <section className="clips">
      {error && <p role="alert" className="error">{error}</p>}
      {playing && <audio aria-label="Clip player" controls autoPlay src={playing.url} />}
      {clips !== null && clips.length === 0 && <p className="hint">No clips yet. Record a sentence or drop in audio files.</p>}
      {clips !== null && clips.length > 0 && (
        <p className="hint">
          Each clip is labelled as speech or as a sound (laugh, cough, breath, hum, um). Live mode trains on all of them,
          so it can laugh and cough in this voice; text to speech only ever copies a voice from speech. Correct a label
          with the list on its row.{" "}
          <button className="secondary" disabled={labelling} onClick={labelSounds}>
            {labelling ? "Looking..." : "Find laughs and other sounds"}
          </button>
        </p>
      )}
      {labelNote && <p className="hint" aria-live="polite">{labelNote}</p>}
      {clips !== null && clips.length > 0 && (
        <ul aria-label="Clips" className="clip-list">
          {[...clips].sort((a, b) => b.index - a.index).map((clip) => (
            <ClipRow key={clip.index} voiceId={voiceId} clip={clip} onPlay={play} onDeleted={deleted} onError={setError} />
          ))}
        </ul>
      )}
    </section>
  );
}
