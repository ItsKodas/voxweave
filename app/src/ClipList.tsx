import { useEffect, useRef, useState } from "react";
import { Clip, studioApi } from "./studioApi";

function ClipRow({ voiceId, clip, onPlay, onDeleted, onError }: {
  voiceId: string;
  clip: Clip;
  onPlay: (index: number) => void;
  onDeleted: (index: number) => void;
  onError: (message: string | null) => void;
}) {
  const [saved, setSaved] = useState(clip.text);
  const [text, setText] = useState(clip.text);

  const save = async () => {
    if (text.trim() === saved) return;
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
      <button className="secondary" aria-label={`Play clip ${clip.index}`} onClick={() => onPlay(clip.index)}>Play</button>
      <button className="danger" aria-label={`Delete clip ${clip.index}`} onClick={remove}>Delete</button>
    </li>
  );
}

export function ClipList({ voiceId, version, onChanged }: { voiceId: string; version: number; onChanged: () => void }) {
  const [clips, setClips] = useState<Clip[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [playing, setPlaying] = useState<{ index: number; url: string } | null>(null);

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
  }, [voiceId, version]);

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
        <ul aria-label="Clips" className="clip-list">
          {[...clips].sort((a, b) => b.index - a.index).map((clip) => (
            <ClipRow key={clip.index} voiceId={voiceId} clip={clip} onPlay={play} onDeleted={deleted} onError={setError} />
          ))}
        </ul>
      )}
    </section>
  );
}
