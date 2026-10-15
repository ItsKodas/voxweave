import argparse
import os
import threading
import time
from pathlib import Path

import numpy as np
import psutil
import uvicorn

from aivoice import config
from aivoice.api import create_app
from aivoice.devices import SoundDeviceStreams, default_devices, list_devices, refresh_devices
from aivoice.manager import LiveManager, default_engine_factory


def _exit_when_parent_dies(pid: int) -> None:
    while psutil.pid_exists(pid):
        time.sleep(1.0)
    os._exit(0)


def unavailable_parts() -> tuple[str | None, str | None]:
    """The messages for training and for text to speech when this app has neither (the installed app), else None."""
    if not config.installed_app():
        return None, None
    from aivoice import training, tts

    return training.NOT_IN_INSTALLED_APP, tts.NOT_IN_INSTALLED_APP


def build_app():
    import onnxruntime as ort

    from aivoice.importer import Importer
    from aivoice.library import Library
    from aivoice.playback import Player
    from aivoice.recorder import Recorder
    from aivoice.studio import Studio
    from aivoice.calibration import Calibration
    from aivoice.training import TrainingManager
    from aivoice.transcribe import WhisperTranscriber
    from aivoice.tts import TtsManager

    data = config.data_dir()
    streams = SoundDeviceStreams()
    library = Library(data)
    # Its own recorder: a calibration take never shares state with a clip being recorded. 30 seconds at most.
    calibration = Calibration(data, Recorder(streams, max_seconds=30.0), list_devices, library)
    from aivoice.insight import Insight
    from aivoice.pretrained import base_voice_id
    from aivoice.speaker import voice_matcher

    cpu_session = lambda path: ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])  # noqa: E731
    # The Live tab's graphs; the voice match runs on the processor, beside the voice on the graphics card.
    insight_factory = lambda voice: Insight(config.SAMPLE_RATE, matcher_factory=lambda: voice_matcher(  # noqa: E731
        data, library, base_voice_id(voice.id), cpu_session))
    def learner_factory(voice, params, gain):
        """Live learning's tools for a voice: measuring speech, and a tuning pass on the processor (a voice engine
        of its own, so the one live mode uses on the graphics card is never shared)."""
        from aivoice.calibration import measure_speech
        from aivoice.deeptune import live_tuner
        from aivoice.engines.pitch import Rmvpe, rmvpe_path
        from aivoice.engines.rvc import RvcEngine
        from aivoice.pretrained import vec_model_path
        from aivoice.speaker import SpeakerModel, model_path, model_ready, speech_clip_paths, voice_fingerprint

        measure = lambda audio: measure_speech(audio, config.SAMPLE_RATE)  # noqa: E731
        if not model_ready(data):
            return measure, None  # without the speaker model, learning measures the voice but does not tune
        tools: dict = {}

        def tune(audio, settings, progress):
            if not tools:
                progress(0, "Getting ready")
                rmvpe = Rmvpe(cpu_session(rmvpe_path(data))) if rmvpe_path(data).is_file() else None
                engine = RvcEngine(voice.model_path, vec_model_path(data), voice.sample_rate, voice.index_path,
                                   session_factory=cpu_session, rmvpe=rmvpe)
                engine.pitch_range = calibration.voice_pitch_range(base_voice_id(voice.id))
                profile = calibration.profile()
                if profile is not None:
                    engine.speech_level_db = profile.level_db + 20.0 * float(np.log10(profile.input_gain))
                speaker = SpeakerModel(cpu_session(model_path(data)))
                target = voice_fingerprint(speaker, speech_clip_paths(library, base_voice_id(voice.id)))
                if target is None:
                    raise ValueError("this voice has no recordings to compare with")
                timing = {"chunk_ms": params.chunk_ms, "context_ms": params.context_ms,
                          "crossfade_ms": params.crossfade_ms}
                tools["tune"] = live_tuner(engine, calibration.wrap, speaker, target, transcriber, timing, gain)
            return tools["tune"](audio, settings, progress)

        return measure, tune

    manager = LiveManager(data, list_devices, default_engine_factory(data), streams,
                          refresh_devices=refresh_devices, calibration=calibration, insight_factory=insight_factory,
                          learner_factory=learner_factory)
    # The model name is read when an import first needs it, so a bad setting cannot stop startup.
    transcriber = WhisperTranscriber(lambda: config.whisper_model(data), data / "pretrained" / "whisper")
    importer = Importer(library, transcriber)
    studio = Studio(library, Recorder(streams), importer, list_devices, recordings=data / "page-recordings")
    root = Path(__file__).resolve().parents[2]
    no_training, no_tts = unavailable_parts()
    training = TrainingManager(data, library, root / "trainer", unavailable=no_training)
    # Speech is played through the same stream factory as live mode. No stream is opened and no worker is
    # started until something is spoken.
    tts = TtsManager(data, library, root / "tts", Player(streams), list_devices, unavailable=no_tts)
    from aivoice.deeptune import Tuner
    from aivoice.speaker import SpeakerModel, model_path

    # Deep calibration: the speaker model is small and runs on the processor, the voice on the graphics card.
    tuner = Tuner(data, library, calibration, default_engine_factory(data), transcriber,
                  lambda: SpeakerModel(ort.InferenceSession(str(model_path(data)), providers=["CPUExecutionProvider"])))
    return create_app(manager, data, list_devices, ort.get_available_providers(), default_devices=default_devices,
                      studio=studio, library=library, importer=importer, training=training, tts=tts,
                      calibration=calibration, tuner=tuner)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--parent-pid", type=int)
    args = parser.parse_args()
    if args.parent_pid:
        threading.Thread(target=_exit_when_parent_dies, args=(args.parent_pid,), daemon=True).start()
    uvicorn.run(build_app(), host="127.0.0.1", port=args.port, log_level="info")


if __name__ == "__main__":
    main()
