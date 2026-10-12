import argparse
import os
import threading
import time
from pathlib import Path

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


def build_app():
    import onnxruntime as ort

    from aivoice.importer import Importer
    from aivoice.library import Library
    from aivoice.recorder import Recorder
    from aivoice.studio import Studio
    from aivoice.training import TrainingManager
    from aivoice.transcribe import WhisperTranscriber

    data = config.data_dir()
    streams = SoundDeviceStreams()
    manager = LiveManager(data, list_devices, default_engine_factory(data), streams,
                          refresh_devices=refresh_devices)
    library = Library(data)
    # The model name is read when an import first needs it, so a bad setting cannot stop startup.
    importer = Importer(library, WhisperTranscriber(lambda: config.whisper_model(data), data / "pretrained" / "whisper"))
    studio = Studio(library, Recorder(streams), importer, list_devices)
    training = TrainingManager(data, library, Path(__file__).resolve().parents[2] / "trainer")
    return create_app(manager, data, list_devices, ort.get_available_providers(), default_devices=default_devices,
                      studio=studio, library=library, importer=importer, training=training)


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
