import argparse
import os
import threading
import time

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

    data = config.data_dir()
    manager = LiveManager(data, list_devices, default_engine_factory(data), SoundDeviceStreams(),
                          refresh_devices=refresh_devices)
    return create_app(manager, data, list_devices, ort.get_available_providers(), default_devices=default_devices)


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
