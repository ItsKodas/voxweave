import argparse
import os
import threading
import time

import psutil
import uvicorn

from aivoice.api import create_app


def _exit_when_parent_dies(pid: int) -> None:
    while psutil.pid_exists(pid):
        time.sleep(1.0)
    os._exit(0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--parent-pid", type=int)
    args = parser.parse_args()
    if args.parent_pid:
        threading.Thread(target=_exit_when_parent_dies, args=(args.parent_pid,), daemon=True).start()
    uvicorn.run(create_app(), host="127.0.0.1", port=args.port, log_level="info")


if __name__ == "__main__":
    main()
