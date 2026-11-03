import socket
import sys
import threading
import time
import webbrowser

import uvicorn
from pydantic import ValidationError

from papertrace.settings import Settings

_HOST = "127.0.0.1"  # local only: ChatGPT plan usage is for locally hosted apps
_PORT = 8000


def main() -> int:
    try:
        Settings()
    except ValidationError as exc:
        fields = ", ".join(str(e["loc"][0]).upper() for e in exc.errors())
        print(f"error: invalid configuration, check: {fields}", file=sys.stderr)
        return 1
    port = _free_port()
    url = f"http://{_HOST}:{port}/"
    print(f"PaperTrace is running at {url} (Ctrl+C to stop)")
    threading.Thread(target=_open_when_ready, args=(url, port), daemon=True).start()
    uvicorn.run("papertrace.api.app:app", host=_HOST, port=port)
    return 0


def _free_port() -> int:
    """The usual port, or any free one: the ChatGPT callback follows the port in use."""
    with socket.socket() as probe:
        try:
            probe.bind((_HOST, _PORT))
        except OSError:
            probe.bind((_HOST, 0))
        return int(probe.getsockname()[1])


def _open_when_ready(url: str, port: int) -> None:
    # Wait for the server to listen, so the first page load does not fail.
    for _ in range(50):
        try:
            socket.create_connection((_HOST, port), timeout=0.2).close()
            break
        except OSError:
            time.sleep(0.2)
    webbrowser.open(url)


if __name__ == "__main__":
    raise SystemExit(main())
