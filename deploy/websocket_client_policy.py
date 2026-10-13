import logging
import time
from typing import Dict, Optional, Tuple
from urllib.parse import urlsplit, urlunsplit

import websockets.sync.client
from typing_extensions import override

from .msgpack_numpy import Packer, unpackb


class WebsocketClientPolicy:
    """Implements the Policy interface by communicating with a server over websocket.

    See WebsocketPolicyServer for a corresponding server implementation.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: Optional[int] = 8006,
        api_key: Optional[str] = None,
        ping_timeout: Optional[float] = 120,
    ) -> None:
        self._uri = _build_websocket_uri(host, port)
        self._packer = Packer()
        self._api_key = api_key
        self._ping_timeout = ping_timeout
        self._ws, self._server_metadata = self._wait_for_server()

    def get_server_metadata(self) -> Dict:
        return self._server_metadata

    def _wait_for_server(self) -> Tuple[websockets.sync.client.ClientConnection, Dict]:
        logging.info(f"Waiting for server at {self._uri}...")
        while True:
            try:
                headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
                conn = websockets.sync.client.connect(
                    self._uri,
                    compression=None,
                    max_size=None,
                    additional_headers=headers,
                    ping_timeout=self._ping_timeout,
                )
                metadata = unpackb(conn.recv())
                return conn, metadata
            except ConnectionRefusedError:
                logging.info("Still waiting for server...")
                time.sleep(5)

    @override
    def infer(self, obs: Dict) -> Dict:  # noqa: UP006
        data = self._packer.pack(obs)
        self._ws.send(data)
        response = self._ws.recv()
        if isinstance(response, str):
            # we're expecting bytes; if the server sends a string, it's an error.
            raise RuntimeError(f"Error in inference server:\n{response}")
        return unpackb(response)

    @override
    def reset(self, robo_name: str) -> None:
        self.infer(dict(reset=True, robo_name=robo_name))

    def close(self) -> None:
        self._ws.close()


def _build_websocket_uri(host: str, port: Optional[int]) -> str:
    """Build a WebSocket URI from a hostname or a ws(s) URL."""
    if "://" not in host:
        return f"ws://{host}" + (f":{port}" if port is not None else "")

    parsed = urlsplit(host)
    if parsed.scheme not in {"ws", "wss"} or not parsed.netloc:
        raise ValueError("WebSocket host URL must start with ws:// or wss:// and include a hostname")
    # A complete URL may include its own port and path. Otherwise, use the
    # separate port argument while preserving the path (for example /infer).
    netloc = parsed.netloc
    if parsed.port is None and port is not None:
        netloc = f"{netloc}:{port}"
    return urlunsplit((parsed.scheme, netloc, parsed.path, parsed.query, parsed.fragment))
