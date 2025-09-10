import time
import enum
from typing import Dict, Optional, Tuple

from typing_extensions import override
from loguru import logger
import websockets.sync.client

from vlarl_infra.client import base_agent as _base_agent
from vlarl_infra.client import msgpack_numpy

class MessageType(enum.Enum):
    INFER = "infer"
    FEEDBACK = "feedback"
    def __str__(self):
        return self.value

class WebSocketWorkerAgent(_base_agent.BaseAgent):
    def __init__(self, host: str = "0.0.0.0", port: Optional[int] = None, api_key: Optional[str] = None) -> None:
        self._uri = f"ws://{host}"
        if port is not None:
            self._uri += f":{port}"
        self._packer = msgpack_numpy.Packer()
        self._api_key = api_key
        self._ws, self._server_metadata = self._wait_for_server()
        
    def get_server_metadata(self) -> Dict:
        return self._server_metadata

    def _wait_for_server(self) -> Tuple[websockets.sync.client.ClientConnection, Dict]:
        logger.info(f"Waiting for server at {self._uri}...")
        while True:
            try:
                headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
                conn = websockets.sync.client.connect(
                    self._uri, compression=None, max_size=None, additional_headers=headers
                )
                metadata = msgpack_numpy.unpackb(conn.recv())
                return conn, metadata
            except ConnectionRefusedError:
                logger.info("Still waiting for server...")
                time.sleep(5)

    @override
    def infer(self, obs: Dict) -> Dict:  # noqa: UP006
        data = self._packer.pack(dict(obs=obs, message_type=MessageType.INFER))
        self._ws.send(data)
        response = self._ws.recv()
        if isinstance(response, str):
            # we're expecting bytes; if the server sends a string, it's an error.
            raise RuntimeError(f"Error in inference server:\n{response}")
        return msgpack_numpy.unpackb(response)

    @override
    def feedback(self, obs: Dict, rewards: float, terminated: bool, truncated: bool, info: Dict) -> None:
        data = self._packer.pack(
            message_type=MessageType.FEEDBACK,
            feedback=dict(obs=obs, rewards=rewards, terminated=terminated, truncated=truncated, info=info)
        )
        self._ws.send(data)
        response = self._ws.recv()
        if isinstance(response, str):
            # we're expecting bytes; if the server sends a string, it's an error.
            raise RuntimeError(f"Error in feedback server:\n{response}")

    @override
    def reset(self) -> None:
        pass