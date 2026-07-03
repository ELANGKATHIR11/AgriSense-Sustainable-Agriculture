import logging
import asyncio
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import List

logger = logging.getLogger("MarketWebSocket")
router = APIRouter(prefix="/market", tags=["Market Realtime"])


class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        async with self._lock:
            self.active_connections.append(websocket)
        logger.info("New WebSocket client connected to Market Intelligence.")

    async def disconnect(self, websocket: WebSocket):
        async with self._lock:
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)
                logger.info("WebSocket client disconnected.")

    async def broadcast(self, message: dict):
        """
        Broadcast incremental updates to all active clients safely.
        """
        async with self._lock:
            # Copy list to iterate safely
            connections = list(self.active_connections)

        disconnected_clients = []
        for connection in connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.warning(f"Error sending message to client: {e}")
                disconnected_clients.append(connection)

        for client in disconnected_clients:
            await self.disconnect(client)


# Instantiate the shared connection manager
manager = ConnectionManager()


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Maintain connection alive, echo back any client pings
            data = await websocket.receive_text()
            await websocket.send_json({"status": "alive", "echo": data})
    except WebSocketDisconnect:
        await manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await manager.disconnect(websocket)
