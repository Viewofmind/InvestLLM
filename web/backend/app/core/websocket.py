"""
WebSocket connection manager for real-time updates
"""

import json
import logging
from typing import Dict, List, Set, Any
from fastapi import WebSocket

logger = logging.getLogger(__name__)


class ConnectionManager:
    """
    Manages WebSocket connections for real-time updates.

    Supports:
    - Multiple concurrent connections
    - Channel-based subscriptions
    - Broadcast to all or specific channels
    """

    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.subscriptions: Dict[WebSocket, Set[str]] = {}

    async def connect(self, websocket: WebSocket):
        """Accept and store new connection"""
        await websocket.accept()
        self.active_connections.append(websocket)
        self.subscriptions[websocket] = set(["all"])  # Default subscription
        logger.info(f"New WebSocket connection. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        """Remove disconnected client"""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        if websocket in self.subscriptions:
            del self.subscriptions[websocket]
        logger.info(f"WebSocket disconnected. Total: {len(self.active_connections)}")

    def subscribe(self, websocket: WebSocket, channels: List[str]):
        """Subscribe client to specific channels"""
        if websocket in self.subscriptions:
            self.subscriptions[websocket].update(channels)

    def unsubscribe(self, websocket: WebSocket, channels: List[str]):
        """Unsubscribe client from channels"""
        if websocket in self.subscriptions:
            self.subscriptions[websocket] -= set(channels)

    async def send_personal(self, websocket: WebSocket, message: Dict[str, Any]):
        """Send message to specific client"""
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            self.disconnect(websocket)

    async def broadcast(self, message: Dict[str, Any], channel: str = "all"):
        """
        Broadcast message to all clients subscribed to channel.

        Args:
            message: Message dict to send
            channel: Channel name (default "all" sends to everyone)
        """
        disconnected = []

        for connection in self.active_connections:
            subscribed_channels = self.subscriptions.get(connection, set())

            if "all" in subscribed_channels or channel in subscribed_channels:
                try:
                    await connection.send_json(message)
                except Exception as e:
                    logger.error(f"Broadcast failed for client: {e}")
                    disconnected.append(connection)

        # Clean up disconnected clients
        for conn in disconnected:
            self.disconnect(conn)

    async def broadcast_portfolio_update(self, data: Dict):
        """Broadcast portfolio update"""
        await self.broadcast({
            "type": "portfolio_update",
            "data": data
        }, channel="portfolio")

    async def broadcast_trade(self, trade: Dict):
        """Broadcast trade execution"""
        await self.broadcast({
            "type": "trade",
            "data": trade
        }, channel="trades")

    async def broadcast_risk_alert(self, alert: Dict):
        """Broadcast risk alert"""
        await self.broadcast({
            "type": "risk_alert",
            "data": alert
        }, channel="risk")

    async def broadcast_signal(self, signal: Dict):
        """Broadcast trading signal"""
        await self.broadcast({
            "type": "signal",
            "data": signal
        }, channel="signals")

    async def broadcast_quote(self, symbol: str, quote: Dict):
        """Broadcast price quote"""
        await self.broadcast({
            "type": "quote",
            "symbol": symbol,
            "data": quote
        }, channel="quotes")
