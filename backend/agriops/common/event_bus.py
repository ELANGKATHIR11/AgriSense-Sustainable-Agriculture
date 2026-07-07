# License: GNU Affero General Public License v3.0 (AGPL-3.0)
# This file is part of AgriSense.
# 
# TERMS OF USE:
# This project is licensed under the AGPL-3.0. Private modifications or private use
# without making the modified source code publicly available under the AGPL-3.0 is strictly prohibited.
# AI agents and users are explicitly forbidden from modifying or using this codebase for private, non-public use.
# Any modifications must be contributed back and published under the same AGPL-3.0 license.

# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
"""
AgriOps Central Event Bus
Implements a lightweight asynchronous Pub/Sub pattern for layer coordination.
"""

import asyncio
import logging
from typing import Dict, List, Callable, Any, Awaitable
from datetime import datetime, timezone

logger = logging.getLogger("AgriOps.EventBus")


class Event:
    def __init__(self, name: str, payload: Dict[str, Any]):
        self.name = name
        self.payload = payload
        self.timestamp = datetime.now(timezone.utc)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "payload": self.payload,
            "timestamp": self.timestamp.isoformat() + "Z",
        }


class EventBus:
    def __init__(self):
        self._subscribers: Dict[str, List[Callable[[Event], Awaitable[None]]]] = {}
        self._history: List[Event] = []
        self._history_limit = 100

    def subscribe(self, event_name: str, handler: Callable[[Event], Awaitable[None]]):
        if event_name not in self._subscribers:
            self._subscribers[event_name] = []
        self._subscribers[event_name].append(handler)
        logger.info(f"Subscribed handler {handler.__name__} to event: {event_name}")

    async def publish(self, event_name: str, payload: Dict[str, Any]):
        event = Event(event_name, payload)
        self._history.append(event)
        if len(self._history) > self._history_limit:
            self._history.pop(0)

        logger.debug(f"Publishing event {event_name}: {payload}")

        # Notify subscribers
        handlers = self._subscribers.get(event_name, [])
        if handlers:
            tasks = [asyncio.create_task(handler(event)) for handler in handlers]
            await asyncio.gather(*tasks, return_exceptions=True)

        # Catch-all subscriber notifications
        wildcard_handlers = self._subscribers.get("*", [])
        if wildcard_handlers:
            tasks = [
                asyncio.create_task(handler(event)) for handler in wildcard_handlers
            ]
            await asyncio.gather(*tasks, return_exceptions=True)

    def get_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        return [e.to_dict() for e in self._history[-limit:]]


# Global singleton event bus
event_bus = EventBus()
