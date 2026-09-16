"""FIFO admission and live queue positions, owned by one gateway event loop."""
import asyncio
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field


class QueueWaitExpired(Exception):
    pass


@dataclass(eq=False)
class Ticket:
    changed: asyncio.Event = field(default_factory=asyncio.Event)
    state: str = 'waiting'


class CallQueue:
    def __init__(self, capacity):
        if capacity < 1:
            raise ValueError('Call capacity must be positive')
        self.capacity = capacity
        self.active = 0
        self.waiting = deque()

    def join(self):
        ticket = Ticket()
        self.waiting.append(ticket)
        self._advance(notify=False)
        return ticket

    def leave(self, ticket):
        if ticket.state == 'closed':
            return
        if ticket.state == 'admitted':
            self.active -= 1
        else:
            self.waiting.remove(ticket)
        ticket.state = 'closed'
        self._advance()

    def _advance(self, notify=True):
        while self.waiting and self.active < self.capacity:
            ticket = self.waiting.popleft()
            ticket.state = 'admitted'
            self.active += 1
            ticket.changed.set()
        if notify:
            for ticket in self.waiting:
                ticket.changed.set()

    def people_ahead(self, ticket):
        return self.waiting.index(ticket)

    @asynccontextmanager
    async def admission(self, ws, incoming, *, timeout=300, progress=5):
        ticket = self.join()
        try:
            deadline = asyncio.get_running_loop().time() + timeout
            while ticket.state == 'waiting':
                if incoming.done():
                    yield False
                    return
                # Snapshot and clear happen without an await, so updates cannot be lost.
                ticket.changed.clear()
                ahead = self.people_ahead(ticket)
                place = "You're next." if ahead == 0 else f'{ahead} {"person" if ahead == 1 else "people"} ahead of you.'
                await ws.send_json({'type':'warming', 'phase':'queued',
                    'people_ahead':ahead,
                    'message':f"Eric is busy right now. {place} Your call will start automatically."})
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    raise QueueWaitExpired()
                changed = asyncio.create_task(ticket.changed.wait())
                try:
                    await asyncio.wait([changed, incoming], timeout=min(progress, remaining),
                                       return_when=asyncio.FIRST_COMPLETED)
                finally:
                    changed.cancel()
                    await asyncio.gather(changed, return_exceptions=True)
            yield not incoming.done()
        finally:
            # Handles timeout, disconnect, failed writes and cancellation during handoff.
            self.leave(ticket)
