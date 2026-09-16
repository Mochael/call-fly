# Waiting-room positions

One Modal CPU gateway owns an in-memory FIFO queue. It admits at most four conversations to the single warm GPU. Waiting WebSockets stay on the CPU gateway and send no microphone audio to the GPU.

A `warming` event with `phase: "queued"` includes `people_ahead`: the number of waiting callers before this connection, excluding active calls. Zero means "You're next." Positions update when callers cancel, disconnect, time out, or are admitted. Each reservation is released exactly once, including cancellation during admission. The queue expires after five minutes; callers can cancel at any time.

The gateway must remain one container with one event loop. Positions and reservations are not stored in a database and do not survive a gateway restart. A reconnect joins at the back of the line. Multiple gateway replicas would require shared queue coordination before enabling them.

Verification: `tests/test_call_queue.py` covers ordering, position changes, timeouts, failed sends and cancellation during handoff; `tests/test_modal_proxy.py` checks actual gateway WebSocket admission and isolation with an in-process fake inference service. `scripts/verify_waiting_room.py` checks four real GPU calls and three waiting connections, including position updates and automatic admission. Its recordings and results remain ignored local artifacts.
