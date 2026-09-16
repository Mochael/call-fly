# Sharing a GPU with Moshi

Research date: 2026-09-16. This change preserves Moshiko bf16, the fitted readout,
and the full connectome in the text-logit path. It does not replace the model with
MoshiRAG or a speech-to-text/text-to-speech pipeline.

## What upstream actually supplies

1. [Moshi](https://github.com/kyutai-labs/moshi) describes Python as its research/
   experimentation implementation and Rust as its production implementation.
   The Python demo server uses a single conversation lock. It is not an
   out-of-the-box multi-user server.
2. The **same pinned official Python version we already run**,
   `e6a55d2722a65870ef52a6c9f6ecfc0e90f38362`, supports batched streaming throughout
   Moshi and Mimi. [`StreamingModule`](https://github.com/kyutai-labs/moshi/blob/e6a55d2722a65870ef52a6c9f6ecfc0e90f38362/moshi/moshi/modules/streaming.py)
   provides `streaming(batch_size)`, `set_exec_mask(mask)`, and
   `reset_streaming(reset_mask)`. Execution masks explicitly support streams at
   different positions. Reset masks reset only the chosen session rows.
3. [`LMGen`](https://github.com/kyutai-labs/moshi/blob/e6a55d2722a65870ef52a6c9f6ecfc0e90f38362/moshi/moshi/models/lm.py)
   has `support_out_of_sync=True`, individual offsets and masked token-cache
   writes. Without that flag, a per-row reset also resets the scalar startup
   counter and can suppress a peer's output. The shared engine enables it and
   discards the negative output-token sentinel for unprimed/inactive rows.
4. [MoshiRAG's Rust server](https://github.com/kyutai-labs/moshi-rag/blob/8c6dfc101b7871baa428424bcdc583b74fb561d9/rust/moshi-backend/src/batched_channels.rs)
   supplies a complete pool of WebSocket session slots around a shared model
   loop. This is useful upstream evidence and a design reference, but using it
   would require a Rust port of our connectome hook, transport adaptation and
   validation of another runtime/checkpoint representation. Its presence is not
   evidence that our modified model will meet latency targets on a chosen GPU.
5. [Modal input concurrency](https://modal.com/docs/guide/concurrent-inputs) routes
   several persistent requests into one container. It does not implement model
   session isolation or batching. We must set its connection capacity to match
   the number of model rows; extra requests scale onto another GPU container.

## Selected implementation

Use official Moshi/Mimi masked streaming on the existing pinned PyTorch backend.
This is the smallest change that reuses upstream model execution and preserves
the validated connectome integration. No changes to installed Moshi source,
model architecture, tokenizer, sampling or audio codec are required.

Our code adds a bounded session-slot scheduler and the existing per-session
connectome hook. The model and graph topology load once. Each row has separate
Moshi/Mimi caches, frame counters and connectome recurrence state. A single GPU
thread owns the model and CUDA graphs. A disconnected row's result is discarded
by session identity before a reused slot can receive it. Slow clients are closed
individually rather than stalling their peers.

This is small real-time batching of 80 ms audio chunks, not waiting for complete
conversations. Idle rows are masked; an idle server does no inference. Each new
session receives its own silent priming frame before its microphone is accepted.

## Validation requirements

- Benchmark full model + CUDA Mimi + connectome, not just the transformer.
- Confirm masked rows retain their model offsets and brain state.
- Confirm reset/rejoin does not suppress or reset ongoing peers.
- Check independent audio/brain results and no stale output after slot reuse.
- Exercise paced simultaneous calls, joining/leaving mid-call, and browser audio.
- Prove multiple callers report the **same GPU container ID**, with different
  session slots and call IDs. Multiple successful connections alone is not proof.
- Publish measured capacity rather than claiming an arbitrary number of callers.
