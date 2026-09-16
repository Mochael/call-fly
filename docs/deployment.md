# Published web interface and GPU voice service

## Architecture

- Sites privately hosts the existing web app, geometry, and a same-origin Worker gateway.
- The Worker proxies only conversation and diagnostic routes to Modal over HTTPS/WSS. A server-side token authenticates requests; it is never included in browser assets. Site access remains owner-private unless explicitly changed.
- Each Modal GPU container loads pinned native PyTorch Moshiko once and serves **four independent calls** using upstream Moshi/Mimi masked streaming. Mimi encoding and decoding run on CUDA. One inference thread owns the model and CUDA graphs; four CPU threads advance the separate full-connectome states against shared read-only topology. The existing PCM audio transport and fitted readout are retained.
- The PyTorch bf16 readout must be calibrated separately; the MLX q4 checkpoint is deliberately rejected by model/revision validation.
- Serving is restricted to US GPUs and routed through `us-west`. Unrestricted scheduling placed a test container in Seoul, causing high network latency; it is no longer used for serving.
- The stable `web_us` endpoint is a small CPU gateway in `us-west`. It forwards WebSockets to `shared_worker`, with `modal.concurrent(max_inputs=4)` matching four model session slots. Each inference step batches one 80 ms audio chunk per active caller. Modal schedules overflow connections onto additional GPU containers. The old `call_worker` endpoint is retained only for rollback, with zero warm containers.
- GPU pool: minimum one warm container, maximum three, ten-minute idle scale-down for extra containers, five-minute calls. This provides four warm slots and up to twelve total slots; ten simultaneous callers are the bounded scaling test. Hardware availability and account quotas still apply. Overflow beyond warm capacity may wait for GPU allocation and model initialization.
- Health, geometry and client diagnostics stay on the CPU gateway. They do not occupy GPU call slots or start extra GPU containers. The gateway reports allocation progress every five seconds and bounds the connection wait to three minutes; callers can cancel at any point.
- Each session owns its Moshi/Mimi caches, counters and connectome activity. Masked rows do not advance; joining or leaving resets only that row. The scheduler discards in-flight results belonging to a previous occupant of a reused slot. Model weights and graph topology are shared. Sampling uses the upstream batched sampler; we do not promise independent per-call RNG sequences.
- Cloud endpoints require the server-side service token. Local development remains loopback-only; browser clients never receive the token. Site access remains owner-private; adding worker capacity does not change who may open the published site.

## Status

The GPU readout has been calibrated and the Modal endpoint is deployed. The local browser uses a lightweight proxy and does not load Moshi on the Mac. GPU validation results are recorded below.

Verified on 2026-09-16: a real Chrome call used `NVIDIA L40S`, native CUDA Mimi, and US compute (`us-east-2`) via the West Coast routing endpoint. It streamed 30 seconds of audio, averaged 39.84 ms of model computation per 80 ms frame, and ended with zero queued input frames. The browser captured 88 microphone frames during audible playback, received 124 full-connectome updates (27.42 ms average graph time), and successfully stopped and reconnected with fresh state. A separate physical-edge ablation on the native Torch backend changed both text and audio; details are in `docs/moshi-reservoir.md`.

The initial Rust CPU codec could not keep up on Linux and was replaced with CUDA Mimi. The shared engine captures CUDA graphs once when the container starts, then reuses them across joins. Cold starts still include model loading and compilation. Warm joins reset only one session's caches and prime that row. These timings describe model/frame processing, not question-to-answer latency.

The private site is https://call-fly-eric.saffron-heal-8308.chatgpt.site. Its runtime environment points to the same authenticated Modal service. Local-to-Modal audio is verified; an authenticated call through the published Sites gateway still needs a check in the owner's signed-in browser.

At Modal rates checked on 2026-09-16, an L40S is $0.000542/second (about $1.95/GPU-hour), plus CPU and memory. Eight physical cores and 16 GiB requested memory put a continuously active shared instance around $2.82/hour with the 1.15× broad-US region multiplier, before discounts, credits or ancillary charges. At four occupied slots this is about $0.71 per call-hour. One continuously warm GPU container is approximately $2,034 per 30-day month, plus roughly $20 for the small CPU gateway; three continuously running GPU containers would be approximately $6,102 plus the gateway. Extra containers are charged during startup, calls and idle grace. Maximum container counts limit concurrency, not monthly spending. See https://modal.com/pricing for current rates.

## Shared-GPU validation (2026-09-16)

- Official-source research and API choices are in `docs/moshi-serving-research.md`. Upstream supplies model batching, masks and resets; our code supplies bounded WebSocket session scheduling and the existing connectome hook. This is not an unmodified turnkey upstream server.
- On one L40S, four rows including CUDA Mimi and four complete connectomes took 52.41 ms mean, 75.24 ms p95 and 76.32 ms maximum over 250 full updates, below the 80 ms audio period. Allocated GPU memory was 22.18 GB. Two rows averaged 46.86 ms.
- A real deterministic GPU isolation test changed one peer's audio, reset that peer midstream and paused another. The unchanged caller's output audio and hidden features had exactly zero difference. Physically removing only one row's graph edges zeroed only that row's reservoir logit correction; the other three remained nonzero.
- Four simultaneous WebSockets to one GPU streamed 60–66 seconds each, receiving all 3,150 audio frames (252 seconds total), separate audio/brain hashes and distinct session slots. Warm joins took 0.37–0.48 seconds. Mean frame round-trip latency was 384–414 ms and the largest per-call p95 was 567 ms. A subsequent eight-second call reused the GPU with fresh state in 0.39 seconds. These measurements include the local proxy and network, exclude browser playback buffering, and are not question-to-answer latency.
- Two independent Chrome contexts shared one GPU and each played more than 16 seconds of audio while capturing microphone input. Every sampled frame had a nonzero reservoir correction. Stopping and restarting one caller preserved the other caller's stream and reused a different session slot with fresh state.
- Production scaling test: ten simultaneous callers occupied **three L40S containers** (4 + 4 + 2 slots), delivering all 4,125 frames / 330 seconds of audio. Calls overlapped for 24.8 seconds, then ended at staggered times through 42 seconds. Mean frame round-trip latency was 327–369 ms per caller, with maximum per-call p95 of 443 ms. Final sampled input queues were 1–3 frames. Eight already-loaded slots connected in 0.62–1.11 seconds; two callers waited 38.67 seconds for an additional GPU. A fresh call reused a loaded container in 0.59 seconds with reset state.
- The first scaling attempt began during Modal's rolling deployment and some requests reached the old CPU gateway, which still targeted the single-call endpoint. It was cancelled; the successful test above ran after the rollout and required every caller to identify shared serving. Existing calls are not migrated between engines during deployment.
- Automated checks: 23 Python server/session/proxy/auth/diagnostic tests, 14 JavaScript component/gateway tests, syntax checks, and the real shared-GPU browser test passed.
- Artifacts: ignored local `artifacts/moshi/shared-benchmark-{2,4}.json`, `shared-isolation.json`, `shared-four-calls.json`, `shared-ten-calls.json`, and `shared-browser.json`.

## Earlier one-GPU-per-call baseline (2026-09-16)

- Ten simultaneous calls through the local web gateway → Modal CPU gateway → ten distinct L40S workers. All ten streamed together for at least 24 seconds; individual calls lasted 24–42 seconds, ending at staggered times. All 4,125 audio frames (330 seconds total) arrived, finite and non-silent, with unique worker IDs, call IDs, audio hashes and brain-activity hashes.
- Every worker kept up with live audio. Final sampled input queues were all zero. Per-call mean input-frame-to-output-frame latency was 363–430 ms; the largest per-call p95 was 496 ms. This includes both gateway hops and network travel, and excludes the browser playback cushion; it is not question-to-answer latency.
- Five already-loaded workers connected in 2.3–2.8 seconds. Five newly allocated workers took 30–42 seconds. One warm worker is kept ready by default; a sudden burst above warm capacity still waits for additional models to load.
- Reconnecting reused an existing worker in 2.5 seconds with a new call ID, frame index 1 and reset reservoir frame numbering. The restarted call delivered another eight seconds of audio.
- Two independent Chrome browser contexts also passed simultaneous microphone capture and Web Audio playback, live reservoir corrections, and stop/restart isolation. Ending one call did not interrupt the other.
- Automated checks: 17 Python server/proxy/auth/diagnostic tests, 14 JavaScript component/gateway checks, and seven targeted browser checks passed. An existing diagnostic-retry test was corrected to wait for the asynchronous upload before asserting its arrival.
- Artifacts are ignored local files: `artifacts/moshi/ten-concurrent-calls.json`, `concurrent-calls.json`, and `concurrent-browser.json`. The published Sites audience gate still requires a separate signed-in owner check; the live transport tests used the local UI with actual Modal inference.

## Preparation and deployment

Use the existing authenticated Modal profile. `deploy/modal_app.py` uses `modal==1.5.5`, native Moshi source revision `e6a55d2722a65870ef52a6c9f6ecfc0e90f38362`, and bf16 model revision `2bfc9ae6e89079a5cc7ed2a68436010d91a3d289`.

1. Prepare the graph with `scripts/setup-connectome.sh`. Supply the three generated calibration WAVs in `artifacts/calibration/clip-{0,1,2}.wav`.
2. Create Modal secret `call-fly-service` with a strong `VOICE_SERVICE_TOKEN`. Store the same value as a secret in the Sites environment; never commit it or print it into logs.
3. Run `.runtime/deploy-venv/bin/modal run deploy/modal_app.py::calibrate`. The fitted bf16 readout is saved on volume `call-fly-model-data` with graph/model hashes. This starts paid GPU work.
4. Run a bounded same-seed ablation and paced duplex probe against the GPU backend. Check reset/reconnect, audio, graph contribution and per-frame latency before serving users.
5. Run `.runtime/deploy-venv/bin/modal deploy deploy/modal_app.py`. Configure the returned **web_us CPU gateway** HTTPS origin as Sites `VOICE_BACKEND_URL` and the matching secret `VOICE_SERVICE_TOKEN`.
6. Run `npm run build`, push the exact source to the Sites repository, package, save a version and deploy it privately. Keep the independent GitHub PR stack for review.
7. Verify the published URL, then a real microphone/audio call through the gateway, with a second device/browser if possible. A successful static page is not evidence of working remote audio.

## Local browser with Modal inference

Create ignored `.runtime/modal-service.json` with permissions `600`:

```json
{"url":"https://YOUR-DEPLOYMENT.modal.run","token":"YOUR-SERVICE-TOKEN"}
```

Use the same token as Modal secret `call-fly-service`. Run `MOSHI_TRANSPORT=modal ./scripts/start.sh`, then open http://localhost:8765. The loopback server serves assets and relays PCM/telemetry over an authenticated WSS connection. It does not import or initialize the inference engine. Health polling and geometry are local and do not start the GPU. A call displays a warming status until the remote model is ready.

`./scripts/start.sh` automatically chooses this proxy when the configuration is valid. `MOSHI_TRANSPORT=local` explicitly selects the original MLX backend. Set `MOSHI_BACKEND=torch` only for GPU deployment; a missing service token rejects all cloud HTTP/WebSocket access. Local loopback host/origin checks remain active.

Run `EXPECT_MODAL=1 npm run test:browser` to require the remote readiness event to identify a Torch backend and NVIDIA GPU. The test exercises actual audio, simultaneous capture and playback, nonzero reservoir corrections, the full graph, stop and reconnect. `scripts/verify_moshi.py --seconds 60` adds a paced transport/latency probe.

For an explicit bounded concurrency test (starts paid GPUs):

```sh
.runtime/moshi-venv/bin/python scripts/verify_concurrent_moshi.py --calls 10 --seconds 24 --max-workers 3 --output artifacts/moshi/shared-ten-calls.json
EXPECT_MODAL=1 npm run test:browser -- tests/browser/concurrent-calls.spec.js
```

The API probe waits for every caller, streams distinct prerecorded inputs concurrently, requires unique call IDs and (container, slot) pairs within the maximum GPU count, checks complete finite/non-silent output and distinct brain/audio hashes, and stops callers at staggered times. It then reuses an existing container and checks that frame numbering and reservoir state restart. The browser test opens independent browser contexts, verifies shared-GPU sessions with actual Web Audio playback while microphone capture continues, and stops/restarts one caller while the other keeps receiving audio. Neither test proves arbitrary-length or unlimited-load reliability; calls still have a five-minute limit.

For isolated GPU benchmarks before changing production, `modal run deploy/modal_shared.py --capacity 4 --steps 250` and `modal run deploy/modal_shared.py --mode isolation` run paid bounded jobs. `modal deploy deploy/modal_shared.py` creates a separate preview endpoint with one GPU and no warm minimum; stop that preview app after testing. The optional `MODAL_SERVICE_CONFIG` path selects a separate private local proxy configuration for preview tests. Set `TEST_BASE_URL` to that local proxy and `EXPECT_SHARED=1` to require both browser callers to occupy the same GPU. Production may distribute two callers across already-loaded containers, so its default browser assertion requires distinct session identities without requiring identical container IDs.

Content-free call diagnostics are also emitted to Modal's container logs, so ending reasons and processing metrics survive container scale-down. Audio, transcripts and hidden states are not included. Inspect logs with `.runtime/deploy-venv/bin/modal app logs call-fly-voice`.

## Stopping spend

Stop the `call-fly-voice` Modal app to stop all serving compute, or reduce `shared_worker` minimum containers to zero to allow the warm GPU to shut down when idle. Clearing `VOICE_BACKEND_URL` only disables new calls from the website; it does not shut down the warm GPU. Configure a provider budget separately if available; concurrency and idle limits alone do not impose a spending ceiling.
