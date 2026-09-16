# Vercel frontend and Modal inference

Deployed at **https://call-fly.vercel.app**, with Vercel Authentication enabled for all deployment URLs.

Vercel hosts the browser assets and a small Node.js WebSocket gateway. Moshi, Mimi and the connectome remain on the existing Modal GPU service. No GPU model is loaded in Vercel, and no Modal deployment change is required.

```text
Browser → Vercel /api/conversation → authenticated Modal CPU gateway → shared Moshi GPU
Browser ← Vercel WebSocket relay   ← audio, text and connectome snapshots
```

The browser continues using same-origin routes. The Vercel gateway supplies the Modal token server-side and does not forward browser cookies or authorization headers upstream. It rejects cross-origin WebSocket upgrades and diagnostic requests, bounds incoming frames and queued output, and releases the upstream connection when a caller stops or disconnects, including during startup.

## Requirements

- A Vercel project with Fluid compute enabled. Vercel WebSocket support is in public beta as of September 2026.
- A Pro/Enterprise duration allowance: `api/conversation.js` uses a 600-second maximum, accommodating GPU startup plus a five-minute call. The gateway closes at 550 seconds as a final bound. Hobby's 300-second maximum cannot preserve this startup-plus-call budget unchanged.
- The existing authenticated Modal service, with the shared Moshi engine and fitted adapter already deployed.

Official references: [WebSockets](https://vercel.com/docs/functions/websockets), [function durations](https://vercel.com/docs/functions/configuring-functions/duration), [Vercel Authentication](https://vercel.com/docs/deployment-protection/methods-to-protect-deployments/vercel-authentication).

## Configuration

Set these Vercel environment variables for production and preview:

- `VOICE_BACKEND_URL`: the HTTPS origin of the existing Modal **CPU gateway**, not a GPU endpoint.
- `VOICE_SERVICE_TOKEN`: the matching Modal secret, stored as a sensitive server-side environment variable.

`vercel.json` selects the Vercel build, static output and San Francisco function region. Geometry routes rewrite to static assets. `/api/health` reports configuration without waking the GPU; `/api/call-diagnostics` forwards bounded operational reports to the CPU gateway.

`npm run build:vercel` copies an explicit allowlist of browser assets. It reads public geometry from local `.runtime/connectome/` when available; otherwise it downloads the three public geometry assets through the authenticated CPU gateway during the build. It verifies neuron/edge counts and geometry hashes before producing output. Neither the service token nor raw recordings, models, adapters or diagnostic logs are written to the build.

## Deploy

```sh
npm ci
npm run check
npm run test:ui
npm run test:gateway
npx vercel login
npx vercel link
# Configure the server-side environment variables in Vercel.
npx vercel deploy
# After testing the protected deployment:
npx vercel deploy --prod
```

Use the existing `call-fly` project rather than creating another on each deployment. The project's Vercel Authentication setting protects all URLs until the owner chooses to make the website public. A public GitHub repository does not imply public GPU access. Do not commit `.vercel/`, environment files or deployment credentials.

The original Sites deployment remains a separate fallback; the old build script and gateway are retained. Deploying Vercel does not change Sites access settings or stop Modal compute. See [Modal deployment](deployment.md) for GPU capacity, billing and shutdown controls.

## Verification

`tests/vercel-gateway.test.mjs` exercises real local WebSocket relay, binary/text preservation, credential isolation, cross-origin rejection, cancellation during upstream startup, and diagnostic size limits. Existing UI and gateway tests remain applicable.

A successful Vercel deployment only proves that the application built. Verify its health and geometry routes, then stream actual audio over its WebSocket and check output audio, nonzero reservoir corrections, changing brain states, stop/reconnect and cleanup. Protected deployments require authenticated test access. Test audio and results belong in ignored `artifacts/`, never in the public repository.

### Deployed verification (2026-09-16)

The production Vercel URL passed an authenticated real-speech WebSocket probe against the actual Modal L40S backend. All 375 audio frames (30 seconds) arrived finite and non-silent, all 375 model frames had a nonzero reservoir correction, and 125 distinct connectome snapshots arrived. A subsequent call had a fresh call ID and delivered all 75 frames (six seconds), with 25 distinct brain snapshots. Warm connection times were 616 ms and 659 ms. Mean frame round-trip time was 386 ms (p95 476 ms) on the first call and 250 ms (p95 276 ms) on restart; these are transport/frame measurements, not question-to-answer latency.

Health and public-geometry checks passed, including geometry hashes. Anonymous access redirected to sign-in. Temporary automated test access was revoked after verification. Six Vercel gateway tests and 14 existing component/gateway tests passed; the dependency audit and source credential scan found no issues.

To repeat the bounded live probe with authorized test speech and authenticated access:

```sh
node scripts/verify_hosted_voice.mjs --url https://YOUR-SITE.vercel.app \
  --audio /path/to/test-speech.wav --seconds 30
```

For a protected deployment, provide `VERCEL_AUTOMATION_BYPASS_SECRET` securely in the test process environment; never put it in a URL, commit it, or log it. The probe requires FFmpeg, checks real audio and reservoir state, stops and reconnects, and writes only operational metrics to ignored `artifacts/vercel/voice-probe.json`. It does not save transcripts or audio.
