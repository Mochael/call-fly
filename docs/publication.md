# Public source and privacy

The public repository is a source snapshot of Call fly with a new history. It excludes the private development repository’s commit metadata and PR discussions. Publication commits use a GitHub no-reply email. The original development repository remains private.

## What is included

Application source, tests, dependency locks, setup/deployment scripts, public legacy connectome assets with their notices, and documentation of the model integration and measurements. Public upstream data URLs and checkpoint identifiers support reproducing the setup. Deployment identifiers are not authentication credentials.

## What is excluded

- Service/API tokens, `.env` files, private keys and local deployment configuration.
- Personal voice recordings and transcripts.
- Calibration audio, fitted adapter artifacts and downloaded model weights.
- Live-call/test audio, diagnostic logs and verification artifacts.
- Private Git history, personal author email addresses and development PR discussions.

These local inputs live in ignored directories (`.runtime/`, `artifacts/`, `voices/eric/`) or file types covered by `.gitignore`. Ignoring a file does not remove an already committed file from history. Before later publication, check both tracked content and history.

## Publication checks

The initial publication audit scanned the source history with Gitleaks and checked tracked blobs against the locally configured service token without printing it. It found no credential matches or committed recordings. The personal email in the original commit metadata was the reason for starting a clean public history. The final public source, its new history and the deployment build are checked again before publication.

To repeat a credential scan with Gitleaks installed:

```sh
gitleaks git . --log-opts="--all --full-history" --redact
```

Also inspect tracked filenames, commit metadata, releases, PR discussions and any build/upload artifacts. Automated scanners are useful checks, not proof that all sensitive information is absent. Never publish scan output containing secret values.

## Runtime boundaries

The browser sends microphone audio to the configured inference service only during calls. The application does not intentionally retain live audio, transcripts or model-state histories; it retains bounded operational diagnostics. Test scripts can deliberately write recordings to ignored artifacts. Provider infrastructure has its own operational logging policies.

The Vercel and Sites builds allowlist browser assets and public geometry. `VOICE_SERVICE_TOKEN` is configured separately in the server runtime; it is not embedded in the browser bundle. Publishing source does not grant access to the hosted voice service, alter the website audience, or provision GPUs for repository visitors.
