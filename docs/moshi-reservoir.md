# Connectome inside Moshi

## Implemented inference path

Moshi temporal hidden state → 64 fixed features → full MaleCNS recurrence → 128 pooled features → fitted bias-free readout → bounded text-logit correction → sampled text → Moshi audio depformer.

This follows FLM's additive readout pattern, rather than replacing Moshi or feeding sound through an unrelated visualizer. Every retained edge participates. The graph runs once every three 80 ms model frames (4.17 Hz); its readout is held between updates. This is an engineering rate, not biological time. The browser cannot drive, pause, reset, or perturb model computation. A call reset clears all reservoir and model state.

The fixed recurrence is `x = tanh(W @ (0.6*x + 0.4*input))`. Anatomical contacts are unsigned and normalized by incoming totals. Input mapping and output pooling are seeded numerical interfaces. They are not identified speech pathways. No self-improvement, reward learning, or additional reservoir-feedback connections are enabled.

## Readout and training

The readout is a 128 × 4096 matrix with no bias, fitted offline by ridge regression. The target is the frozen Moshi teacher's captured hidden state; the training script does not apply separate target normalization. The residual is `0.03*tanh(pooled @ readout)`, projected through Moshi's existing text head, centered across the vocabulary, and capped at 0.10 RMS in logit units. This changes text probabilities before sampling. Sampled text also conditions audio generation; no independent waveform transformation is added.

This is a **causal prototype**, not a demonstrated improvement in conversation. The calibration objective reconstructs teacher features; it does not teach a fly persona, comprehension, or better answers. We do not load FLM's adapter because its hidden dimensions and language backbone differ. Moshi, graph weights, and fitted adapter remain frozen during calls.

Calibration used two generated speech clips and a third held-out clip: 361 training and 176 validation reservoir observations. Held-out reconstruction MSE was 0.7002, compared with 2.0845 for zero prediction and 1.0485 for the training-mean prediction. These are small development splits, not a speech-quality benchmark.

The artifact is `.runtime/moshi-reservoir/adapter.npz` plus its manifest. Loading verifies exact model ID/revision, graph manifest hash, interface settings, weight checksum, dimensions, and finiteness. Serving fails explicitly if the adapter is missing or incompatible; it does not silently fall back to a visualizer.

To calibrate with the voice server stopped:

```sh
.runtime/moshi-venv/bin/python scripts/train_moshi_reservoir.py \
  --audio path/to/train-1.wav path/to/train-2.wav path/to/validation.wav --seconds 55
```

Use generated or otherwise authorized audio. The last clip is validation; earlier clips are training. A saved adapter is never overwritten without `--replace`. Live visitor conversations are not training data and are not stored.

## Visualization

The call's own reservoir snapshots travel with its audio frames. States are quantized to signed 8-bit values over [-1,1] for transport and displayed near the associated playback time. Quantization error is at most approximately 0.004 per state. The readout always uses the full-precision state. The UI's bursts are changes between these displayed snapshots, not biological spikes. There is no second simulated brain or graph WebSocket.

## Causal validation

- Tests physically remove the graph's edges while retaining the readout: its signal becomes exactly zero, and the text head matches the original scores.
- The intact path changes scores before sampling; corrections obey the RMS bound.
- Reset removes state across calls; graph cadence follows model frames, not rendering or network requests.
- `scripts/verify_reservoir_causality.py` compares real Moshi with identical audio and RNG seed, intact versus disconnected graph, and saves measurements under `artifacts/moshi/`.
- Real browser tests cover duplex audio, changing graph state, stop/restart, neuron inspection and zoom.

## Attribution

The recurrence and additive-readout architecture are adapted from [nftechie/flm](https://github.com/nftechie/flm), MIT, Alex Wormuth (2026). Its notice is preserved in `dist/neural/FLM_LICENSE.txt`. See `docs/full-connectome.md` for MaleCNS CC BY 4.0 attribution and data hashes. Moshi is by Kyutai; its code and weights retain their upstream licenses.

## Local live validation

All 19 targeted Python checks, 11 component tests, and 9 browser tests passed. The actual 30-second Moshi browser call averaged 66.36 ms of model computation per 80 ms frame, with zero input backlog at the last checkpoint and 73 microphone frames captured during audible output. A separate 12.8-second same-seed ablation measured a mean 0.03034 logit correction with the graph intact versus exactly zero with physical edge removal; that short run sampled identical text and audio. Score-level causality is established; a conversational quality benefit is not.

## NVIDIA checkpoint calibration

The Modal backend uses native PyTorch Moshiko bf16 and a separately fitted readout at `torch/adapter.npz` on volume `call-fly-model-data`. On the same generated speech split (361 training / 176 validation observations), held-out MSE was 0.84764, versus 2.17979 for zero prediction and 1.23550 for the training-mean prediction. Calibration ran on an NVIDIA L4. This remains a feature-reconstruction check, not evidence of improved speech quality.

The 30-second native PyTorch comparison used identical input audio and RNG seed with intact versus physically removed graph edges. Mean logit correction was 0.02983 with the graph and exactly zero without it; both sampled text and audio changed. On the L4, model compute averaged 90.71 ms intact versus 76.17 ms disconnected. Serving therefore uses an L40S. This establishes causal influence, not a quality benefit.
