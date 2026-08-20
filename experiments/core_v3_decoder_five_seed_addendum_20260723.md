# Core-v3 Decoder Five-Seed Addendum

Status: completed; the frozen five-seed gate passed on 2026-07-24.

## Scope

This addendum increases the decoder confirmation from three independent
full-pipeline seeds (`17/29/43`) to five (`17/29/43/59/71`). It does not change
the model, data, losses, optimization, checkpoint selection, practical
bitstream, decoder comparison, metrics, or statistical gate.

For each new seed, the required chain is:

1. train the frozen joint sender model;
2. build the exact entropy-decoded sender cache;
3. train the frozen original heavy decoder;
4. train the frozen enhanced server decoder;
5. evaluate enhanced versus heavy on byte-identical prefixes;
6. combine the two new seeds with the three preserved historical seeds.

The five-seed analysis retains 10,000 hierarchical bootstrap iterations, 95%
confidence intervals, Holm correction across four prefixes, and the requirement
that every included seed effect is nonnegative for a prefix to pass.

## Resource Bound

The existing A100 measurements give expected per-seed times of about 2.6 h
for joint training, 0.8 h for cache construction, 2.4 h for the heavy decoder,
5.8 h for the enhanced decoder, and at most 1 h for evaluation. The expected
two-seed cost is about CNY 131 including requested CPU cores. The requested
wall-time upper bound is CNY 285; storage and scheduler rounding are not
included.

Both new seeds finished, byte identity was verified for every paired action,
and the five-seed gate passed. Completion evidence and actual cost are recorded
in `experiments/core_v3_decoder_five_seed_completion_20260724.md`.
