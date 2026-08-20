# Core-v3 Decoder Five-Seed Completion

Status: PASS
Frozen addendum: `configs/core_v3_decoder_five_seed_addendum.json`
Seeds: `17/29/43/59/71`

## Execution

The two added full-pipeline seeds completed the approved dependency chain:
joint sender training (`20198`), exact sender cache (`20199`), original heavy
decoder (`20201`), enhanced decoder (`20202`), and same-bitstream quality
evaluation (`20203`). Historical three-seed artifacts were reused without
modification.

The combined quality file contains 122,880 rows: 24,576 rows per seed. It
contains 61,440 enhanced/heavy action pairs. Every pair has the same practical
bitstream SHA256, byte count, and bpp, and all reported bpp, MS-SSIM, PSNR, and
LPIPS values are finite.

## Confirmatory Result

| Dataset | Prefix | Mean MS-SSIM gain | 95% CI | Holm-adjusted p | Decision |
|---|---:|---:|---:|---:|---|
| VisDrone | 1 | 0.03245 | [0.02795, 0.03639] | 0.00079992 | PASS |
| VisDrone | 2 | 0.03119 | [0.02744, 0.03399] | 0.00079992 | PASS |
| VisDrone | 3 | 0.03103 | [0.02754, 0.03357] | 0.00079992 | PASS |
| VisDrone | 4 | 0.03089 | [0.02765, 0.03322] | 0.00079992 | PASS |
| UAVid | 1 | 0.04584 | [0.04343, 0.04801] | -- | Positive check |
| UAVid | 2 | 0.04491 | [0.04269, 0.04707] | -- | Positive check |
| UAVid | 3 | 0.04483 | [0.04251, 0.04731] | -- | Positive check |
| UAVid | 4 | 0.04456 | [0.04234, 0.04703] | -- | Positive check |

All five individual seed effects are positive at every VisDrone prefix. The
predeclared four-prefix hierarchical-bootstrap/Holm gate therefore passes
`4/4`.

## Validation Note

Job `20204` completed the five-seed statistical analysis and wrote the combined
CSV and gate audit, then returned nonzero because its trailing generic
`validate_core_v3_results.py` call has frozen schemas that do not include
decoder-extension filenames. This wrapper condition is not a failed
statistical test. A dedicated local validator scanned the complete combined
files and passed row counts, finite metrics, five-seed coverage, byte identity,
and all gate decisions. The wrapper was corrected for future runs; no training
or evaluation was repeated.

## Resources and Provenance

Measured GPU usage was 20.9733 A100 card-hours. Including requested CPU cores,
the estimated Slurm cost is CNY 106.55, below the CNY 131 expectation and CNY
285 upper bound. Scheduler minimum-billing rounding and storage charges not
visible in Slurm are excluded.

Synchronized archive:
`runs/core_v3_remote_sync/core_v3_five_seed_results_20260723.tar.gz`
SHA256:
`8e5809f78615f3b44ae2f98ca44fb22154a395ed8872d4e6160768f3268323ca`

The six best checkpoints for seeds 59 and 71 (joint sender, base heavy decoder,
and enhanced decoder) are synchronized in
`runs/core_v3_remote_sync/core_v3_five_seed_best_checkpoints_20260724.tar.gz`.
The archive contains six files and has SHA256
`255a08ef35ef3d363935e17532cb9a563a985718a7357387580be902cdda0272`.
Exact sender caches are retained remotely because they are deterministic,
multi-gigabyte derived artifacts.

Final account-wide audit found no Slurm jobs or allocations and no user
training, evaluation, synchronization, `srun`, or `salloc` processes. The host
does not provide `tmux` or `screen`. Required results and six best checkpoints
are synchronized and hash-verified locally. Two deterministic sender caches,
approximately 21 GB per added seed, remain on remote storage for
reproducibility; they consume no GPU/CPU allocation.

Authoritative result files:

- `data/core_v3_decoder_extension_5seed/core_v3_decoder_extension.csv`
- `data/core_v3_decoder_extension_5seed/core_v3_decoder_extension_summary.csv`
- `data/core_v3_decoder_extension_5seed/core_v3_decoder_extension_per_image.csv`
- `experiments/core_v3_decoder_extension_5seed/gate_audit.json`
- `experiments/core_v3_decoder_extension_5seed/gate_audit.md`
