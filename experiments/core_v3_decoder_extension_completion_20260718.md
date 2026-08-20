# Core-v3 Decoder Extension Completion

Date: 2026-07-18

## Scope

This note closes the deterministic decoder-side extension
`core_v3_decoder_extension_20260717_r1`. The extension is decoder-only: it
reuses the frozen core-v3 sender checkpoints, entropy strings, q-indices,
prefix groups, manifests, and test splits.

## Remote Jobs

| Stage | Job | State | Cost estimate |
|---|---|---|---:|
| Benchmark | `19709` | `COMPLETED/0:0` | `0.67 CNY` |
| D2 pilot | `19712` | `COMPLETED/0:0` | `5.08 CNY` |
| Three-seed training | `19714` | `COMPLETED/0:0` | `82.07 CNY` |
| Quality/complexity | `19733` | `COMPLETED/0:0` | `1.56 CNY` |
| Statistical analysis | `19736` | `COMPLETED/0:0` | `0.0003 CNY` |

Total Slurm-derived compute estimate: `89.38 CNY`, excluding provider minimum
billing-rounding rules not visible through Slurm.

## Evidence

- `data/core_v3_decoder_extension.csv`: `73728` combined rows.
- `data/core_v3_decoder_extension_summary.csv`: paired summary table.
- `data/core_v3_decoder_extension_per_image.csv`: paired per-image aggregates.
- `experiments/core_v3_decoder_extension/gate_audit.json`: confirmatory gate.
- `experiments/core_v3_decoder_extension/gate_audit.md`: readable gate report.

The gate audit reports byte-identical pairs for `36864` paired actions,
VisDrone positive prefixes `4/4`, and confirmatory gate `PASS`.

VisDrone enhanced-over-heavy MS-SSIM effects:

- Prefix 1: `0.033150`, 95% CI `[0.031624, 0.035209]`, Holm p `0.00079992`.
- Prefix 2: `0.032398`, 95% CI `[0.031185, 0.033887]`, Holm p `0.00079992`.
- Prefix 3: `0.032108`, 95% CI `[0.031071, 0.033260]`, Holm p `0.00079992`.
- Prefix 4: `0.032046`, 95% CI `[0.031003, 0.033052]`, Holm p `0.00079992`.

## Manuscript and QA

The manuscript was updated only after the confirmatory gate passed. Current QA:

- References: `PASS`, 45 cited references.
- Claim support: `PASS`.
- Formula consistency: `PASS`.
- Chinese manuscript audit: `PASS`.
- MDPI template compliance: `PASS_WITH_WARNINGS`.
- Submission readiness: `PASS_WITH_WARNINGS`.

Warnings are limited to final author/affiliation/funding/contribution metadata
placeholders.

The DOCX was rebuilt and rendered to 15 pages. The current compact review
archive is `BAA-NIC_core_v3_decoder_extension_review_package_20260718.zip`;
its SHA256 is recorded in the project README and handoff file after packaging.

## Resource Audit

BAA-NIC decoder-extension jobs `19709`, `19712`, `19714`, `19733`, and `19736`
are all complete. No BAA-NIC training/evaluation processes or tmux/screen
sessions remain. The account still has a separate `mining1_E200` A100 array
(`19702`) running/pending; it is unrelated and was not modified.
