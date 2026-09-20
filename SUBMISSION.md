# Submission

## Memory mechanism

### Write rule

The writer processes policies in reverse priority order. For every condition,
it tracks which contexts have already been claimed by later policies. A policy
keeps only the contexts in its original scope that have not already been
claimed, producing its residual region. Policies with empty residual regions
are discarded.

The writer scores each remaining region using its noisy context-traffic
estimates restricted to that residual scope, selects the highest-value `K`
regions, and packs one region per slot. The same parameters support every
tested budget: only the number of regions selected by top-k changes with `K`.

### Slot layout

Each selected 24-byte slot contains:

- byte 0: the exact signed condition code;
- byte 1: all eight residual-scope bits, losslessly encoded as `mask - 128`;
- bytes 2–23: a learned compression of the action payload.

Priority is not stored as a separate value. It is resolved before packing by
the reverse-order residual compilation: later policies claim their contexts
first, and earlier policies retain only the parts that remain active.

### Read rule

The reader recovers the scope mask from byte 1 and marks a slot compatible only
when both its exact condition and its residual scope match the request. Attention
runs over compatible slots using the compressed action bytes. If the budget has
omitted every compatible slot, the request stays on the residual path rather
than attending to an unrelated rule.

## Public results

Checkpoint: [artifacts/model.pt](artifacts/model.pt)

Sweep artifact: [artifacts/sweep.json](artifacts/sweep.json)

Entries are percentages in the form `overall accuracy [95% Wilson CI]`.

The selected model achieves 67.26% aggregate accuracy over all 327,680 public
decisions. All 24 cells with `K >= N` reach 100% observed accuracy.

### Uniform traffic, sigma=0.0

| N policies \ K slots | 2 | 4 | 8 | 16 | 32 |
|---:|---:|---:|---:|---:|---:|
| 8 | 41.1 [39.6, 42.6] | 72.6 [71.2, 74.0] | 100.0 [99.9, 100.0] | 100.0 [99.9, 100.0] | 100.0 [99.9, 100.0] |
| 16 | 20.0 [18.8, 21.2] | 39.0 [37.5, 40.5] | 73.2 [71.8, 74.5] | 100.0 [99.9, 100.0] | 100.0 [99.9, 100.0] |
| 32 | 9.8 [8.9, 10.8] | 20.3 [19.1, 21.5] | 38.1 [36.6, 39.6] | 72.8 [71.4, 74.1] | 100.0 [99.9, 100.0] |
| 64 | 5.5 [4.9, 6.3] | 10.8 [9.9, 11.8] | 19.0 [17.8, 20.2] | 39.8 [38.3, 41.3] | 72.3 [70.9, 73.7] |

### Uniform traffic, sigma=0.5

| N policies \ K slots | 2 | 4 | 8 | 16 | 32 |
|---:|---:|---:|---:|---:|---:|
| 8 | 38.0 [36.6, 39.5] | 71.6 [70.2, 73.0] | 100.0 [99.9, 100.0] | 100.0 [99.9, 100.0] | 100.0 [99.9, 100.0] |
| 16 | 21.0 [19.8, 22.3] | 38.1 [36.6, 39.6] | 71.9 [70.5, 73.3] | 100.0 [99.9, 100.0] | 100.0 [99.9, 100.0] |
| 32 | 9.3 [8.4, 10.2] | 19.3 [18.1, 20.5] | 39.5 [38.0, 41.0] | 68.5 [67.0, 69.9] | 100.0 [99.9, 100.0] |
| 64 | 5.2 [4.5, 5.9] | 9.2 [8.4, 10.1] | 18.5 [17.3, 19.7] | 38.9 [37.4, 40.4] | 71.7 [70.3, 73.0] |

### Zipf traffic, sigma=0.0

| N policies \ K slots | 2 | 4 | 8 | 16 | 32 |
|---:|---:|---:|---:|---:|---:|
| 8 | 62.3 [60.8, 63.8] | 82.6 [81.5, 83.8] | 100.0 [99.9, 100.0] | 100.0 [99.9, 100.0] | 100.0 [99.9, 100.0] |
| 16 | 51.8 [50.3, 53.3] | 69.0 [67.5, 70.4] | 86.6 [85.6, 87.7] | 100.0 [99.9, 100.0] | 100.0 [99.9, 100.0] |
| 32 | 46.2 [44.6, 47.7] | 62.3 [60.8, 63.8] | 75.9 [74.5, 77.2] | 89.7 [88.7, 90.6] | 100.0 [99.9, 100.0] |
| 64 | 41.9 [40.4, 43.4] | 55.5 [54.0, 57.0] | 69.3 [67.8, 70.7] | 80.7 [79.5, 81.9] | 92.4 [91.5, 93.2] |

### Zipf traffic, sigma=0.5

| N policies \ K slots | 2 | 4 | 8 | 16 | 32 |
|---:|---:|---:|---:|---:|---:|
| 8 | 59.9 [58.4, 61.4] | 82.2 [81.0, 83.3] | 100.0 [99.9, 100.0] | 100.0 [99.9, 100.0] | 100.0 [99.9, 100.0] |
| 16 | 52.7 [51.2, 54.2] | 69.5 [68.1, 70.9] | 87.7 [86.7, 88.7] | 100.0 [99.9, 100.0] | 100.0 [99.9, 100.0] |
| 32 | 44.3 [42.7, 45.8] | 61.5 [60.0, 63.0] | 76.4 [75.0, 77.6] | 89.4 [88.4, 90.3] | 100.0 [99.9, 100.0] |
| 64 | 41.9 [40.4, 43.4] | 54.1 [52.6, 55.6] | 68.1 [66.7, 69.5] | 80.4 [79.1, 81.6] | 92.1 [91.2, 92.9] |

## Partial override behavior

Across all cells, conflict accuracy is 49.0%, versus 70.8% for single-rule
requests. The gap is primarily a capacity effect: conflict accuracy rises from
12.9% at `K=2` to 86.3% at `K=32`, while single-rule accuracy rises from 38.6%
to 97.4%. Every public cell with `K >= N` reaches 100% accuracy. This indicates
that reverse-order residual compilation and exact condition/scope retrieval are
correct on the public nested scopes; the remaining errors when `K < N` are
mainly caused by discarding lower-ranked residual regions.

## Allocation and subgroup behavior

Zipf traffic performs much better than uniform traffic in the most compressed
regimes. Skewed traffic concentrates requests in fewer residual regions, which
favors selection by residual estimated value. Aggregate accuracy is 78.2% for
Zipf and 56.4% for uniform traffic. Noise has only a small effect: aggregate
accuracy is 67.5% at `sigma=0.0` and 67.0% at `sigma=0.5`. Hot, cold,
common-context, and rare-context accuracy are 79.2%, 55.9%, 63.9%, and 70.0%,
respectively. When the payload is full, the writer preserves the highest-value
residual regions and discards complete lower-value condition/action regions.

## Objective and information allocation

Training uses request-action cross-entropy as the main objective. The bridge
also uses an auxiliary action-reconstruction objective: a small shared decoder
reconstructs the selected policy's action lane from bytes 2–23 after the
signed-byte boundary. Mean squared error is computed over valid slots only and
weighted by `0.1`. The target is detached so the loss trains the quantized
action code and decoder without directly encouraging the action embeddings to
collapse.

The exact bytes preserve condition identity and residual scope. The learned
bytes preserve action information. Under insufficient capacity, the writer
discards entire lower-value residual regions rather than partially degrading
every stored rule. The decoder parameters are shared across examples and all
values of `K`; they do not store sample-dependent information.

The subgroup results match this allocation strategy. Hot accuracy (79.2%) is
substantially higher than cold accuracy (55.9%), showing that residual-value
ranking preferentially preserves high-traffic rules. The lower conflict
accuracy (49.0%, versus 70.8% for single-rule requests) reflects the extra
slots required by partial overrides. Once `K >= N`, exact condition/scope
routing and the learned action bytes reach 100% observed accuracy, so the main
failure below full capacity is omitted regions rather than corrupted stored
rules. The auxiliary objective itself is effectively neutral in the measured
ablation, improving aggregate accuracy by only 0.0055 percentage points.

## Prediction made before the holdout

- Predicted smallest `K` for 90% overall action accuracy: 64
- Functional family: monotone saturating residual-traffic coverage as a
  function of the compression ratio `K/N`.
- Fit or calculation: the nearest public scaling point is `N=64, K=32`, where
  `K/N=0.5` and Zipf accuracy is 92.1–92.4%. Holding that ratio constant at
  `N=128` gives `K=64`. At fixed `K/N=0.5`, uniform accuracy remains near
  71–73% while Zipf accuracy improves as `N` grows, so the public curves do not
  show a scaling penalty at this ratio.
- Expected effect of 128 policies: a fixed absolute `K` covers a smaller share
  of residual regions, so `K` should grow approximately with `N`.
- Expected effect of the partial-policy-rate change: reducing the rate from
  35% to 20% creates fewer overrides and less residual fragmentation, which
  should make conflict handling easier, although it also leaves more distinct
  base-condition regions competing for slots.
- Expected effect of `crossing` scopes: the writer's bitwise residual
  compilation is valid for arbitrary overlaps, but the unseen geometry may
  still cause a modest distribution-shift penalty.
- Expected effect of log-normal traffic and `sigma=1.0` noise: log-normal
  traffic is highly concentrated and should help top-k allocation, while the
  larger estimate noise can misrank valuable regions and reduce that benefit.
- Reason that the relationship can extrapolate: selection quality is governed
  mainly by residual traffic captured at a given `K/N`; exact condition and
  scope bytes make retrieval largely independent of `N` once a region is
  selected.

Machine-readable commitment: [artifacts/prediction.json](artifacts/prediction.json).

## Holdout

Holdout artifact: [artifacts/holdout.json](artifacts/holdout.json)

The holdout uses 128 policies, 20% partial policies, log-normal traffic,
`sigma=1.0` estimate noise, and unseen crossing scopes. Overall entries include
their 95% Wilson interval; subgroup entries are point estimates.

| K | Overall [95% CI] | Conflict | Single rule | Hot | Cold | Common context | Rare context |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 91.7 [90.9, 92.6] | 87.8 | 92.3 | 93.9 | 28.1 | 86.7 | 93.7 |
| 48 | 96.0 [95.4, 96.6] | 93.7 | 96.4 | 97.8 | 45.7 | 93.8 | 96.9 |
| 64 | 98.0 [97.5, 98.4] | 96.3 | 98.2 | 99.2 | 62.5 | 97.2 | 98.3 |
| 96 | 99.8 [99.6, 99.9] | 100.0 | 99.7 | 100.0 | 93.6 | 99.7 | 99.8 |
| 128 | 100.0 [99.9, 100.0] | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |

The smallest tested budget reaching 90% is `K=32`, with 3,758 correct
decisions out of 4,096. The committed prediction was `K=64`, so its absolute
error is 32 slots in the conservative direction. The prediction
underestimated how strongly log-normal traffic concentrates request mass and
how much the lower partial-policy rate reduces fragmentation. The unseen
crossing scopes did not break compilation: at `K=128`, where no region is
omitted, every request is correct.

Low-budget errors remain strongly concentrated in cold requests. At `K=32`,
cold accuracy is only 28.1% (38/135, 95% CI `[21.2, 36.3]`), compared with
93.9% for hot requests. This is the expected tradeoff of selecting regions by
estimated residual traffic. The cold subgroup is also small under log-normal
traffic, so its interval is substantially wider than the overall interval.

## Weaker approach or ablation

### Auxiliary reconstruction loss ablation

The selected model uses the `0.1`-weighted action-reconstruction MSE. It is
slightly better overall than the checkpoint trained without that objective,
but the difference is too small to establish a meaningful improvement:

| Variant | Overall | Conflict | Single rule | Hot | Cold | Common context | Rare context |
|---|---:|---:|---:|---:|---:|---:|---:|
| With auxiliary loss | **67.2644** | **49.0264** | **70.8239** | **79.2248** | 55.9377 | 63.8742 | **70.0013** |
| Without auxiliary loss | 67.2589 | 49.0189 | 70.8188 | 79.2128 | **55.9383** | **63.8762** | 69.9897 |

Values are percentages. The auxiliary model makes only 18 additional correct
predictions among 327,680 decisions, improves 25 cells, ties 36, and loses 19.
Its 95% Wilson interval for overall accuracy is `[67.1035, 67.4249]`, almost
identical to the no-auxiliary interval `[67.0980, 67.4194]`. Therefore the
measured benefit is effectively neutral. The no-auxiliary checkpoint also
predates the correction that aligned compression with the documented action
lane, so this is a practical variant comparison rather than a perfectly
isolated loss-only ablation. Full results are in
[`artifacts/sweep_no_auxiliary_loss.json`](artifacts/sweep_no_auxiliary_loss.json).

### Earlier packing baseline

The first packing approach used eight values for the exact residual-scope mask
and 16 learned values for the combined condition/action payload. Its aggregate
accuracy was 61.37%, compared with 67.26% for the selected packing. Its main
limit was lossy condition retrieval: even when `K >= N`, it did not reach 100%
accuracy. The full measurements are in
[`artifacts/sweep_old_packing.json`](artifacts/sweep_old_packing.json).

#### Uniform traffic, sigma=0.0

| N policies \ K slots | 2 | 4 | 8 | 16 | 32 |
|---:|---:|---:|---:|---:|---:|
| 8 | 40.3 [38.8, 41.8] | 69.6 [68.1, 70.9] | 93.0 [92.2, 93.8] | 94.3 [93.6, 95.0] | 94.5 [93.7, 95.1] |
| 16 | 19.8 [18.7, 21.1] | 37.4 [36.0, 38.9] | 67.5 [66.0, 68.9] | 88.2 [87.1, 89.1] | 88.2 [87.2, 89.2] |
| 32 | 9.5 [8.7, 10.5] | 19.5 [18.3, 20.7] | 34.6 [33.2, 36.1] | 60.3 [58.8, 61.8] | 77.1 [75.7, 78.3] |
| 64 | 5.3 [4.6, 6.0] | 10.3 [9.4, 11.2] | 16.9 [15.8, 18.1] | 32.3 [30.9, 33.7] | 47.4 [45.9, 49.0] |

#### Uniform traffic, sigma=0.5

| N policies \ K slots | 2 | 4 | 8 | 16 | 32 |
|---:|---:|---:|---:|---:|---:|
| 8 | 37.3 [35.8, 38.8] | 68.8 [67.4, 70.2] | 92.7 [91.9, 93.5] | 94.3 [93.6, 95.0] | 94.3 [93.6, 95.0] |
| 16 | 20.5 [19.3, 21.7] | 36.4 [34.9, 37.9] | 66.5 [65.1, 68.0] | 87.5 [86.5, 88.5] | 87.6 [86.6, 88.6] |
| 32 | 9.0 [8.2, 9.9] | 18.7 [17.5, 19.9] | 35.9 [34.5, 37.4] | 57.0 [55.5, 58.5] | 75.8 [74.5, 77.1] |
| 64 | 5.1 [4.5, 5.8] | 8.7 [7.9, 9.6] | 16.9 [15.8, 18.1] | 31.4 [30.0, 32.9] | 47.5 [46.0, 49.1] |

#### Zipf traffic, sigma=0.0

| N policies \ K slots | 2 | 4 | 8 | 16 | 32 |
|---:|---:|---:|---:|---:|---:|
| 8 | 61.3 [59.8, 62.8] | 79.6 [78.3, 80.8] | 94.9 [94.2, 95.5] | 95.0 [94.3, 95.6] | 95.2 [94.5, 95.8] |
| 16 | 51.0 [49.5, 52.5] | 67.3 [65.9, 68.7] | 81.6 [80.4, 82.7] | 90.4 [89.4, 91.2] | 90.3 [89.3, 91.1] |
| 32 | 45.4 [43.9, 47.0] | 60.1 [58.6, 61.6] | 72.1 [70.7, 73.4] | 79.4 [78.2, 80.6] | 83.4 [82.2, 84.5] |
| 64 | 41.6 [40.1, 43.1] | 54.2 [52.7, 55.7] | 64.3 [62.9, 65.8] | 72.5 [71.1, 73.9] | 77.0 [75.7, 78.2] |

#### Zipf traffic, sigma=0.5

| N policies \ K slots | 2 | 4 | 8 | 16 | 32 |
|---:|---:|---:|---:|---:|---:|
| 8 | 59.0 [57.5, 60.5] | 79.9 [78.6, 81.1] | 94.9 [94.2, 95.5] | 94.4 [93.6, 95.0] | 94.4 [93.7, 95.1] |
| 16 | 52.3 [50.8, 53.8] | 67.6 [66.1, 69.0] | 82.5 [81.3, 83.6] | 90.9 [90.0, 91.7] | 91.0 [90.1, 91.8] |
| 32 | 43.6 [42.0, 45.1] | 59.9 [58.4, 61.4] | 72.0 [70.6, 73.3] | 80.6 [79.3, 81.7] | 85.3 [84.2, 86.4] |
| 64 | 41.5 [40.0, 43.0] | 52.5 [50.9, 54.0] | 64.5 [63.0, 65.9] | 71.9 [70.6, 73.3] | 76.0 [74.7, 77.3] |

## Recommendation

For the stated goal of minimizing memory while reaching 90% overall accuracy,
I recommend `K=32`. It uses 768 bytes per example and achieves 91.75% overall
accuracy with a 95% Wilson interval of `[90.87%, 92.55%]`; even the lower bound
is above the target.

This recommendation is specific to an aggregate-accuracy objective under
skewed traffic. Its main failure condition is poor coverage of cold rules:
`K=32` achieves only 28.1% cold accuracy, as well as 87.8% conflict and 86.7%
common-context accuracy. Applications that require balanced subgroup coverage
should instead use `K=96`, which reaches 99.76% overall accuracy and 93.6% cold
accuracy. Other risks are noisier traffic estimates or a less concentrated
request distribution, both of which make top-k residual allocation less
reliable and may require a larger budget.
