# Fallback Ablation Report (2026-02-20)

## Scope
- Compares Tier-1/2/3 base resolution usage and temporal jitter proxies.
- Validates geometry check flags and emission log-probability constraints.
- Uses BEV from Track C when present; otherwise uses deterministic local BEV stub.

## Aggregate
- Clips processed: 1
- Detections total: 7689
- Tier 1 count: 136
- Tier 2 count: 2588
- Tier 3 count: 4965
- Geometry check failures: 72
- Clips with stub BEV usage: 1

## Jitter Comparison
- Tier-2 jitter std median: 108.8655
- Tier-3 jitter std median: 69.6393
- Tier-3 counterfactual on Tier-2 events (median std): 192.6240
- Pass criterion (Tier-2 lower jitter than Tier-3 baseline): PASS
- Detail: Tier-2 median jitter std (108.866) < Tier-3 counterfactual median jitter std (192.624).

## Per-clip Summary
### mmexport1704089261026
- Frames written: 798
- Detections: 7689
- Tier counts: {'1': 136, '2': 2588, '3': 4965}
- Geometry failures: 72
- Jitter std: {'tier2': 108.86551553609566, 'tier3': 69.63930796905522, 'tier2_counterfactual_tier3': 192.6239810725591}
- Output JSON: `/tmp/unseen_test/detections/mmexport1704089261026_detections.json`
