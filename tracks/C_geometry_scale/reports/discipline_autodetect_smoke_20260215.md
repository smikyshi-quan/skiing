# Discipline Auto-Detection Smoke Check (2026-02-15)

## Auto-Detected Run (no --discipline)
- discipline: `slalom`
- discipline_source: `auto_detected`
- detection: `{'discipline': 'slalom', 'gate_count': 9, 'median_gap_px': 29.957855224609375, 'median_gap_ratio': 0.062412198384602866, 'rule': 'gate_count>=6'}`
- gate_spacing_m: `9.5` (discipline_default)

## Explicit Discipline Run (--discipline giant_slalom)
- discipline: `giant_slalom`
- discipline_source: `explicit`
- detection: `None`
- gate_spacing_m: `27.0` (discipline_default)

## Auto-Detected Stabilized Run (no --discipline, --stabilize)
- example output: `Auto-detected: slalom (8 gates visible, median gap 30px = 6.2% of frame height)`
- output JSON: `/tmp/cgeom_autodisc_stab_smoke/2909_1765738725(Video in Original Quality)_analysis.json`
- discipline: `slalom`
- gate_spacing_m: `9.5` (discipline_default)

## Notes
- Auto-detection was applied only when discipline was omitted.
- Explicit discipline bypassed auto-detection and still used discipline-aware default spacing when gate spacing was omitted.
