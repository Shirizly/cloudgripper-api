# Design Documents — Granular-Manipulation Refactor

Design-only artifacts (no code changes) for refactoring the chickpea/granular data-collection
stack. Written July 2026 by analyzing the code on branch `backgammon`, including the abandoned
mid-refactor state.

| Doc | Contents |
|---|---|
| [01_current_architecture.md](01_current_architecture.md) | The system as it exists today, traced from `main_chickpeas.py`: thread topology, component responsibilities, episode flow, dataset layout, and a numbered list of defects & mid-refactor debris (broken imports, missing files, races). Baseline for any refactor. |
| [02_proposed_architecture.md](02_proposed_architecture.md) | Target architecture: hardware / observation / perception / planning / execution / session / recording layers, swappable `Planner` protocol, single robot connection + single observation stream, event-driven episode state machine, `DryRunRobot` for hardware-free testing, migration map old→new. |
| [03_segmenter_native_design.md](03_segmenter_native_design.md) | Delta design assuming the accurate YOLOv11 segmenter (robot-in-view tolerant): removes all move-aside mask choreography, adds mask freshness rules, real-time lower-tool safety guard, adaptive wall sweeps, and online emission of the transition dataset (`masks_before/after` + push endpoints). This is the design for the never-committed `SegGranularPusher`. |

Reading order: 01 → 02 → 03. Doc 03 only describes deltas relative to 02.

Related pre-existing notes: `../MD files/transition_dataset_design.md` (target dataset format,
kept as the contract), `../MD files/ACTION_TRACKING_*.md` (action-tracking system the designs
build on).
