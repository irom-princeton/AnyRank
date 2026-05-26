# Data from: A careful examination of large behavior models for multitask dexterous manipulation

This dataset contains the raw evaluation data used to generate the Compact Letter Display (CLD) plots in the paper. The paper rigorously evaluates multitask robot manipulation policies — Large Behavior Models (LBMs) obtained through multitask pretraining on approximately 1,700 hours of demonstration data — and compares them against single-task baselines through blind, randomized A/B trials in both simulation and the real world (1,800 real-world rollouts and over 47,000 simulation rollouts). Two performance metrics are used: success rate (SR), a binary outcome indicating whether the task was completed, and task completion (TC), a continuous measure of the fraction of task-specific milestones achieved per rollout. Each CSV file corresponds to one or more figure panels, recording per-rollout outcomes for comparing these policies. Statistical comparisons between methods use sequential A/B testing with Barnard's exact test, and the resulting Compact Letter Display (CLD) grouping letters are included (methods that share a letter are not statistically significantly different). For SR, violin plots in the paper represent Bayesian posteriors of success rates under a uniform Beta prior. For TC, violin plots represent the Bayesian posterior of the mean TC under a uniform Dirichlet prior. Policies not sharing the same CLD letters are statistically distinguishable at a 95 percent confidence level.

## Description of the data and file structure

The original Excel workbook has been converted to 20 individual CSV files (one per figure) to ensure accessibility and ease of reanalysis. All derived columns (counts, rates) are stored as plain values with no formulas.

### Panel naming convention

Each row's `Panel` value encodes the experimental setting:

```
Fig{Number}{Panel}_{Environment}_{TaskSet}_{Condition}[_Variant]
```

* **Environment:** `HW` (hardware / real robot) or `Sim` (simulation)
* **TaskSet:** `Seen` (tasks included in training data) or `Unseen` (held-out tasks not in training data)
* **Condition:** `Nominal` (standard evaluation conditions) or `DistShift` (distribution shift — altered object positions, lighting, or other environmental perturbations)
* **Variant** (optional): `Scaling` (data-scaling ablation), `Progress` (task-progress metric instead of binary success), `MeanProgress` (mean task progress metric)

### Methods

The following policy methods appear across the files:

| Method                                              | Description                                                                                                                                                                    |
| --------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Single Task                                         | A policy trained from scratch on a single task only (baseline)                                                                                                                 |
| LBM                                                 | Pretrained Large Behavior Model evaluated without any task-specific fine-tuning (also labeled "LBM zeroshot" in some panels). Appears in Fig. 2 only                           |
| LBM finetuned                                       | Large Behavior Model pretrained on the full dataset (OXE-Ramen and TRI-Ramen) and then fine-tuned on the target task                                                           |
| LBM Finetuned 15 % / 50 %                           | LBM fine-tuned with 15 % or 50 % of the available task-specific data (data-scaling ablation)                                                                                   |
| LBM finetuned \[TRI-Ramen]                          | LBM pretrained on TRI-Ramen data only (no OXE-Ramen), then fine-tuned (pretraining scaling experiment)                                                                         |
| LBM finetuned \[TRI-Ramen-50 %] / \[TRI-Ramen-25 %] | LBM pretrained on 50 % or 25 % of all tasks in the TRI-Ramen dataset, then fine-tuned (pretraining scaling experiment)                                                         |
| Single Task (DiT-FP) / LBM (DiT-FP) finetuned       | Variants using the Diffusion Transformer with flow matching (DiT-FP) architecture instead of the default UNet with diffusion denoising (supplementary architecture comparison) |
| LBM (UNet) BASE Mix / OXE Mix                       | UNet-based LBM variants pretrained with different data mixtures: BASE Mix (TRI-Ramen only) vs. OXE Mix (TRI-Ramen + OXE-Ramen) (supplementary architecture comparison)         |

### Column schemas

The CSV files use one of four column schemas, described below.

#### Schema A — Binary success/failure (8 columns)

Used by: `Fig2`, `Fig3`, `Fig4_AB`, `Fig5_A`, `FigS14`, `FigS15`, `FigS16`, `FigS32`, `FigS33`

| Column          | Type    | Description                                                                                                                                                                               |
| --------------- | ------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Panel           | string  | Figure panel identifier encoding the experimental setting (see naming convention above)                                                                                                   |
| Task            | string  | Manipulation task name (e.g., `PutKiwiInCenterOfTable`) or `Aggregate` for pooled results across tasks                                                                                    |
| Method          | string  | Policy method name (see Methods table above)                                                                                                                                              |
| Success/Failure | string  | JSON-style array of `True`/`False` values, one per rollout, indicating whether the rollout succeeded                                                                                      |
| Num\_Successes  | integer | Count of `True` entries in the Success/Failure array                                                                                                                                      |
| Num\_Rollouts   | integer | Total number of rollouts (length of the Success/Failure array)                                                                                                                            |
| Success\_Rate   | float   | Fraction of successful rollouts (`Num_Successes / Num_Rollouts`), range \[0, 1]                                                                                                           |
| CLD\_Letter     | string  | Compact Letter Display grouping letter(s) from sequential Barnard's test. Methods sharing a letter are not statistically significantly different; `a` indicates the best-performing group |

#### Schema B — Binary success/failure with data fraction (9 columns)

Used by: `Fig4_C`, `Fig5_B`

Same as Schema A with one additional column:

| Column   | Type  | Description                                                                                                                      |
| -------- | ----- | -------------------------------------------------------------------------------------------------------------------------------- |
| Fraction | float | Fraction of available task-specific fine-tuning data used (e.g., 0.0, 0.15, 0.5, 1.0). Used in data-scaling ablation experiments |

Column order: Panel, Fraction, Task, Method, Success/Failure, Num_Successes, Num_Rollouts, Success_Rate, CLD_Letter

#### Schema C — Task progress (9 columns)

Used by: `Fig4_DE`, `Fig5_C`, `Fig6`, `FigS28`, `FigS29`

| Column                  | Type    | Description                                                                                                                                                            |
| ----------------------- | ------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Panel                   | string  | Figure panel identifier (see naming convention above)                                                                                                                  |
| Task                    | string  | Manipulation task name or `Aggregate` for pooled results                                                                                                               |
| Method                  | string  | Policy method name (see Methods table above)                                                                                                                           |
| Task\_Progress\_Results | string  | JSON-style array of continuous values in \[0, 1], one per rollout, representing the fraction of task milestones completed                                              |
| Task\_Progress\_Bins    | string  | JSON-style array of milestone thresholds defining the discrete progress levels (e.g., `[0, 0.0833, 0.1667, ..., 1.0]`). The number of bins equals `Num_Milestones + 1` |
| Avg\_Task\_Progress     | float   | Mean of the Task\_Progress\_Results array, range \[0, 1]                                                                                                               |
| Num\_Rollouts           | integer | Total number of rollouts (length of the Task\_Progress\_Results array)                                                                                                 |
| Num\_Milestones         | integer | Number of discrete progress milestones for the task                                                                                                                    |
| CLD\_Letter             | string  | Compact Letter Display grouping letter(s) from sequential testing on task progress distributions                                                                       |

#### Schema D — Task progress with data fraction (10 columns)

Used by: `Fig4_F`, `Fig5_D`, `Fig7`

Same as Schema C with one additional column:

| Column   | Type  | Description                                                                                          |
| -------- | ----- | ---------------------------------------------------------------------------------------------------- |
| Fraction | float | Fraction of available task-specific fine-tuning data used. Used in data-scaling ablation experiments |

Column order: Panel, Fraction, Task, Method, Task_Progress_Results, Task_Progress_Bins, Avg_Task_Progress, Num_Rollouts, Num_Milestones, CLD_Letter

#### Schema E — Pre-aggregated binary (7 columns)

Used by: `FigS13`

Same as Schema A but without the raw `Success/Failure` array (only aggregated counts are provided):

Column order: Panel, Task, Method, Num_Successes, Num_Rollouts, Success_Rate, CLD_Letter

### File descriptions

| File          | Figure    | Schema | Description                                                                                                                                                                                                                                                                                                                                    |
| ------------- | --------- | ------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Fig2.csv`    | Fig. 2    | A      | LBM performance on "seen" tasks (tasks included in the pretraining dataset) in real-world and simulation. Panel (A) is under nominal conditions; panel (B) is under distribution shift. Compares Single Task, pretrained LBM, and finetuned LBM. Real-world tasks: 50 rollouts per task; simulation tasks: \~200 rollouts per task             |
| `Fig3.csv`    | Fig. 3    | A      | LBM performance on "unseen" simulation tasks from scenarios that are part of the simulation training set. Panel (A) nominal conditions; panel (B) distribution shift. Compares Single Task and finetuned LBM                                                                                                                                   |
| `Fig4_AB.csv` | Fig. 4A–B | A      | LBM performance on "unseen" tasks in real-world and simulation under nominal conditions (success rate). (A) Hardware tasks; (B) Simulation tasks (Kitchen scenario). Compares Single Task and finetuned LBM                                                                                                                                    |
| `Fig4_C.csv`  | Fig. 4C   | B      | Finetuning data-scaling ablation (success rate) on aggregate unseen simulation tasks under nominal conditions. Success rate as a function of finetuning dataset fraction                                                                                                                                                                       |
| `Fig4_DE.csv` | Fig. 4D–E | C      | Task completion on "unseen" tasks under nominal conditions. (D) Hardware tasks; (E) Simulation tasks (Kitchen scenario). Compares Single Task and finetuned LBM                                                                                                                                                                                |
| `Fig4_F.csv`  | Fig. 4F   | D      | Finetuning data-scaling ablation (task completion) on aggregate unseen simulation tasks under nominal conditions. Task completion as a function of finetuning dataset fraction                                                                                                                                                                 |
| `Fig5_A.csv`  | Fig. 5A   | A      | LBM performance on "unseen" simulation tasks under distribution shift (success rate). Compares Single Task and finetuned LBM on Kitchen scenario tasks                                                                                                                                                                                         |
| `Fig5_B.csv`  | Fig. 5B   | B      | Finetuning data-scaling ablation (success rate) on aggregate unseen simulation tasks under distribution shift                                                                                                                                                                                                                                  |
| `Fig5_C.csv`  | Fig. 5C   | C      | Task completion on "unseen" simulation tasks under distribution shift. Compares Single Task and finetuned LBM on Kitchen scenario tasks                                                                                                                                                                                                        |
| `Fig5_D.csv`  | Fig. 5D   | D      | Finetuning data-scaling ablation (task completion) on aggregate unseen simulation tasks under distribution shift                                                                                                                                                                                                                               |
| `Fig6.csv`    | Fig. 6    | C      | Data-scaling on the SetBreakfastTable real-world "unseen" task (229 demonstrations, 50 rollouts per policy, nominal conditions). Compares Single Task with LBM finetuned at 15 %, 50 %, and 100 % of task-specific data. Reports task completion                                                                                               |
| `Fig7.csv`    | Fig. 7    | D      | Pretraining scaling laws: LBM performance when pretraining with varying subsets of the full pretraining dataset, evaluated on five "unseen" simulation tasks under nominal conditions. Reports mean task completion as a function of finetuning dataset fraction for four pretraining data mixtures plus Single Task baseline                  |
| `FigS13.csv`  | Fig. S13  | E      | LBM performance as a function of training data. All LBMs use the UNet architecture. Reports success rate on "seen" and "unseen" simulation tasks (from scenarios in the training set) under nominal conditions and distribution shift, comparing different pre-training data mixtures. Pre-aggregated success counts (no raw per-rollout data) |
| `FigS14.csv`  | Fig. S14  | A      | Architecture comparison on "unseen" simulation tasks from scenarios that are part of the simulation training set. (A) Diffusion Transformer (DiT) with flow matching objective; (B) UNet with diffusion objective. Counterpart to Fig. 3 (which uses the default DiT architecture)                                                             |
| `FigS15.csv`  | Fig. S15  | A      | Architecture comparison on "unseen" simulation tasks from novel scenarios under nominal conditions. (A) DiT with flow matching; (B) UNet with diffusion. Counterpart to Fig. 4B                                                                                                                                                                |
| `FigS16.csv`  | Fig. S16  | A      | Architecture comparison on "unseen" simulation tasks from novel scenarios under distribution shift. (A) DiT with flow matching; (B) UNet with diffusion. Counterpart to Fig. 5A                                                                                                                                                                |
| `FigS28.csv`  | Fig. S28  | C      | Task completion (Bayesian posterior of mean TC) on "unseen" tasks in hardware and simulation under nominal conditions. Complements the full TC data distributions shown in Fig. 4D–E                                                                                                                                                           |
| `FigS29.csv`  | Fig. S29  | C      | Task completion (Bayesian posterior of mean TC) on "unseen" simulation tasks under distribution shift. Complements the full TC data distributions shown in Fig. 5C                                                                                                                                                                             |
| `FigS32.csv`  | Fig. S32  | A      | Analyzing the effect of low-motion data filtering in the pretraining dataset on "seen" simulation tasks under nominal conditions. (A) Pretrained LBM uses unfiltered data, single-task and finetuned LBM use filtered data; (B) all models use filtered data                                                                                   |
| `FigS33.csv`  | Fig. S33  | A      | Comparing pretrained LBMs with and without the data normalization error on "seen" simulation tasks. (A) Nominal conditions; (B) distribution shift. Adds the corrected pretrained LBM ("LBM Fixed") to the results in Fig. 2 for comparison                                                                                                    |

### Missing data

There are no missing data codes. All rollouts have valid outcomes. Each real-world task was evaluated with 50 rollouts per task per policy per condition. Simulation tasks were run approximately 200 times per task per policy per condition; due to missing data, a few tasks were analyzed with fewer than 200 rollouts. A `Fraction` value of `0.0` in scaling files indicates zero-shot evaluation (no task-specific fine-tuning data used).
