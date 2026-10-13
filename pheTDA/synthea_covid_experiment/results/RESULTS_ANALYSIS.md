# Synthea COVID benchmark analysis — redesigned

Generated: 2026-10-12T14:07:34+00:00

## Result coverage

The results directory contains **64 of 64** planned completed studies.

| Configuration | TPE budgets | Random budgets |
|---|---:|---:|
| Final phenotype only | 50, 100, 250, 500 | 50, 100, 250, 500 |
| Categorical initial phenotype only | 50, 100, 250, 500 | 50, 100, 250, 500 |
| Categorical initial + final alignment | 50, 100, 250, 500 | 50, 100, 250, 500 |
| Categorical initial mixing + final separation | 50, 100, 250, 500 | 50, 100, 250, 500 |
| Continuous initial phenotype only | 50, 100, 250, 500 | 50, 100, 250, 500 |
| Continuous initial + final alignment | 50, 100, 250, 500 | 50, 100, 250, 500 |
| Continuous initial mixing + final separation | 50, 100, 250, 500 | 50, 100, 250, 500 |
| Unsupervised reduction + clustering | 50, 100, 250, 500 | 50, 100, 250, 500 |

## Pareto-front comparison

### Hypervolume versus trial budget

**What the plot shows.** Each panel corresponds to one phenotype scenario. The x-axis is the number of trials and the y-axis is the hypervolume of the valid Pareto front found by that search. Solid blue lines represent TPE and dashed orange lines represent random search. At the same scenario and budget, the higher curve represents a front that is closer to the ideal objective values and/or spans a broader range of useful trade-offs.

**How it is computed.** Objective values are first converted to normalized benefits, so larger is always better. For observed lower and upper bounds $L_j$ and $U_j$ of objective $j$:

$$
z_j(x)=\begin{cases}\dfrac{x_j-L_j}{U_j-L_j}, & j\text{ is maximized},\\[6pt]\dfrac{U_j-x_j}{U_j-L_j}, & j\text{ is minimized}.\end{cases}
$$

The same bounds are used for both strategies and every available budget within a scenario. Constant objectives are removed because they cannot distinguish solutions. With reference point $\mathbf{0}$, hypervolume is the area (two objectives) or volume (three objectives) of the union of boxes dominated by the normalized Pareto points:

$$
HV(F)=\lambda_d\!\left(\bigcup_{x\in F}\prod_{j=1}^{d}[0,z_j(x)]\right),
\qquad \Delta HV=HV(F_{TPE})-HV(F_{Random}).
$$

Thus, higher hypervolume and positive $\Delta HV$ favor TPE. Hypervolume is comparable across strategies and budgets within a scenario, but it is not an absolute clinical quantity and should not be compared directly between different phenotype scenarios.

![Pareto hypervolume](analysis_plots_new/pareto_hypervolume.png)

**Descriptive conclusion.** Across 28 matched scenario/budget comparisons, TPE has higher hypervolume in **24**, random search in **4**, with **0** ties. The median TPE-minus-random hypervolume difference is **0.0596**.
The largest observed advantage for TPE is **0.3018** for **Continuous initial phenotype only**, budget **100**; the most negative difference is **-0.1131** for **Final phenotype only**, budget **50**.
TPE coverage exceeds reverse random-search coverage in **25 of 28** matched comparisons. TPE contributes a larger fraction of the pooled front in **27**, random search in **1**, with **0** equal contributions. These counts describe the observed runs and are not a statistical test of optimizer superiority.

### Combined-front contribution and set coverage

**What the table shows.** The hypervolume columns summarize the overall quality of each front. Contribution percentages show which search produced the unique solutions that remain after the two fronts are pooled. A solution found by both strategies, within a numerical tolerance, is classified as shared. The two coverage columns measure directional domination between the fronts.

**How it is computed.** Let $F_T$ and $F_R$ be the TPE and random Pareto fronts and $ND(\cdot)$ return the non-dominated subset. The combined front is:

$$
F_C=ND(F_T\cup F_R),\qquad p_T=\frac{|F_C^{TPE\ only}|}{|F_C|},\quad p_R=\frac{|F_C^{Random\ only}|}{|F_C|}.
$$

After orienting all objectives as benefits, $a$ dominates $b$ when it is no worse in every objective and strictly better in at least one. Set coverage is:

$$
C(T,R)=\frac{|\{r\in F_R:\exists t\in F_T,\ t\succ r\}|}{|F_R|}.
$$

$C(T,R)=1$ means every random-front point is dominated by at least one TPE point; zero means none is. $C(T,R)$ and $C(R,T)$ are calculated separately and are not complements.

| Configuration | Budget | TPE HV | Random HV | Δ HV | TPE-only contribution | Random-only contribution | Shared | C(TPE, Random) | C(Random, TPE) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Final phenotype only | 50 | 0.2981 | 0.4112 | -0.1131 | 25.0% | 75.0% | 0.0% | 0.000 | 0.600 |
| Final phenotype only | 100 | 0.4247 | 0.3851 | 0.0397 | 66.7% | 33.3% | 0.0% | 0.571 | 0.000 |
| Final phenotype only | 250 | 0.5568 | 0.4276 | 0.1292 | 81.8% | 18.2% | 0.0% | 0.667 | 0.000 |
| Final phenotype only | 500 | 0.6139 | 0.4949 | 0.1190 | 82.4% | 17.6% | 0.0% | 0.625 | 0.059 |
| Categorical initial phenotype only | 50 | 0.2091 | 0.1932 | 0.0159 | 66.7% | 22.2% | 11.1% | 0.250 | 0.125 |
| Categorical initial phenotype only | 100 | 0.2241 | 0.1751 | 0.0490 | 62.5% | 37.5% | 0.0% | 0.500 | 0.286 |
| Categorical initial phenotype only | 250 | 0.3233 | 0.1946 | 0.1287 | 77.8% | 11.1% | 11.1% | 0.667 | 0.000 |
| Categorical initial phenotype only | 500 | 0.3862 | 0.2752 | 0.1109 | 66.7% | 33.3% | 0.0% | 0.625 | 0.000 |
| Categorical initial + final alignment | 50 | 0.0917 | 0.0306 | 0.0612 | 81.8% | 18.2% | 0.0% | 0.333 | 0.000 |
| Categorical initial + final alignment | 100 | 0.0403 | 0.0531 | -0.0128 | 65.0% | 35.0% | 0.0% | 0.300 | 0.316 |
| Categorical initial + final alignment | 250 | 0.2032 | 0.1341 | 0.0691 | 65.0% | 30.0% | 5.0% | 0.125 | 0.176 |
| Categorical initial + final alignment | 500 | 0.2535 | 0.2355 | 0.0180 | 76.9% | 19.2% | 3.8% | 0.538 | 0.083 |
| Categorical initial mixing + final separation | 50 | 0.2712 | 0.2420 | 0.0292 | 73.7% | 21.1% | 5.3% | 0.500 | 0.000 |
| Categorical initial mixing + final separation | 100 | 0.3725 | 0.3052 | 0.0673 | 65.9% | 31.7% | 2.4% | 0.440 | 0.034 |
| Categorical initial mixing + final separation | 250 | 0.4285 | 0.3709 | 0.0576 | 76.3% | 20.3% | 3.4% | 0.533 | 0.020 |
| Categorical initial mixing + final separation | 500 | 0.4649 | 0.4669 | -0.0020 | 68.6% | 29.4% | 2.0% | 0.333 | 0.302 |
| Continuous initial phenotype only | 50 | 0.1752 | 0.1593 | 0.0160 | 55.6% | 33.3% | 11.1% | 0.333 | 0.250 |
| Continuous initial phenotype only | 100 | 0.5610 | 0.2592 | 0.3018 | 100.0% | 0.0% | 0.0% | 1.000 | 0.000 |
| Continuous initial phenotype only | 250 | 0.5073 | 0.2704 | 0.2368 | 71.4% | 28.6% | 0.0% | 0.714 | 0.167 |
| Continuous initial phenotype only | 500 | 0.6064 | 0.3864 | 0.2201 | 77.8% | 22.2% | 0.0% | 0.714 | 0.000 |
| Continuous initial + final alignment | 50 | 0.1542 | 0.0832 | 0.0710 | 71.4% | 28.6% | 0.0% | 0.600 | 0.000 |
| Continuous initial + final alignment | 100 | 0.1353 | 0.1674 | -0.0322 | 64.3% | 35.7% | 0.0% | 0.643 | 0.308 |
| Continuous initial + final alignment | 250 | 0.2199 | 0.0959 | 0.1240 | 89.5% | 10.5% | 0.0% | 0.818 | 0.000 |
| Continuous initial + final alignment | 500 | 0.4407 | 0.2526 | 0.1882 | 85.0% | 15.0% | 0.0% | 0.769 | 0.000 |
| Continuous initial mixing + final separation | 50 | 0.3333 | 0.2752 | 0.0581 | 54.5% | 18.2% | 27.3% | 0.643 | 0.000 |
| Continuous initial mixing + final separation | 100 | 0.3550 | 0.3472 | 0.0078 | 54.2% | 37.5% | 8.3% | 0.389 | 0.167 |
| Continuous initial mixing + final separation | 250 | 0.4357 | 0.4142 | 0.0215 | 65.9% | 34.1% | 0.0% | 0.462 | 0.129 |
| Continuous initial mixing + final separation | 500 | 0.5544 | 0.4057 | 0.1487 | 91.7% | 8.3% | 0.0% | 0.914 | 0.143 |

### Raw Pareto-front views

**What the plots show.** In two-objective scenarios, every dot is an actual valid Pareto-optimal Optuna trial and the step line traces the discrete front. In three-objective scenarios, each polyline is one Pareto solution crossing the initial-entropy, final-entropy, and silhouette axes. All three axes show the normalized benefit $z_j(x)$, so moving upward is always better. The thicker polyline is the selected single solution.

**How they are computed.** The two-dimensional staircase is obtained by sorting the observed Pareto points by the x objective; its horizontal and vertical segments are only visual guides and do not represent unobserved trials. Parallel-coordinate positions use the same direction-aware normalization equation as hypervolume. Dominance is defined by:

$$
a\succ b\iff [z_j(a)\ge z_j(b)\ \forall j]\ \land\ [z_k(a)>z_k(b)\ \text{for at least one }k].
$$

#### Final phenotype only

2D scatter/staircase.

![Final phenotype only Pareto front](analysis_plots_new/pareto_front__final_only.png)

**Descriptive conclusion.** Across 4 matched budgets, TPE has higher hypervolume in 3, random search in 1, and they tie in 0. Use the plotted front to see whether this difference comes from uniformly better solutions or from different trade-off regions.

#### Categorical initial phenotype only

2D scatter/staircase.

![Categorical initial phenotype only Pareto front](analysis_plots_new/pareto_front__categorical_initial_only.png)

**Descriptive conclusion.** Across 4 matched budgets, TPE has higher hypervolume in 4, random search in 0, and they tie in 0. Use the plotted front to see whether this difference comes from uniformly better solutions or from different trade-off regions.

#### Categorical initial + final alignment

parallel coordinates.

![Categorical initial + final alignment Pareto front](analysis_plots_new/pareto_front__categorical_both_alignment.png)

**Descriptive conclusion.** Across 4 matched budgets, TPE has higher hypervolume in 3, random search in 1, and they tie in 0. Use the plotted front to see whether this difference comes from uniformly better solutions or from different trade-off regions.

#### Categorical initial mixing + final separation

parallel coordinates.

![Categorical initial mixing + final separation Pareto front](analysis_plots_new/pareto_front__categorical_initial_mixing_final_separation.png)

**Descriptive conclusion.** Across 4 matched budgets, TPE has higher hypervolume in 3, random search in 1, and they tie in 0. Use the plotted front to see whether this difference comes from uniformly better solutions or from different trade-off regions.

#### Continuous initial phenotype only

2D scatter/staircase.

![Continuous initial phenotype only Pareto front](analysis_plots_new/pareto_front__continuous_initial_only.png)

**Descriptive conclusion.** Across 4 matched budgets, TPE has higher hypervolume in 4, random search in 0, and they tie in 0. Use the plotted front to see whether this difference comes from uniformly better solutions or from different trade-off regions.

#### Continuous initial + final alignment

parallel coordinates.

![Continuous initial + final alignment Pareto front](analysis_plots_new/pareto_front__continuous_both_alignment.png)

**Descriptive conclusion.** Across 4 matched budgets, TPE has higher hypervolume in 3, random search in 1, and they tie in 0. Use the plotted front to see whether this difference comes from uniformly better solutions or from different trade-off regions.

#### Continuous initial mixing + final separation

parallel coordinates.

![Continuous initial mixing + final separation Pareto front](analysis_plots_new/pareto_front__continuous_initial_mixing_final_separation.png)

**Descriptive conclusion.** Across 4 matched budgets, TPE has higher hypervolume in 4, random search in 0, and they tie in 0. Use the plotted front to see whether this difference comes from uniformly better solutions or from different trade-off regions.

## Runtime versus trial budget

**What the plot shows.** The left panel contains pheTDA studies. Each thin line is one phenotype scenario; the thick lines summarize the scenario-specific runtimes at every budget, and their shaded regions show the interquartile range across scenarios. The right panel shows the phenotype-independent dimensionality-reduction-plus-clustering baseline on its own scale. Solid blue lines are TPE and dashed orange lines are random search. A logarithmic y-axis is used automatically when the positive runtimes span at least a factor of ten.

**How it is computed.** Active runtime excludes interruption downtime and shared preprocessing, but includes optimization and selected-solution postprocessing:

$$
R_{s,b,m}=\frac{\text{active execution seconds}_{s,b,m}}{3600}.
$$

For search method $m$ and budget $b$, the thick pheTDA line is the median of $R_{s,b,m}$ across the available phenotype scenarios $s$. The band is $[Q_{0.25},Q_{0.75}]$; it describes variation between scenarios, not a confidence interval. The paired runtime ratio is:

$$
\rho_{s,b}=\frac{R_{s,b,TPE}}{R_{s,b,Random}}.
$$

$\rho<1$ means TPE was faster, $\rho>1$ means random search was faster, and $\rho=1$ means equal recorded runtime.

![Runtime versus trial budget](analysis_plots_new/runtime_by_trials.png)

**Descriptive conclusion.** TPE is faster in **20 of 32** matched comparisons, random search is faster in **12**, and **0** are equal within numerical tolerance. The overall median TPE/random runtime ratio is **0.811** (pheTDA median ratio **0.738**; unsupervised-baseline median ratio **1.930**).
Runtime differences reflect which hyperparameters each search sampled as well as sampler overhead; they should be interpreted together with Pareto quality rather than as a stand-alone measure of performance.

| Configuration | Budget | TPE runtime (h) | Random runtime (h) | TPE/random ratio |
|---|---:|---:|---:|---:|
| Final phenotype only | 50 | 1.2002 | 1.4588 | 0.823 |
| Final phenotype only | 100 | 2.3593 | 2.6577 | 0.888 |
| Final phenotype only | 250 | 6.3281 | 4.4364 | 1.426 |
| Final phenotype only | 500 | 6.1077 | 9.9255 | 0.615 |
| Categorical initial phenotype only | 50 | 0.9205 | 0.8179 | 1.126 |
| Categorical initial phenotype only | 100 | 1.4530 | 1.8168 | 0.800 |
| Categorical initial phenotype only | 250 | 4.8709 | 4.6076 | 1.057 |
| Categorical initial phenotype only | 500 | 7.4286 | 8.5560 | 0.868 |
| Categorical initial + final alignment | 50 | 0.7588 | 0.7256 | 1.046 |
| Categorical initial + final alignment | 100 | 2.1619 | 1.4863 | 1.455 |
| Categorical initial + final alignment | 250 | 3.2269 | 3.1353 | 1.029 |
| Categorical initial + final alignment | 500 | 5.1870 | 5.8051 | 0.894 |
| Categorical initial mixing + final separation | 50 | 0.9129 | 0.5765 | 1.584 |
| Categorical initial mixing + final separation | 100 | 1.5091 | 1.4310 | 1.055 |
| Categorical initial mixing + final separation | 250 | 3.5243 | 5.3010 | 0.665 |
| Categorical initial mixing + final separation | 500 | 8.6156 | 11.9920 | 0.718 |
| Continuous initial phenotype only | 50 | 0.6214 | 1.1398 | 0.545 |
| Continuous initial phenotype only | 100 | 1.0692 | 1.8008 | 0.594 |
| Continuous initial phenotype only | 250 | 2.6177 | 5.5569 | 0.471 |
| Continuous initial phenotype only | 500 | 5.3784 | 9.7866 | 0.550 |
| Continuous initial + final alignment | 50 | 0.3849 | 1.4419 | 0.267 |
| Continuous initial + final alignment | 100 | 1.6992 | 3.4283 | 0.496 |
| Continuous initial + final alignment | 250 | 2.7463 | 4.8908 | 0.562 |
| Continuous initial + final alignment | 500 | 6.1358 | 10.5941 | 0.579 |
| Continuous initial mixing + final separation | 50 | 0.5282 | 0.8258 | 0.640 |
| Continuous initial mixing + final separation | 100 | 1.2223 | 2.2226 | 0.550 |
| Continuous initial mixing + final separation | 250 | 3.0504 | 4.7793 | 0.638 |
| Continuous initial mixing + final separation | 500 | 7.4191 | 9.7816 | 0.758 |
| Unsupervised reduction + clustering | 50 | 0.7839 | 0.5214 | 1.504 |
| Unsupervised reduction + clustering | 100 | 3.9341 | 1.8588 | 2.117 |
| Unsupervised reduction + clustering | 250 | 9.4048 | 5.3937 | 1.744 |
| Unsupervised reduction + clustering | 500 | 23.1362 | 10.1448 | 2.281 |

## Objectives of the selected solution

**What the heatmaps show.** The panels are silhouette, initial graph entropy, and final graph entropy. Rows are applicable configurations and columns are trial budgets. Each cell compares the raw objective of the single solution selected from the TPE front with the corresponding random-search solution. Blue cells favor TPE, red cells favor random search, values near zero are similar, and an em dash means that a matched comparison is unavailable. Because the unsupervised baseline optimizes only silhouette, it appears only in the first panel.

**How it is computed.** The displayed difference is oriented so that a positive value always favors TPE:

$$
A_{s,b,j}=\begin{cases}x^{TPE}_{s,b,j}-x^{Random}_{s,b,j}, & j\text{ is maximized},\\[6pt]x^{Random}_{s,b,j}-x^{TPE}_{s,b,j}, & j\text{ is minimized}.\end{cases}
$$

The color scale is centered at zero and is calculated independently for each objective because entropy and silhouette have different numerical scales. These are raw objective differences, not standardized effect sizes. The selected pheTDA solution minimizes its normalized distance to the study-specific ideal point, so it is a compromise rather than the independently best trial for every objective.

![Direction-adjusted selected objectives](analysis_plots_new/selected_objective_advantage.png)

**Descriptive conclusion.**
- **Silhouette:** TPE is better in **18 of 32** matched cells, random search in **10**, with **4** ties. The largest TPE advantage is **0.4172** for Continuous initial mixing + final separation at budget 100; the most negative value is **-0.2498** for Categorical initial + final alignment at budget 250.
- **Initial graph entropy:** TPE is better in **16 of 24** matched cells, random search in **6**, with **2** ties. The largest TPE advantage is **0.6710** for Continuous initial phenotype only at budget 250; the most negative value is **-0.1025** for Continuous initial + final alignment at budget 100.
- **Final graph entropy:** TPE is better in **11 of 20** matched cells, random search in **8**, with **1** ties. The largest TPE advantage is **0.0675** for Categorical initial + final alignment at budget 250; the most negative value is **-0.0903** for Categorical initial + final alignment at budget 500.
A win on one selected objective can be accompanied by a loss on another because the pheTDA solution is selected as a multi-objective compromise.

### Raw selected-objective values

| Configuration | Strategy | Budget | Silhouette | Initial entropy | Final entropy |
|---|---|---:|---:|---:|---:|
| Final phenotype only | TPE pheTDA | 50 | 0.3987 | — | 0.8767 |
| Final phenotype only | TPE pheTDA | 100 | 0.3170 | — | 0.8425 |
| Final phenotype only | TPE pheTDA | 250 | 0.3530 | — | 0.8253 |
| Final phenotype only | TPE pheTDA | 500 | 0.1945 | — | 0.7963 |
| Final phenotype only | Random pheTDA | 50 | 0.0738 | — | 0.8373 |
| Final phenotype only | Random pheTDA | 100 | 0.3581 | — | 0.8653 |
| Final phenotype only | Random pheTDA | 250 | 0.0051 | — | 0.8150 |
| Final phenotype only | Random pheTDA | 500 | 0.0243 | — | 0.7913 |
| Categorical initial phenotype only | TPE pheTDA | 50 | 0.1782 | 1.5326 | — |
| Categorical initial phenotype only | TPE pheTDA | 100 | 0.0834 | 1.5351 | — |
| Categorical initial phenotype only | TPE pheTDA | 250 | 0.3527 | 1.5091 | — |
| Categorical initial phenotype only | TPE pheTDA | 500 | 0.3527 | 1.5153 | — |
| Categorical initial phenotype only | Random pheTDA | 50 | 0.1782 | 1.5326 | — |
| Categorical initial phenotype only | Random pheTDA | 100 | -0.0431 | 1.5082 | — |
| Categorical initial phenotype only | Random pheTDA | 250 | 0.1415 | 1.5239 | — |
| Categorical initial phenotype only | Random pheTDA | 500 | 0.3497 | 1.5420 | — |
| Categorical initial + final alignment | TPE pheTDA | 50 | -0.0625 | 1.5021 | 0.8055 |
| Categorical initial + final alignment | TPE pheTDA | 100 | -0.1312 | 1.5183 | 0.8442 |
| Categorical initial + final alignment | TPE pheTDA | 250 | -0.0108 | 1.4724 | 0.7598 |
| Categorical initial + final alignment | TPE pheTDA | 500 | 0.1423 | 1.4733 | 0.7904 |
| Categorical initial + final alignment | Random pheTDA | 50 | -0.2415 | 1.5307 | 0.8307 |
| Categorical initial + final alignment | Random pheTDA | 100 | 0.0107 | 1.5265 | 0.8309 |
| Categorical initial + final alignment | Random pheTDA | 250 | 0.2389 | 1.5161 | 0.8273 |
| Categorical initial + final alignment | Random pheTDA | 500 | -0.2380 | 1.3876 | 0.7002 |
| Categorical initial mixing + final separation | TPE pheTDA | 50 | 0.3070 | 1.5795 | 0.8785 |
| Categorical initial mixing + final separation | TPE pheTDA | 100 | 0.4842 | 1.5769 | 0.8861 |
| Categorical initial mixing + final separation | TPE pheTDA | 250 | 0.4383 | 1.5765 | 0.8647 |
| Categorical initial mixing + final separation | TPE pheTDA | 500 | 0.3646 | 1.5624 | 0.8423 |
| Categorical initial mixing + final separation | Random pheTDA | 50 | 0.0666 | 1.5751 | 0.8727 |
| Categorical initial mixing + final separation | Random pheTDA | 100 | 0.1945 | 1.5579 | 0.8772 |
| Categorical initial mixing + final separation | Random pheTDA | 250 | 0.4773 | 1.5772 | 0.8894 |
| Categorical initial mixing + final separation | Random pheTDA | 500 | 0.3646 | 1.5624 | 0.8423 |
| Continuous initial phenotype only | TPE pheTDA | 50 | 0.3527 | 4.1443 | — |
| Continuous initial phenotype only | TPE pheTDA | 100 | 0.0892 | 3.3508 | — |
| Continuous initial phenotype only | TPE pheTDA | 250 | 0.2158 | 3.4925 | — |
| Continuous initial phenotype only | TPE pheTDA | 500 | 0.3527 | 3.5700 | — |
| Continuous initial phenotype only | Random pheTDA | 50 | 0.1637 | 4.1046 | — |
| Continuous initial phenotype only | Random pheTDA | 100 | 0.1945 | 3.8968 | — |
| Continuous initial phenotype only | Random pheTDA | 250 | 0.2956 | 4.1635 | — |
| Continuous initial phenotype only | Random pheTDA | 500 | 0.1678 | 3.7865 | — |
| Continuous initial + final alignment | TPE pheTDA | 50 | 0.0282 | 3.6643 | 0.7962 |
| Continuous initial + final alignment | TPE pheTDA | 100 | -0.0116 | 3.7454 | 0.7929 |
| Continuous initial + final alignment | TPE pheTDA | 250 | -0.0233 | 3.4598 | 0.7920 |
| Continuous initial + final alignment | TPE pheTDA | 500 | 0.3527 | 3.4898 | 0.8001 |
| Continuous initial + final alignment | Random pheTDA | 50 | 0.1465 | 3.8665 | 0.8452 |
| Continuous initial + final alignment | Random pheTDA | 100 | -0.0024 | 3.6429 | 0.8169 |
| Continuous initial + final alignment | Random pheTDA | 250 | 0.0907 | 3.8679 | 0.8260 |
| Continuous initial + final alignment | Random pheTDA | 500 | 0.1668 | 3.7325 | 0.8264 |
| Continuous initial mixing + final separation | TPE pheTDA | 50 | 0.3727 | 4.3529 | 0.8661 |
| Continuous initial mixing + final separation | TPE pheTDA | 100 | 0.4461 | 4.3541 | 0.8732 |
| Continuous initial mixing + final separation | TPE pheTDA | 250 | 0.3533 | 4.3375 | 0.8614 |
| Continuous initial mixing + final separation | TPE pheTDA | 500 | 0.3438 | 4.2996 | 0.8041 |
| Continuous initial mixing + final separation | Random pheTDA | 50 | 0.3724 | 4.3074 | 0.8874 |
| Continuous initial mixing + final separation | Random pheTDA | 100 | 0.0289 | 4.1432 | 0.8541 |
| Continuous initial mixing + final separation | Random pheTDA | 250 | 0.5166 | 4.3247 | 0.8815 |
| Continuous initial mixing + final separation | Random pheTDA | 500 | 0.2415 | 4.3390 | 0.8614 |
| Unsupervised reduction + clustering | Unsupervised TPE baseline | 50 | 0.5333 | — | — |
| Unsupervised reduction + clustering | Unsupervised TPE baseline | 100 | 0.5333 | — | — |
| Unsupervised reduction + clustering | Unsupervised TPE baseline | 250 | 0.5333 | — | — |
| Unsupervised reduction + clustering | Unsupervised TPE baseline | 500 | 0.5333 | — | — |
| Unsupervised reduction + clustering | Unsupervised random baseline | 50 | 0.5329 | — | — |
| Unsupervised reduction + clustering | Unsupervised random baseline | 100 | 0.5333 | — | — |
| Unsupervised reduction + clustering | Unsupervised random baseline | 250 | 0.5333 | — | — |
| Unsupervised reduction + clustering | Unsupervised random baseline | 500 | 0.5329 | — | — |

## Interpretation cautions

- Hypervolume normalization bounds are calculated from the available Pareto results and are fixed across strategies and budgets within each scenario when this report is generated. Adding studies can change those bounds, so regenerate the complete report before the final comparison.
- A large combined-front contribution can be influenced by many nearby solutions. Interpret it together with hypervolume and set coverage.
- Separate budget studies are not prefixes of one common optimization run.
- One seed per condition supports a descriptive comparison only. Repeated seeds are needed for uncertainty intervals or formal claims.
- Runtime comparisons require the same machine and similar system load.

## Regeneration

Run from the repository root:

```text
python -m synthea_covid_experiment.analyze_results_new --results-directory synthea_covid_experiment/results
```
