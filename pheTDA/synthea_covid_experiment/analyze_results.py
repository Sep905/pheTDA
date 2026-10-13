"""Generate the redesigned Synthea benchmark report without changing the old one."""

import argparse
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from .analyze_results import (
    BASELINE_STRATEGIES,
    RANDOM_BASELINE_STRATEGY,
    SCENARIO_LABELS,
    STRATEGIES,
    STRATEGY_LABELS,
    TPE_BASELINE_STRATEGY,
    _format_value,
    _load_studies,
    _metric_spec,
)
from .config import SCENARIOS, TRIAL_BUDGETS, UNSUPERVISED_BASELINE_NAME

plt.switch_backend("Agg")


BUDGETS = tuple(budget for budget, _startup in TRIAL_BUDGETS)
SEARCH_COLORS = {"tpe": "#1f77b4", "random": "#ff7f0e"}
SEARCH_LINESTYLES = {"tpe": "-", "random": "--"}
METRIC_LABELS = {
    "stratification_silhouette": "Silhouette",
    "initial_graph_entropy": "Initial graph entropy",
    "final_graph_entropy": "Final graph entropy",
}
SELECTED_METRICS = (
    "stratification_silhouette",
    "initial_graph_entropy",
    "final_graph_entropy",
)
DOMINANCE_TOLERANCE = 1e-12


def _finite_values(frame, metric_names):
    if frame.empty or any(metric not in frame for metric in metric_names):
        return np.empty((0, len(metric_names)), dtype=float)
    values = frame.loc[:, metric_names].apply(pd.to_numeric, errors="coerce")
    values = values.to_numpy(dtype=float)
    return values[np.isfinite(values).all(axis=1)]


def _normalization_bounds(studies, scenario, metric_names):
    arrays = []
    for strategy in STRATEGIES:
        for budget in BUDGETS:
            study = studies.get((strategy, scenario.name, budget))
            if study is None or study.get("status") != "complete":
                continue
            values = _finite_values(study["pareto"], metric_names)
            if len(values):
                arrays.append(values)
    if not arrays:
        return None
    combined = np.vstack(arrays)
    return combined.min(axis=0), combined.max(axis=0)


def _normalize_as_benefit(values, directions, bounds, drop_constant=True):
    """Normalize objectives to [0, 1], where larger always means better."""
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    minima, maxima = bounds
    spans = maxima - minima
    informative = spans > DOMINANCE_TOLERANCE
    if drop_constant and not informative.any():
        return np.ones((len(values), 1), dtype=float)
    retained = informative if drop_constant else np.ones(len(spans), dtype=bool)
    retained_spans = spans[retained].copy()
    retained_spans[retained_spans <= DOMINANCE_TOLERANCE] = 1.0
    normalized = (values[:, retained] - minima[retained]) / retained_spans
    retained_directions = [
        direction for direction, keep in zip(directions, retained) if keep
    ]
    for column, direction in enumerate(retained_directions):
        if direction == "minimize":
            normalized[:, column] = 1.0 - normalized[:, column]
    if not drop_constant:
        normalized[:, ~informative] = 1.0
    return np.clip(normalized, 0.0, 1.0)


def _dominates(first, second):
    return np.all(first >= second - DOMINANCE_TOLERANCE) and np.any(
        first > second + DOMINANCE_TOLERANCE
    )


def _nondominated_indices(benefit_values):
    values = np.asarray(benefit_values, dtype=float)
    keep = []
    for candidate_index, candidate in enumerate(values):
        dominated = any(
            other_index != candidate_index and _dominates(other, candidate)
            for other_index, other in enumerate(values)
        )
        if not dominated:
            keep.append(candidate_index)
    return np.asarray(keep, dtype=int)


def _hypervolume_2d(points):
    """Exact union area of origin-anchored rectangles."""
    maximum_y_by_x = {}
    for x_value, y_value in points:
        maximum_y_by_x[x_value] = max(maximum_y_by_x.get(x_value, 0.0), y_value)
    x_values = np.asarray(sorted(maximum_y_by_x), dtype=float)
    y_values = np.asarray([maximum_y_by_x[x_value] for x_value in x_values])
    suffix_maximum = np.maximum.accumulate(y_values[::-1])[::-1]
    widths = np.diff(np.concatenate(([0.0], x_values)))
    return float(np.sum(widths * suffix_maximum))


def _hypervolume(points):
    """Exact hypervolume for one-, two-, or three-objective normalized fronts."""
    points = np.asarray(points, dtype=float)
    if points.size == 0:
        return 0.0
    points = np.clip(points, 0.0, 1.0)
    points = np.unique(points, axis=0)
    points = points[_nondominated_indices(points)]
    dimensions = points.shape[1]
    if dimensions == 1:
        return float(points[:, 0].max())
    if dimensions == 2:
        return _hypervolume_2d(points)
    if dimensions != 3:
        raise ValueError("hypervolume is implemented for at most three objectives")

    volume = 0.0
    previous_x = 0.0
    for x_value in np.unique(points[:, 0]):
        active = points[points[:, 0] >= x_value - DOMINANCE_TOLERANCE, 1:]
        volume += float(x_value - previous_x) * _hypervolume_2d(active)
        previous_x = float(x_value)
    return float(volume)


def _set_coverage(source, target):
    if not len(target):
        return float("nan")
    dominated = [
        any(_dominates(source_point, target_point) for source_point in source)
        for target_point in target
    ]
    return float(np.mean(dominated))


def _combined_front_contributions(tpe_values, random_values):
    """Count unique combined-front points by originating search strategy."""
    provenance = {}
    for strategy, values in (("tpe", tpe_values), ("random", random_values)):
        for point in values:
            key = tuple(np.round(point, decimals=12))
            provenance.setdefault(key, set()).add(strategy)

    unique_points = np.asarray(list(provenance), dtype=float)
    if not len(unique_points):
        return {"tpe": 0, "random": 0, "shared": 0, "total": 0}
    front_indices = _nondominated_indices(unique_points)
    counts = {"tpe": 0, "random": 0, "shared": 0}
    for index in front_indices:
        origins = provenance[tuple(unique_points[index])]
        if len(origins) > 1:
            counts["shared"] += 1
        else:
            counts[next(iter(origins))] += 1
    counts["total"] = len(front_indices)
    return counts


def _pareto_quality_rows(studies):
    rows = []
    for scenario in SCENARIOS:
        metric_spec = _metric_spec(scenario)
        metric_names = [name for name, _direction in metric_spec]
        directions = [direction for _name, direction in metric_spec]
        bounds = _normalization_bounds(studies, scenario, metric_names)
        if bounds is None:
            continue
        for budget in BUDGETS:
            tpe_study = studies.get(("tpe", scenario.name, budget))
            random_study = studies.get(("random", scenario.name, budget))
            if (
                tpe_study is None
                or random_study is None
                or tpe_study.get("status") != "complete"
                or random_study.get("status") != "complete"
            ):
                continue
            tpe_raw = _finite_values(tpe_study["pareto"], metric_names)
            random_raw = _finite_values(random_study["pareto"], metric_names)
            if not len(tpe_raw) or not len(random_raw):
                continue
            tpe = _normalize_as_benefit(tpe_raw, directions, bounds)
            random = _normalize_as_benefit(random_raw, directions, bounds)
            contributions = _combined_front_contributions(tpe, random)
            denominator = contributions["total"] or 1
            tpe_hypervolume = _hypervolume(tpe)
            random_hypervolume = _hypervolume(random)
            rows.append(
                {
                    "scenario": scenario.name,
                    "budget": budget,
                    "tpe_hypervolume": tpe_hypervolume,
                    "random_hypervolume": random_hypervolume,
                    "hypervolume_difference": tpe_hypervolume - random_hypervolume,
                    "tpe_contribution": contributions["tpe"] / denominator,
                    "random_contribution": contributions["random"] / denominator,
                    "shared_contribution": contributions["shared"] / denominator,
                    "coverage_tpe_random": _set_coverage(tpe, random),
                    "coverage_random_tpe": _set_coverage(random, tpe),
                }
            )
    return rows


def _save_hypervolume_plot(quality_rows, output_path):
    frame = pd.DataFrame(quality_rows)
    figure, axes = plt.subplots(4, 2, figsize=(15, 18), constrained_layout=True)
    axes = axes.ravel()
    for axis, scenario in zip(axes, SCENARIOS):
        scenario_frame = (
            frame.loc[frame["scenario"] == scenario.name]
            if not frame.empty
            else pd.DataFrame()
        )
        for strategy in STRATEGIES:
            column = f"{strategy}_hypervolume"
            if scenario_frame.empty:
                continue
            ordered = scenario_frame.sort_values("budget")
            axis.plot(
                ordered["budget"],
                ordered[column],
                color=SEARCH_COLORS[strategy],
                linestyle=SEARCH_LINESTYLES[strategy],
                marker="o",
                label=STRATEGY_LABELS[strategy],
            )
        axis.set_title(SCENARIO_LABELS[scenario.name])
        axis.set_xlabel("Trial budget")
        axis.set_ylabel("Normalized hypervolume")
        axis.set_xticks(BUDGETS)
        axis.set_ylim(-0.02, 1.02)
        axis.grid(alpha=0.25)
        if scenario_frame.empty:
            axis.text(
                0.5,
                0.5,
                "No matched completed studies",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
    axes[-1].axis("off")
    handles = [
        Line2D(
            [0],
            [0],
            color=SEARCH_COLORS[strategy],
            linestyle=SEARCH_LINESTYLES[strategy],
            marker="o",
            label=STRATEGY_LABELS[strategy],
        )
        for strategy in STRATEGIES
    ]
    figure.legend(handles=handles, loc="lower center", ncol=2)
    figure.suptitle("Pareto hypervolume by scenario and trial budget", fontsize=16)
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _front_trial_numbers(study):
    frame = study["pareto"]
    if "trial_number" not in frame:
        return np.full(len(frame), -1, dtype=int)
    return pd.to_numeric(frame["trial_number"], errors="coerce").fillna(-1).astype(int)


def _plot_2d_front(axis, study, metric_names, strategy):
    values = _finite_values(study["pareto"], metric_names)
    if not len(values):
        return False
    color = SEARCH_COLORS[strategy]
    marker = "o" if strategy == "tpe" else "^"
    axis.scatter(values[:, 0], values[:, 1], color=color, marker=marker, s=30)
    order = np.argsort(values[:, 0])
    axis.step(
        values[order, 0],
        values[order, 1],
        where="post",
        color=color,
        linestyle=SEARCH_LINESTYLES[strategy],
        alpha=0.8,
    )
    return True


def _plot_parallel_front(
    axis,
    study,
    metric_names,
    directions,
    bounds,
    strategy,
):
    values = _finite_values(study["pareto"], metric_names)
    if not len(values):
        return False
    normalized = _normalize_as_benefit(values, directions, bounds, drop_constant=False)
    trial_numbers = _front_trial_numbers(study)
    finite_mask = np.isfinite(
        study["pareto"]
        .loc[:, metric_names]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=float)
    ).all(axis=1)
    trial_numbers = np.asarray(trial_numbers)[finite_mask]
    selected_number = (study.get("selected_solution") or {}).get("trial_number")
    x_values = np.arange(normalized.shape[1])
    for trial_number, row in zip(trial_numbers, normalized):
        selected = trial_number == selected_number
        axis.plot(
            x_values,
            row,
            color=SEARCH_COLORS[strategy],
            linestyle=SEARCH_LINESTYLES[strategy],
            alpha=0.95 if selected else 0.18,
            linewidth=2.8 if selected else 0.8,
        )
    return True


def _safe_filename(value):
    return re.sub(r"[^a-z0-9_-]+", "_", value.lower()).strip("_")


def _save_raw_pareto_plots(studies, plot_directory):
    outputs = []
    for scenario in SCENARIOS:
        metric_spec = _metric_spec(scenario)
        metric_names = [name for name, _direction in metric_spec]
        directions = [direction for _name, direction in metric_spec]
        available = any(
            studies.get((strategy, scenario.name, budget), {}).get("status")
            == "complete"
            for strategy in STRATEGIES
            for budget in BUDGETS
        )
        if not available:
            continue
        bounds = _normalization_bounds(studies, scenario, metric_names)
        if bounds is None:
            continue
        figure, axes = plt.subplots(
            1,
            len(BUDGETS),
            figsize=(20, 5),
            constrained_layout=True,
        )
        for axis, budget in zip(axes, BUDGETS):
            plotted = False
            for strategy in STRATEGIES:
                study = studies.get((strategy, scenario.name, budget))
                if study is None or study.get("status") != "complete":
                    continue
                if len(metric_names) == 2:
                    plotted |= _plot_2d_front(axis, study, metric_names, strategy)
                else:
                    plotted |= _plot_parallel_front(
                        axis,
                        study,
                        metric_names,
                        directions,
                        bounds,
                        strategy,
                    )
            axis.set_title(f"Budget {budget}")
            axis.grid(alpha=0.25)
            if len(metric_names) == 2:
                axis.set_xlabel(f"{METRIC_LABELS[metric_names[0]]} ({directions[0]})")
                axis.set_ylabel(f"{METRIC_LABELS[metric_names[1]]} ({directions[1]})")
                minima, maxima = bounds
                x_padding = (
                    0.05 * (maxima[0] - minima[0]) if maxima[0] > minima[0] else 0.05
                )
                y_padding = (
                    0.05 * (maxima[1] - minima[1]) if maxima[1] > minima[1] else 0.05
                )
                axis.set_xlim(minima[0] - x_padding, maxima[0] + x_padding)
                axis.set_ylim(minima[1] - y_padding, maxima[1] + y_padding)
            else:
                axis.set_xticks(
                    np.arange(len(metric_names)),
                    [METRIC_LABELS[name] for name in metric_names],
                    rotation=20,
                    ha="right",
                )
                axis.set_ylim(-0.02, 1.02)
                axis.set_ylabel("Normalized benefit (higher is better)")
            if not plotted:
                axis.text(
                    0.5,
                    0.5,
                    "No completed study",
                    ha="center",
                    va="center",
                    transform=axis.transAxes,
                )
        plot_kind = (
            "2D scatter and Pareto staircase"
            if len(metric_names) == 2
            else "parallel coordinates; thick lines are selected solutions"
        )
        figure.suptitle(f"{SCENARIO_LABELS[scenario.name]} — {plot_kind}", fontsize=15)
        handles = [
            Line2D(
                [0],
                [0],
                color=SEARCH_COLORS[strategy],
                linestyle=SEARCH_LINESTYLES[strategy],
                marker="o" if strategy == "tpe" else "^",
                label=STRATEGY_LABELS[strategy],
            )
            for strategy in STRATEGIES
        ]
        figure.legend(handles=handles, loc="lower center", ncol=2)
        filename = f"pareto_front__{_safe_filename(scenario.name)}.png"
        figure.savefig(plot_directory / filename, dpi=160, bbox_inches="tight")
        plt.close(figure)
        outputs.append((scenario.name, filename, len(metric_names)))
    return outputs


def _runtime_points(studies, strategy, scenario_name):
    points = []
    for budget in BUDGETS:
        study = studies.get((strategy, scenario_name, budget))
        if study is None or study.get("status") != "complete":
            continue
        seconds = study.get("active_execution_seconds")
        if seconds is not None and np.isfinite(float(seconds)) and float(seconds) > 0:
            points.append((budget, float(seconds) / 3600))
    return points


def _set_runtime_scale(axis, values):
    positive = [value for value in values if value > 0]
    if positive and max(positive) / min(positive) >= 10:
        axis.set_yscale("log")


def _save_runtime_plot(studies, output_path):
    figure, (phetda_axis, baseline_axis) = plt.subplots(
        1, 2, figsize=(17, 7), constrained_layout=True
    )
    all_phetda_values = []
    for strategy in STRATEGIES:
        values_by_budget = {budget: [] for budget in BUDGETS}
        for scenario in SCENARIOS:
            points = _runtime_points(studies, strategy, scenario.name)
            if not points:
                continue
            x_values, y_values = zip(*points)
            all_phetda_values.extend(y_values)
            for budget, value in points:
                values_by_budget[budget].append(value)
            phetda_axis.plot(
                x_values,
                y_values,
                color=SEARCH_COLORS[strategy],
                linestyle=SEARCH_LINESTYLES[strategy],
                linewidth=0.8,
                marker="o",
                markersize=3,
                alpha=0.2,
            )

        summary_budgets = []
        medians = []
        lower_quartiles = []
        upper_quartiles = []
        for budget in BUDGETS:
            values = values_by_budget[budget]
            if not values:
                continue
            summary_budgets.append(budget)
            medians.append(float(np.median(values)))
            lower_quartiles.append(float(np.quantile(values, 0.25)))
            upper_quartiles.append(float(np.quantile(values, 0.75)))
        if summary_budgets:
            phetda_axis.plot(
                summary_budgets,
                medians,
                color=SEARCH_COLORS[strategy],
                linestyle=SEARCH_LINESTYLES[strategy],
                linewidth=3,
                marker="o",
                label=f"{STRATEGY_LABELS[strategy]} median",
            )
            phetda_axis.fill_between(
                summary_budgets,
                lower_quartiles,
                upper_quartiles,
                color=SEARCH_COLORS[strategy],
                alpha=0.15,
            )

    phetda_axis.set_title("pheTDA: thin lines are scenarios; bands are IQRs")
    phetda_axis.set_xlabel("Trial budget")
    phetda_axis.set_ylabel("Active runtime (hours)")
    phetda_axis.set_xticks(BUDGETS)
    phetda_axis.grid(alpha=0.25)
    _set_runtime_scale(phetda_axis, all_phetda_values)
    if phetda_axis.lines:
        phetda_axis.legend()

    all_baseline_values = []
    baseline_mapping = {
        "tpe": TPE_BASELINE_STRATEGY,
        "random": RANDOM_BASELINE_STRATEGY,
    }
    for search_name, strategy in baseline_mapping.items():
        points = _runtime_points(studies, strategy, UNSUPERVISED_BASELINE_NAME)
        if not points:
            continue
        x_values, y_values = zip(*points)
        all_baseline_values.extend(y_values)
        baseline_axis.plot(
            x_values,
            y_values,
            color=SEARCH_COLORS[search_name],
            linestyle=SEARCH_LINESTYLES[search_name],
            linewidth=2.5,
            marker="^",
            label=STRATEGY_LABELS[strategy],
        )
    baseline_axis.set_title("Unsupervised reduction + clustering")
    baseline_axis.set_xlabel("Trial budget")
    baseline_axis.set_ylabel("Active runtime (hours)")
    baseline_axis.set_xticks(BUDGETS)
    baseline_axis.grid(alpha=0.25)
    _set_runtime_scale(baseline_axis, all_baseline_values)
    if baseline_axis.lines:
        baseline_axis.legend()

    figure.suptitle("Runtime versus trial budget", fontsize=16)
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _paired_runtime_rows(studies):
    configurations = [(scenario.name, "tpe", "random") for scenario in SCENARIOS]
    configurations.append(
        (
            UNSUPERVISED_BASELINE_NAME,
            TPE_BASELINE_STRATEGY,
            RANDOM_BASELINE_STRATEGY,
        )
    )
    rows = []
    for scenario_name, tpe_strategy, random_strategy in configurations:
        for budget in BUDGETS:
            tpe = studies.get((tpe_strategy, scenario_name, budget))
            random = studies.get((random_strategy, scenario_name, budget))
            if (
                tpe is None
                or random is None
                or tpe.get("status") != "complete"
                or random.get("status") != "complete"
            ):
                continue
            tpe_seconds = tpe.get("active_execution_seconds")
            random_seconds = random.get("active_execution_seconds")
            if tpe_seconds is None or random_seconds is None:
                continue
            tpe_hours = float(tpe_seconds) / 3600
            random_hours = float(random_seconds) / 3600
            rows.append(
                {
                    "scenario": scenario_name,
                    "budget": budget,
                    "tpe_hours": tpe_hours,
                    "random_hours": random_hours,
                    "runtime_ratio": (
                        tpe_hours / random_hours if random_hours > 0 else float("nan")
                    ),
                }
            )
    return rows


def _selected_value(study, metric_name):
    if study is None or study.get("status") != "complete":
        return None
    selected = study.get("selected_solution") or {}
    value = (selected.get("values") or {}).get(metric_name)
    if value is None or not np.isfinite(float(value)):
        return None
    return float(value)


def _objective_rows(metric_name):
    rows = []
    for scenario in SCENARIOS:
        direction_by_metric = dict(_metric_spec(scenario))
        if metric_name in direction_by_metric:
            rows.append((scenario.name, direction_by_metric[metric_name]))
    if metric_name == "stratification_silhouette":
        rows.append((UNSUPERVISED_BASELINE_NAME, "maximize"))
    return rows


def _objective_advantage_matrix(studies, metric_name):
    row_specification = _objective_rows(metric_name)
    matrix = np.full((len(row_specification), len(BUDGETS)), np.nan)
    for row, (scenario_name, direction) in enumerate(row_specification):
        baseline = scenario_name == UNSUPERVISED_BASELINE_NAME
        tpe_strategy = TPE_BASELINE_STRATEGY if baseline else "tpe"
        random_strategy = RANDOM_BASELINE_STRATEGY if baseline else "random"
        for column, budget in enumerate(BUDGETS):
            tpe_value = _selected_value(
                studies.get((tpe_strategy, scenario_name, budget)), metric_name
            )
            random_value = _selected_value(
                studies.get((random_strategy, scenario_name, budget)), metric_name
            )
            if tpe_value is None or random_value is None:
                continue
            matrix[row, column] = (
                tpe_value - random_value
                if direction == "maximize"
                else random_value - tpe_value
            )
    return row_specification, matrix


def _save_selected_advantage_heatmaps(studies, output_path):
    figure, axes = plt.subplots(1, 3, figsize=(22, 10), constrained_layout=True)
    for axis, metric_name in zip(axes, SELECTED_METRICS):
        row_specification, matrix = _objective_advantage_matrix(studies, metric_name)
        finite = np.abs(matrix[np.isfinite(matrix)])
        limit = float(finite.max()) if finite.size else 1.0
        if limit <= DOMINANCE_TOLERANCE:
            limit = 1.0
        image = axis.imshow(
            np.ma.masked_invalid(matrix),
            cmap="RdBu",
            vmin=-limit,
            vmax=limit,
            aspect="auto",
        )
        axis.set_title(METRIC_LABELS[metric_name])
        axis.set_xticks(np.arange(len(BUDGETS)), BUDGETS)
        axis.set_xlabel("Trial budget")
        axis.set_yticks(
            np.arange(len(row_specification)),
            [SCENARIO_LABELS[name] for name, _direction in row_specification],
        )
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                value = matrix[row, column]
                label = "—" if not np.isfinite(value) else f"{value:.3f}"
                text_color = (
                    "white"
                    if np.isfinite(value) and abs(value) > 0.55 * limit
                    else "black"
                )
                axis.text(
                    column,
                    row,
                    label,
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=8,
                )
        figure.colorbar(
            image,
            ax=axis,
            orientation="horizontal",
            shrink=0.75,
            pad=0.08,
            label="TPE advantage (+ favors TPE)",
        )
    figure.suptitle("Selected-solution objective comparison: TPE versus random")
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _selected_objective_rows(studies):
    rows = []
    ordered_configurations = [
        (scenario.name, strategy) for scenario in SCENARIOS for strategy in STRATEGIES
    ]
    ordered_configurations.extend(
        (UNSUPERVISED_BASELINE_NAME, strategy) for strategy in BASELINE_STRATEGIES
    )
    for scenario_name, strategy in ordered_configurations:
        for budget in BUDGETS:
            study = studies.get((strategy, scenario_name, budget))
            if study is None or study.get("status") != "complete":
                continue
            rows.append(
                {
                    "scenario": scenario_name,
                    "strategy": strategy,
                    "budget": budget,
                    **{
                        metric_name: _selected_value(study, metric_name)
                        for metric_name in SELECTED_METRICS
                    },
                }
            )
    return rows


def _coverage_table(studies):
    lines = [
        "| Configuration | TPE budgets | Random budgets |",
        "|---|---:|---:|",
    ]
    configurations = [(scenario.name, "tpe", "random") for scenario in SCENARIOS] + [
        (
            UNSUPERVISED_BASELINE_NAME,
            TPE_BASELINE_STRATEGY,
            RANDOM_BASELINE_STRATEGY,
        )
    ]
    for scenario_name, tpe_strategy, random_strategy in configurations:
        cells = []
        for strategy in (tpe_strategy, random_strategy):
            complete = [
                str(budget)
                for budget in BUDGETS
                if studies.get((strategy, scenario_name, budget), {}).get("status")
                == "complete"
            ]
            cells.append(", ".join(complete) if complete else "—")
        lines.append(f"| {SCENARIO_LABELS[scenario_name]} | {cells[0]} | {cells[1]} |")
    return lines


def _quality_table(rows):
    lines = [
        "| Configuration | Budget | TPE HV | Random HV | Δ HV | TPE-only contribution | Random-only contribution | Shared | C(TPE, Random) | C(Random, TPE) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {scenario} | {budget} | {tpe_hv:.4f} | {random_hv:.4f} | "
            "{delta:.4f} | {tpe:.1%} | {random:.1%} | {shared:.1%} | "
            "{coverage_tpe:.3f} | {coverage_random:.3f} |".format(
                scenario=SCENARIO_LABELS[row["scenario"]],
                budget=row["budget"],
                tpe_hv=row["tpe_hypervolume"],
                random_hv=row["random_hypervolume"],
                delta=row["hypervolume_difference"],
                tpe=row["tpe_contribution"],
                random=row["random_contribution"],
                shared=row["shared_contribution"],
                coverage_tpe=row["coverage_tpe_random"],
                coverage_random=row["coverage_random_tpe"],
            )
        )
    if len(lines) == 2:
        lines.append(
            "| No matched completed studies | — | — | — | — | — | — | — | — | — |"
        )
    return lines


def _runtime_table(rows):
    lines = [
        "| Configuration | Budget | TPE runtime (h) | Random runtime (h) | TPE/random ratio |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {SCENARIO_LABELS[row['scenario']]} | {row['budget']} | "
            f"{row['tpe_hours']:.4f} | {row['random_hours']:.4f} | "
            f"{row['runtime_ratio']:.3f} |"
        )
    if len(lines) == 2:
        lines.append("| No matched completed studies | — | — | — | — |")
    return lines


def _selected_table(rows):
    lines = [
        "| Configuration | Strategy | Budget | Silhouette | Initial entropy | Final entropy |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {SCENARIO_LABELS[row['scenario']]} | "
            f"{STRATEGY_LABELS[row['strategy']]} | {row['budget']} | "
            f"{_format_value(row['stratification_silhouette'])} | "
            f"{_format_value(row['initial_graph_entropy'])} | "
            f"{_format_value(row['final_graph_entropy'])} |"
        )
    if len(lines) == 2:
        lines.append("| No completed studies | — | — | — | — | — |")
    return lines


def _comparison_counts(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return {
        "total": len(values),
        "positive": int(np.sum(values > DOMINANCE_TOLERANCE)),
        "negative": int(np.sum(values < -DOMINANCE_TOLERANCE)),
        "ties": int(np.sum(np.abs(values) <= DOMINANCE_TOLERANCE)),
    }


def _pareto_summary(quality_rows):
    if not quality_rows:
        return [
            "**Descriptive conclusion.** No matched completed TPE/random studies are "
            "available, so no Pareto-quality comparison can yet be made."
        ]

    differences = [row["hypervolume_difference"] for row in quality_rows]
    counts = _comparison_counts(differences)
    largest_tpe = max(quality_rows, key=lambda row: row["hypervolume_difference"])
    largest_random = min(quality_rows, key=lambda row: row["hypervolume_difference"])
    coverage_advantages = [
        row["coverage_tpe_random"] - row["coverage_random_tpe"] for row in quality_rows
    ]
    coverage_counts = _comparison_counts(coverage_advantages)
    contribution_counts = _comparison_counts(
        [row["tpe_contribution"] - row["random_contribution"] for row in quality_rows]
    )
    return [
        "**Descriptive conclusion.** Across "
        f"{counts['total']} matched scenario/budget comparisons, TPE has higher "
        f"hypervolume in **{counts['positive']}**, random search in "
        f"**{counts['negative']}**, with **{counts['ties']}** ties. The median "
        f"TPE-minus-random hypervolume difference is "
        f"**{np.median(differences):.4f}**.",
        "The largest observed advantage for TPE is "
        f"**{largest_tpe['hypervolume_difference']:.4f}** for "
        f"**{SCENARIO_LABELS[largest_tpe['scenario']]}**, budget "
        f"**{largest_tpe['budget']}**; the most negative difference is "
        f"**{largest_random['hypervolume_difference']:.4f}** for "
        f"**{SCENARIO_LABELS[largest_random['scenario']]}**, budget "
        f"**{largest_random['budget']}**.",
        "TPE coverage exceeds reverse random-search coverage in "
        f"**{coverage_counts['positive']} of {coverage_counts['total']}** matched "
        "comparisons. TPE contributes a larger fraction of the pooled front in "
        f"**{contribution_counts['positive']}**, random search in "
        f"**{contribution_counts['negative']}**, with "
        f"**{contribution_counts['ties']}** equal contributions. These counts "
        "describe the observed runs and are not a statistical test of optimizer "
        "superiority.",
    ]


def _scenario_pareto_summary(quality_rows, scenario_name):
    rows = [row for row in quality_rows if row["scenario"] == scenario_name]
    if not rows:
        return (
            "No matched completed TPE/random budgets are available for a numerical "
            "comparison of this front."
        )
    counts = _comparison_counts([row["hypervolume_difference"] for row in rows])
    return (
        f"Across {counts['total']} matched budgets, TPE has higher hypervolume in "
        f"{counts['positive']}, random search in {counts['negative']}, and they tie "
        f"in {counts['ties']}. Use the plotted front to see whether this difference "
        "comes from uniformly better solutions or from different trade-off regions."
    )


def _runtime_summary(runtime_rows):
    valid_rows = [row for row in runtime_rows if np.isfinite(row["runtime_ratio"])]
    if not valid_rows:
        return [
            "**Descriptive conclusion.** No matched TPE/random runtime pairs are "
            "available."
        ]
    ratios = [row["runtime_ratio"] for row in valid_rows]
    counts = _comparison_counts([1.0 - ratio for ratio in ratios])
    phetda_ratios = [
        row["runtime_ratio"]
        for row in valid_rows
        if row["scenario"] != UNSUPERVISED_BASELINE_NAME
    ]
    baseline_ratios = [
        row["runtime_ratio"]
        for row in valid_rows
        if row["scenario"] == UNSUPERVISED_BASELINE_NAME
    ]
    details = []
    if phetda_ratios:
        details.append(f"pheTDA median ratio **{np.median(phetda_ratios):.3f}**")
    if baseline_ratios:
        details.append(
            f"unsupervised-baseline median ratio **{np.median(baseline_ratios):.3f}**"
        )
    detail_text = "; ".join(details)
    return [
        "**Descriptive conclusion.** TPE is faster in "
        f"**{counts['positive']} of {counts['total']}** matched comparisons, random "
        f"search is faster in **{counts['negative']}**, and **{counts['ties']}** are "
        "equal within numerical tolerance. The overall median TPE/random runtime "
        f"ratio is **{np.median(ratios):.3f}**"
        + (f" ({detail_text})." if detail_text else "."),
        "Runtime differences reflect which hyperparameters each search sampled as "
        "well as sampler overhead; they should be interpreted together with Pareto "
        "quality rather than as a stand-alone measure of performance.",
    ]


def _selected_objective_summary(studies):
    lines = ["**Descriptive conclusion.**"]
    any_comparison = False
    for metric_name in SELECTED_METRICS:
        row_specification, matrix = _objective_advantage_matrix(studies, metric_name)
        counts = _comparison_counts(matrix.ravel())
        if counts["total"] == 0:
            lines.append(
                f"- **{METRIC_LABELS[metric_name]}:** no matched selected solutions."
            )
            continue
        any_comparison = True
        finite_locations = np.argwhere(np.isfinite(matrix))
        best_location = max(
            finite_locations,
            key=lambda location: matrix[tuple(location)],
        )
        worst_location = min(
            finite_locations,
            key=lambda location: matrix[tuple(location)],
        )
        best_scenario = row_specification[int(best_location[0])][0]
        worst_scenario = row_specification[int(worst_location[0])][0]
        best_budget = BUDGETS[int(best_location[1])]
        worst_budget = BUDGETS[int(worst_location[1])]
        lines.append(
            f"- **{METRIC_LABELS[metric_name]}:** TPE is better in "
            f"**{counts['positive']} of {counts['total']}** matched cells, random "
            f"search in **{counts['negative']}**, with **{counts['ties']}** ties. "
            f"The largest TPE advantage is **{matrix[tuple(best_location)]:.4f}** "
            f"for {SCENARIO_LABELS[best_scenario]} at budget {best_budget}; the most "
            f"negative value is **{matrix[tuple(worst_location)]:.4f}** for "
            f"{SCENARIO_LABELS[worst_scenario]} at budget {worst_budget}."
        )
    if any_comparison:
        lines.append(
            "A win on one selected objective can be accompanied by a loss on another "
            "because the pheTDA solution is selected as a multi-objective compromise."
        )
    return lines


def generate_report(results_directory):
    """Generate the redesigned report and plots from existing study outputs."""
    results_directory = Path(results_directory)
    results_directory.mkdir(parents=True, exist_ok=True)
    plot_directory = results_directory / "analysis_plots_new"
    plot_directory.mkdir(parents=True, exist_ok=True)
    studies = _load_studies(results_directory)

    quality_rows = _pareto_quality_rows(studies)
    runtime_rows = _paired_runtime_rows(studies)
    selected_rows = _selected_objective_rows(studies)

    _save_hypervolume_plot(quality_rows, plot_directory / "pareto_hypervolume.png")
    raw_pareto_outputs = _save_raw_pareto_plots(studies, plot_directory)
    _save_runtime_plot(studies, plot_directory / "runtime_by_trials.png")
    _save_selected_advantage_heatmaps(
        studies, plot_directory / "selected_objective_advantage.png"
    )

    complete_count = sum(
        study.get("status") == "complete" for study in studies.values()
    )
    expected_count = (
        len(SCENARIOS) * len(STRATEGIES) + len(BASELINE_STRATEGIES)
    ) * len(BUDGETS)
    lines = [
        "# Synthea COVID benchmark analysis — redesigned",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "",
        "## Result coverage",
        "",
        f"The results directory contains **{complete_count} of {expected_count}** "
        "planned completed studies.",
        "",
        *_coverage_table(studies),
        "",
        "## Pareto-front comparison",
        "",
        "### Hypervolume versus trial budget",
        "",
        "**What the plot shows.** Each panel corresponds to one phenotype scenario. "
        "The x-axis is the number of trials and the y-axis is the hypervolume of the "
        "valid Pareto front found by that search. Solid blue lines represent TPE and "
        "dashed orange lines represent random search. At the same scenario and budget, "
        "the higher curve represents a front that is closer to the ideal objective "
        "values and/or spans a broader range of useful trade-offs.",
        "",
        "**How it is computed.** Objective values are first converted to normalized "
        "benefits, so larger is always better. For observed lower and upper bounds "
        "$L_j$ and $U_j$ of objective $j$:",
        "",
        "$$",
        r"z_j(x)=\begin{cases}\dfrac{x_j-L_j}{U_j-L_j}, & j\text{ is maximized},\\[6pt]\dfrac{U_j-x_j}{U_j-L_j}, & j\text{ is minimized}.\end{cases}",
        "$$",
        "",
        "The same bounds are used for both strategies and every available budget "
        "within a scenario. Constant objectives are removed because they cannot "
        r"distinguish solutions. With reference point $\mathbf{0}$, hypervolume is the "
        "area (two objectives) or volume (three objectives) of the union of boxes "
        "dominated by the normalized Pareto points:",
        "",
        "$$",
        r"HV(F)=\lambda_d\!\left(\bigcup_{x\in F}\prod_{j=1}^{d}[0,z_j(x)]\right),",
        r"\qquad \Delta HV=HV(F_{TPE})-HV(F_{Random}).",
        "$$",
        "",
        r"Thus, higher hypervolume and positive $\Delta HV$ favor TPE. Hypervolume is "
        "comparable across strategies and budgets within a scenario, but it is not an "
        "absolute clinical quantity and should not be compared directly between "
        "different phenotype scenarios.",
        "",
        "![Pareto hypervolume](analysis_plots_new/pareto_hypervolume.png)",
        "",
        *_pareto_summary(quality_rows),
        "",
        "### Combined-front contribution and set coverage",
        "",
        "**What the table shows.** The hypervolume columns summarize the overall "
        "quality of each front. Contribution percentages show which search produced "
        "the unique solutions that remain after the two fronts are pooled. A solution "
        "found by both strategies, within a numerical tolerance, is classified as "
        "shared. The two coverage columns measure directional domination between the "
        "fronts.",
        "",
        "**How it is computed.** Let $F_T$ and $F_R$ be the TPE and random Pareto "
        r"fronts and $ND(\cdot)$ return the non-dominated subset. The combined front is:",
        "",
        "$$",
        r"F_C=ND(F_T\cup F_R),\qquad p_T=\frac{|F_C^{TPE\ only}|}{|F_C|},\quad p_R=\frac{|F_C^{Random\ only}|}{|F_C|}.",
        "$$",
        "",
        "After orienting all objectives as benefits, $a$ dominates $b$ when it is no "
        "worse in every objective and strictly better in at least one. Set coverage is:",
        "",
        "$$",
        r"C(T,R)=\frac{|\{r\in F_R:\exists t\in F_T,\ t\succ r\}|}{|F_R|}.",
        "$$",
        "",
        "$C(T,R)=1$ means every random-front point is dominated by at least one TPE "
        "point; zero means none is. $C(T,R)$ and $C(R,T)$ are calculated separately "
        "and are not complements.",
        "",
        *_quality_table(quality_rows),
        "",
        "### Raw Pareto-front views",
        "",
        "**What the plots show.** In two-objective scenarios, every dot is an actual "
        "valid Pareto-optimal Optuna trial and the step line traces the discrete front. "
        "In three-objective scenarios, each polyline is one Pareto solution crossing "
        "the initial-entropy, final-entropy, and silhouette axes. All three axes show "
        "the normalized benefit $z_j(x)$, so moving upward is always better. The "
        "thicker polyline is the selected single solution.",
        "",
        "**How they are computed.** The two-dimensional staircase is obtained by "
        "sorting the observed Pareto points by the x objective; its horizontal and "
        "vertical segments are only visual guides and do not represent unobserved "
        "trials. Parallel-coordinate positions use the same direction-aware "
        "normalization equation as hypervolume. Dominance is defined by:",
        "",
        "$$",
        r"a\succ b\iff [z_j(a)\ge z_j(b)\ \forall j]\ \land\ [z_k(a)>z_k(b)\ \text{for at least one }k].",
        "$$",
        "",
    ]
    if raw_pareto_outputs:
        for scenario_name, filename, objective_count in raw_pareto_outputs:
            kind = (
                "2D scatter/staircase"
                if objective_count == 2
                else "parallel coordinates"
            )
            lines.extend(
                [
                    f"#### {SCENARIO_LABELS[scenario_name]}",
                    "",
                    f"{kind}.",
                    "",
                    f"![{SCENARIO_LABELS[scenario_name]} Pareto front]"
                    f"(analysis_plots_new/{filename})",
                    "",
                    "**Descriptive conclusion.** "
                    + _scenario_pareto_summary(quality_rows, scenario_name),
                    "",
                ]
            )
    else:
        lines.extend(["No completed Pareto studies are currently available.", ""])

    lines.extend(
        [
            "## Runtime versus trial budget",
            "",
            "**What the plot shows.** The left panel contains pheTDA studies. Each "
            "thin line is one phenotype scenario; the thick lines summarize the "
            "scenario-specific runtimes at every budget, and their shaded regions show "
            "the interquartile range across scenarios. The right panel shows the "
            "phenotype-independent dimensionality-reduction-plus-clustering baseline "
            "on its own scale. Solid blue lines are TPE and dashed orange lines are "
            "random search. A logarithmic y-axis is used automatically when the "
            "positive runtimes span at least a factor of ten.",
            "",
            "**How it is computed.** Active runtime excludes interruption downtime and "
            "shared preprocessing, but includes optimization and selected-solution "
            "postprocessing:",
            "",
            "$$",
            r"R_{s,b,m}=\frac{\text{active execution seconds}_{s,b,m}}{3600}.",
            "$$",
            "",
            "For search method $m$ and budget $b$, the thick pheTDA line is the median "
            "of $R_{s,b,m}$ across the available phenotype scenarios $s$. The band is "
            "$[Q_{0.25},Q_{0.75}]$; it describes variation between scenarios, not a "
            "confidence interval. The paired runtime ratio is:",
            "",
            "$$",
            r"\rho_{s,b}=\frac{R_{s,b,TPE}}{R_{s,b,Random}}.",
            "$$",
            "",
            r"$\rho<1$ means TPE was faster, $\rho>1$ means random search was faster, "
            r"and $\rho=1$ means equal recorded runtime.",
            "",
            "![Runtime versus trial budget](analysis_plots_new/runtime_by_trials.png)",
            "",
            *_runtime_summary(runtime_rows),
            "",
            *_runtime_table(runtime_rows),
            "",
            "## Objectives of the selected solution",
            "",
            "**What the heatmaps show.** The panels are silhouette, initial graph "
            "entropy, and final graph entropy. Rows are applicable configurations and "
            "columns are trial budgets. Each cell compares the raw objective of the "
            "single solution selected from the TPE front with the corresponding "
            "random-search solution. Blue cells favor TPE, red cells favor random "
            "search, values near zero are similar, and an em dash means that a matched "
            "comparison is unavailable. Because the unsupervised baseline optimizes "
            "only silhouette, it appears only in the first panel.",
            "",
            "**How it is computed.** The displayed difference is oriented so that a "
            "positive value always favors TPE:",
            "",
            "$$",
            r"A_{s,b,j}=\begin{cases}x^{TPE}_{s,b,j}-x^{Random}_{s,b,j}, & j\text{ is maximized},\\[6pt]x^{Random}_{s,b,j}-x^{TPE}_{s,b,j}, & j\text{ is minimized}.\end{cases}",
            "$$",
            "",
            "The color scale is centered at zero and is calculated independently for "
            "each objective because entropy and silhouette have different numerical "
            "scales. These are raw objective differences, not standardized effect "
            "sizes. The selected pheTDA solution minimizes its normalized distance to "
            "the study-specific ideal point, so it is a compromise rather than the "
            "independently best trial for every objective.",
            "",
            "![Direction-adjusted selected objectives]"
            "(analysis_plots_new/selected_objective_advantage.png)",
            "",
            *_selected_objective_summary(studies),
            "",
            "### Raw selected-objective values",
            "",
            *_selected_table(selected_rows),
            "",
            "## Interpretation cautions",
            "",
            "- Hypervolume normalization bounds are calculated from the available "
            "Pareto results and are fixed across strategies and budgets within each "
            "scenario when this report is generated. Adding studies can change those "
            "bounds, so regenerate the complete report before the final comparison.",
            "- A large combined-front contribution can be influenced by many nearby "
            "solutions. Interpret it together with hypervolume and set coverage.",
            "- Separate budget studies are not prefixes of one common optimization run.",
            "- One seed per condition supports a descriptive comparison only. Repeated "
            "seeds are needed for uncertainty intervals or formal claims.",
            "- Runtime comparisons require the same machine and similar system load.",
            "",
            "## Regeneration",
            "",
            "Run from the repository root:",
            "",
            "```text",
            "python -m synthea_covid_experiment.analyze_results_new "
            "--results-directory synthea_covid_experiment/results",
            "```",
            "",
        ]
    )
    report_path = results_directory / "RESULTS_ANALYSIS_new.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-directory",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    report_path = generate_report(args.results_directory)
    print(f"Analysis report: {report_path}")


if __name__ == "__main__":
    main()
