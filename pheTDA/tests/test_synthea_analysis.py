import numpy as np

from synthea_covid_experiment.analyze_results_new import (
    _combined_front_contributions,
    _hypervolume,
    _normalize_as_benefit,
    _pareto_summary,
    _runtime_summary,
    _set_coverage,
)


def test_direction_aware_normalization_makes_larger_values_better():
    values = np.asarray([[2.0, 0.8], [4.0, 0.4]])
    bounds = (np.asarray([2.0, 0.4]), np.asarray([4.0, 0.8]))

    normalized = _normalize_as_benefit(
        values,
        directions=["minimize", "maximize"],
        bounds=bounds,
    )

    np.testing.assert_allclose(normalized, [[1.0, 1.0], [0.0, 0.0]])


def test_exact_two_and_three_dimensional_hypervolume():
    two_dimensional = np.asarray([[0.8, 0.4], [0.5, 0.7]])
    three_dimensional = np.asarray([[0.5, 0.5, 0.5]])

    assert np.isclose(_hypervolume(two_dimensional), 0.47)
    assert np.isclose(_hypervolume(three_dimensional), 0.125)


def test_combined_front_contribution_and_directional_coverage():
    tpe = np.asarray([[0.8, 0.8], [0.4, 0.9]])
    random = np.asarray([[0.7, 0.7], [0.9, 0.4], [0.4, 0.9]])

    contributions = _combined_front_contributions(tpe, random)

    assert contributions == {"tpe": 1, "random": 1, "shared": 1, "total": 3}
    assert np.isclose(_set_coverage(tpe, random), 1 / 3)
    assert np.isclose(_set_coverage(random, tpe), 0.0)


def test_dynamic_conclusions_summarize_available_comparisons():
    quality_rows = [
        {
            "scenario": "final_only",
            "budget": 50,
            "hypervolume_difference": 0.2,
            "coverage_tpe_random": 0.8,
            "coverage_random_tpe": 0.1,
            "tpe_contribution": 0.75,
            "random_contribution": 0.25,
        }
    ]
    runtime_rows = [
        {
            "scenario": "final_only",
            "runtime_ratio": 0.8,
        }
    ]

    pareto_text = " ".join(_pareto_summary(quality_rows))
    runtime_text = " ".join(_runtime_summary(runtime_rows))

    assert "TPE has higher hypervolume in **1**" in pareto_text
    assert "TPE contributes a larger fraction" in pareto_text
    assert "TPE is faster in **1 of 1**" in runtime_text
