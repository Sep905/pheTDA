"""Command-line entry point for the semi-supervised TDA pipeline."""

import argparse
import ast
from pathlib import Path

import numpy as np
import optuna
import pandas as pd

from optuna_pipeline import (
    SemiSupervised_TDA_pipeline,
    save_ensemble_memberships,
    save_selection_report,
    select_pareto_trials,
    validate_selection_options,
)
from utils import distance_matrix_computation

PIPELINE_NAME = "SemiSupervisedTDA"


def _feature_list(value):
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            "feature lists must use Python-list syntax, for example ['age', 'BMI']"
        ) from exc
    if not isinstance(parsed, list) or not all(
        isinstance(column, str) for column in parsed
    ):
        raise argparse.ArgumentTypeError("features must be a list of column names")
    return parsed


def _read_dataset(path):
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xls", ".xlsx"}:
        return pd.read_excel(path)
    raise ValueError("dataset_path must point to a CSV or Excel file")


def _load_or_compute_preprocessing(args, dataset):
    distance_path = args.results_path / "distance_matrix.npy"
    features_path = args.results_path / "dataset_preprocessed.csv"
    if args.reuse_preprocessing and distance_path.exists() and features_path.exists():
        distance_matrix = np.load(distance_path, allow_pickle=False)
        features = pd.read_csv(features_path)
        if distance_matrix.shape != (len(dataset), len(dataset)):
            raise ValueError(
                "cached distance matrix is not aligned with the current dataset"
            )
        return distance_matrix, features

    return distance_matrix_computation(
        dataset,
        args.continuous_features,
        args.categorical_features,
        args.binary_features,
        args.results_path,
    )


def main(args):
    validate_selection_options(args.solution_mode, args.ensemble_percentage)
    dataset = _read_dataset(args.dataset_path)
    phenotype_columns = {
        "initial": args.initial_class,
        "final": args.final_class,
    }
    supplied_phenotypes = {
        name: column for name, column in phenotype_columns.items() if column is not None
    }
    if not supplied_phenotypes:
        raise ValueError("supply --initial_class, --final_class, or both")
    for name, column in supplied_phenotypes.items():
        if column not in dataset.columns:
            raise ValueError(
                f"{name} phenotype column {column!r} is absent from the dataset"
            )

    selected_features = {
        *args.continuous_features,
        *args.categorical_features,
        *args.binary_features,
    }
    phenotype_features = selected_features.intersection(supplied_phenotypes.values())
    if phenotype_features:
        raise ValueError(
            "phenotype columns cannot also be used as input features: "
            f"{sorted(phenotype_features)}"
        )

    args.results_path.mkdir(parents=True, exist_ok=True)
    distance_matrix, features = _load_or_compute_preprocessing(args, dataset)
    sample_ids = pd.Series(np.arange(len(dataset)), name="dataset_id")

    objective = SemiSupervised_TDA_pipeline(
        distance_matrix=distance_matrix,
        dataset=features,
        random_seed=args.seed,
        sample_ids=sample_ids,
        results_path=args.results_path,
        initial_class=(
            dataset[args.initial_class] if args.initial_class is not None else None
        ),
        initial_class_type=args.initial_class_type,
        initial_entropy=args.initial_entropy,
        final_class=(
            dataset[args.final_class] if args.final_class is not None else None
        ),
        final_class_type=args.final_class_type,
        final_entropy=args.final_entropy,
        pipeline_type_name=PIPELINE_NAME,
    )

    directions = []
    metric_names = []
    if args.initial_class is not None:
        directions.append(args.initial_entropy)
        metric_names.append("initial_graph_entropy")
    if args.final_class is not None:
        directions.append(args.final_entropy)
        metric_names.append("final_graph_entropy")
    directions.append("maximize")
    metric_names.append("stratification_silhouette")
    sampler = optuna.samplers.TPESampler(
        seed=args.seed,
        n_startup_trials=min(args.n_startup_trials, args.n_trials),
        multivariate=True,
        group=True,
    )
    study = optuna.create_study(directions=directions, sampler=sampler)
    study.set_metric_names(metric_names)
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

    output_directory = args.results_path / f"{args.seed}_{PIPELINE_NAME}"
    study.trials_dataframe().to_excel(output_directory / "df_results.xlsx", index=False)
    selected_trials = select_pareto_trials(
        study,
        mode=args.solution_mode,
        ensemble_percentage=args.ensemble_percentage,
    )
    selection_directory = output_directory / "selection"
    save_selection_report(
        selected_trials,
        metric_names,
        selection_directory,
        args.solution_mode,
    )
    if args.solution_mode == "ensemble":
        save_ensemble_memberships(
            selected_trials,
            objective.results_path_scomplex,
            selection_directory,
        )


def build_parser():
    parser = argparse.ArgumentParser(prog="TDA")
    parser.add_argument("--dataset_path", type=Path, required=True)
    parser.add_argument("--initial_class")
    parser.add_argument(
        "--initial_class_type",
        choices=("categorical", "continuous"),
        default="categorical",
    )
    parser.add_argument("--final_class")
    parser.add_argument(
        "--final_class_type",
        choices=("categorical", "continuous"),
        default="categorical",
    )
    parser.add_argument("--seed", type=int, default=203)
    parser.add_argument("--results_path", type=Path, default=Path("results"))
    parser.add_argument(
        "--continuous_features",
        "--continue_features",
        dest="continuous_features",
        type=_feature_list,
        default=[],
    )
    parser.add_argument("--categorical_features", type=_feature_list, default=[])
    parser.add_argument("--binary_features", type=_feature_list, default=[])
    parser.add_argument(
        "--initial_entropy",
        "--entropy",
        dest="initial_entropy",
        choices=("minimize", "maximize"),
        default="minimize",
    )
    parser.add_argument(
        "--final_entropy",
        choices=("minimize", "maximize"),
        default="minimize",
    )
    parser.add_argument("--n_trials", type=int, default=1000)
    parser.add_argument("--n_startup_trials", type=int, default=500)
    parser.add_argument(
        "--solution_mode",
        choices=("single", "ensemble"),
        default="single",
        help="select one Pareto solution or a fraction for a future ensemble",
    )
    parser.add_argument(
        "--ensemble_percentage",
        type=float,
        default=None,
        help="fraction of the Pareto front to select (0, 1]; 0.25 means 25%%",
    )
    parser.add_argument(
        "--reuse_preprocessing",
        action="store_true",
        help="reuse distance_matrix.npy and dataset_preprocessed.csv if present",
    )
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
