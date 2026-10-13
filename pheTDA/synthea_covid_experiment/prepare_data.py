"""Create one leakage-controlled row per Synthea COVID-19 patient."""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    CONDITION_GROUPS,
    CONTINUOUS_FEATURES,
    COVID_CODE,
    COVID_HOSPITAL_CODE,
    FEATURE_COLUMNS,
    ICU_CODE,
    MIN_CONDITION_PREVALENCE,
    OUTCOME_WINDOW_DAYS,
    PREPARATION_VERSION,
    SYMPTOM_CODES,
    VENTILATION_CODE,
    VITAL_CODES,
)

SOURCE_FILES = (
    "patients.csv",
    "conditions.csv",
    "observations.csv",
    "encounters.csv",
    "procedures.csv",
    "medications.csv",
)


def _source_fingerprint(source_directory):
    return {
        name: {
            "size": (source_directory / name).stat().st_size,
            "mtime_ns": (source_directory / name).stat().st_mtime_ns,
        }
        for name in SOURCE_FILES
    }


def _cache_is_current(output_path, metadata_path, source_directory):
    if not output_path.exists() or not metadata_path.exists():
        return False
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return metadata.get("preparation_version") == PREPARATION_VERSION and metadata.get(
        "source_fingerprint"
    ) == _source_fingerprint(source_directory)


def _increment_counts(target, values):
    counts = values.value_counts()
    target.loc[counts.index] = target.loc[counts.index].add(counts, fill_value=0)


def _load_day_zero_vitals(source_directory, index_dates):
    code_to_column = {code: column for column, code in VITAL_CODES.items()}
    rows = []
    row_offset = 0
    usecols = ["DATE", "PATIENT", "CODE", "VALUE"]
    for chunk in pd.read_csv(
        source_directory / "observations.csv",
        dtype=str,
        usecols=usecols,
        chunksize=100_000,
    ):
        chunk = chunk[
            chunk["CODE"].isin(code_to_column)
            & chunk["PATIENT"].isin(index_dates.index)
        ].copy()
        if chunk.empty:
            continue
        chunk["date"] = pd.to_datetime(chunk["DATE"], errors="coerce").dt.normalize()
        chunk["index_date"] = chunk["PATIENT"].map(index_dates).dt.normalize()
        chunk = chunk[chunk["date"].eq(chunk["index_date"])]
        if chunk.empty:
            continue
        chunk["value"] = pd.to_numeric(chunk["VALUE"], errors="coerce")
        chunk["feature"] = chunk["CODE"].map(code_to_column)
        chunk["row_order"] = np.arange(row_offset, row_offset + len(chunk))
        row_offset += len(chunk)
        rows.append(chunk[["PATIENT", "feature", "value", "row_order"]])

    if not rows:
        raise ValueError("no day-zero vital observations were found")
    observations = pd.concat(rows, ignore_index=True)
    observations = observations.sort_values("row_order").drop_duplicates(
        ["PATIENT", "feature"], keep="first"
    )
    return observations.pivot(index="PATIENT", columns="feature", values="value")


def _encounter_features_and_outcomes(source_directory, index_dates):
    patient_index = index_dates.index
    prior_encounters = pd.Series(0, index=patient_index, dtype=int)
    prior_emergency = pd.Series(0, index=patient_index, dtype=int)
    prior_inpatient = pd.Series(0, index=patient_index, dtype=int)
    hospitalized = set()
    intensive_care = set()
    window = pd.Timedelta(days=OUTCOME_WINDOW_DAYS)

    usecols = [
        "START",
        "PATIENT",
        "ENCOUNTERCLASS",
        "CODE",
        "REASONCODE",
    ]
    for chunk in pd.read_csv(
        source_directory / "encounters.csv",
        dtype=str,
        usecols=usecols,
        chunksize=100_000,
    ):
        chunk = chunk[chunk["PATIENT"].isin(patient_index)].copy()
        if chunk.empty:
            continue
        chunk["start"] = pd.to_datetime(
            chunk["START"], errors="coerce", utc=True
        ).dt.tz_localize(None)
        chunk["index_date"] = chunk["PATIENT"].map(index_dates)
        before = chunk["start"].lt(chunk["index_date"])
        _increment_counts(prior_encounters, chunk.loc[before, "PATIENT"])
        _increment_counts(
            prior_emergency,
            chunk.loc[before & chunk["ENCOUNTERCLASS"].eq("emergency"), "PATIENT"],
        )
        _increment_counts(
            prior_inpatient,
            chunk.loc[before & chunk["ENCOUNTERCLASS"].eq("inpatient"), "PATIENT"],
        )

        in_window = chunk["start"].between(
            chunk["index_date"], chunk["index_date"] + window
        )
        hospital_mask = (
            in_window
            & chunk["CODE"].eq(COVID_HOSPITAL_CODE)
            & chunk["REASONCODE"].eq(COVID_CODE)
        )
        hospitalized.update(chunk.loc[hospital_mask, "PATIENT"])
        intensive_care.update(
            chunk.loc[in_window & chunk["CODE"].eq(ICU_CODE), "PATIENT"]
        )

    return (
        prior_encounters,
        prior_emergency,
        prior_inpatient,
        hospitalized,
        intensive_care,
    )


def _ventilated_patients(source_directory, index_dates):
    patients = set()
    window = pd.Timedelta(days=OUTCOME_WINDOW_DAYS)
    for chunk in pd.read_csv(
        source_directory / "procedures.csv",
        dtype=str,
        usecols=["DATE", "PATIENT", "CODE"],
        chunksize=100_000,
    ):
        chunk = chunk[
            chunk["PATIENT"].isin(index_dates.index)
            & chunk["CODE"].eq(VENTILATION_CODE)
        ].copy()
        if chunk.empty:
            continue
        procedure_date = pd.to_datetime(chunk["DATE"], errors="coerce")
        index_date = chunk["PATIENT"].map(index_dates)
        in_window = procedure_date.between(index_date, index_date + window)
        patients.update(chunk.loc[in_window, "PATIENT"])
    return patients


def _active_medication_counts(source_directory, index_dates):
    active_codes = {patient: set() for patient in index_dates.index}
    usecols = ["START", "STOP", "PATIENT", "CODE"]
    for chunk in pd.read_csv(
        source_directory / "medications.csv",
        dtype=str,
        usecols=usecols,
        chunksize=100_000,
    ):
        chunk = chunk[chunk["PATIENT"].isin(index_dates.index)].copy()
        if chunk.empty:
            continue
        start = pd.to_datetime(chunk["START"], errors="coerce")
        stop = pd.to_datetime(chunk["STOP"], errors="coerce")
        index_date = chunk["PATIENT"].map(index_dates)
        active = start.lt(index_date) & (stop.isna() | stop.ge(index_date))
        for patient, codes in chunk.loc[active].groupby("PATIENT")["CODE"]:
            active_codes[patient].update(codes.dropna())
    return pd.Series(
        {patient: len(codes) for patient, codes in active_codes.items()}, dtype=int
    )


def prepare_dataset(source_directory, output_directory, force=False):
    source_directory = Path(source_directory)
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / "synthea_covid_experiment.csv"
    metadata_path = output_directory / "feature_manifest.json"
    if not force and _cache_is_current(output_path, metadata_path, source_directory):
        return output_path, metadata_path

    conditions = pd.read_csv(source_directory / "conditions.csv", dtype=str)
    conditions["start_date"] = pd.to_datetime(conditions["START"], errors="coerce")
    conditions["stop_date"] = pd.to_datetime(conditions["STOP"], errors="coerce")
    covid = conditions[conditions["CODE"].eq(COVID_CODE)]
    index_dates = covid.groupby("PATIENT")["start_date"].min().sort_index()
    if index_dates.empty:
        raise ValueError("the source contains no patients with the COVID-19 code")

    patients = pd.read_csv(source_directory / "patients.csv", dtype=str)
    patients = patients[patients["Id"].isin(index_dates.index)].set_index("Id")
    prepared = pd.DataFrame(index=index_dates.index)
    prepared.index.name = "patient_id"
    prepared["covid_index_date"] = index_dates
    birth_dates = pd.to_datetime(patients.loc[prepared.index, "BIRTHDATE"])
    prepared["age_at_covid"] = ((index_dates - birth_dates).dt.days / 365.2425).astype(
        float
    )
    gender = patients.loc[prepared.index, "GENDER"].map({"F": 0, "M": 1})
    if gender.isna().any():
        raise ValueError("GENDER contains values other than F and M")
    prepared["gender_male"] = gender.astype(int)

    vitals = _load_day_zero_vitals(source_directory, index_dates)
    prepared = prepared.join(vitals)

    conditions = conditions[conditions["PATIENT"].isin(index_dates.index)].copy()
    conditions["index_date"] = conditions["PATIENT"].map(index_dates)
    active = conditions[
        conditions["start_date"].lt(conditions["index_date"])
        & (
            conditions["stop_date"].isna()
            | conditions["stop_date"].ge(conditions["index_date"])
        )
    ]
    prepared["active_condition_count"] = (
        active.groupby("PATIENT")["CODE"]
        .nunique()
        .reindex(prepared.index, fill_value=0)
    )
    for feature, codes in CONDITION_GROUPS.items():
        positive = set(active.loc[active["CODE"].isin(codes), "PATIENT"])
        prepared[feature] = prepared.index.isin(positive).astype(int)

    day_zero_conditions = conditions[
        conditions["start_date"]
        .dt.normalize()
        .eq(conditions["index_date"].dt.normalize())
    ]
    for feature, code in SYMPTOM_CODES.items():
        positive = set(
            day_zero_conditions.loc[day_zero_conditions["CODE"].eq(code), "PATIENT"]
        )
        prepared[feature] = prepared.index.isin(positive).astype(int)

    (
        prior_encounters,
        prior_emergency,
        prior_inpatient,
        hospitalized,
        intensive_care,
    ) = _encounter_features_and_outcomes(source_directory, index_dates)
    prepared["prior_encounter_count"] = prior_encounters
    prepared["prior_emergency_count"] = prior_emergency
    prepared["prior_inpatient_count"] = prior_inpatient
    prepared["active_medication_count"] = _active_medication_counts(
        source_directory, index_dates
    )
    ventilated = _ventilated_patients(source_directory, index_dates)

    death_date = pd.to_datetime(
        patients.loc[prepared.index, "DEATHDATE"], errors="coerce"
    )
    death_in_window = death_date.between(
        index_dates, index_dates + pd.Timedelta(days=OUTCOME_WINDOW_DAYS)
    )
    severe = (
        prepared.index.isin(intensive_care)
        | prepared.index.isin(ventilated)
        | death_in_window.to_numpy()
    )
    moderate = prepared.index.isin(hospitalized) & ~severe
    prepared["final_severity_3"] = np.select(
        [severe, moderate], ["severe", "moderate"], default="mild"
    )

    required_complete = [*VITAL_CODES, *CONTINUOUS_FEATURES]
    before_complete_cases = len(prepared)
    prepared = prepared.dropna(subset=list(dict.fromkeys(required_complete))).copy()
    prepared["oxygen_saturation_tertile"] = pd.qcut(
        prepared["oxygen_saturation_day0"],
        q=3,
        labels=["low", "middle", "high"],
    ).astype(str)

    prevalence = prepared[list(CONDITION_GROUPS)].mean()
    rare_groups = prevalence[prevalence < MIN_CONDITION_PREVALENCE]
    if not rare_groups.empty:
        details = ", ".join(
            f"{name}={value:.3%}" for name, value in rare_groups.items()
        )
        raise ValueError(
            "configured condition groups below prevalence threshold: " + details
        )
    if prepared[FEATURE_COLUMNS].isna().any().any():
        raise ValueError("prepared feature matrix contains missing values")

    output_columns = [
        "covid_index_date",
        "oxygen_saturation_day0",
        "oxygen_saturation_tertile",
        "final_severity_3",
        *FEATURE_COLUMNS,
    ]
    prepared = prepared[output_columns].reset_index()
    prepared.to_csv(output_path, index=False)

    metadata = {
        "preparation_version": PREPARATION_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_directory": str(source_directory.resolve()),
        "source_fingerprint": _source_fingerprint(source_directory),
        "covid_patients_before_complete_case_filter": before_complete_cases,
        "prepared_patients": len(prepared),
        "dropped_for_missing_day_zero_values": before_complete_cases - len(prepared),
        "feature_count": len(FEATURE_COLUMNS),
        "continuous_features": CONTINUOUS_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "binary_features": BINARY_FEATURES,
        "condition_groups": {
            name: sorted(codes) for name, codes in CONDITION_GROUPS.items()
        },
        "condition_prevalence": prevalence.to_dict(),
        "symptom_codes": SYMPTOM_CODES,
        "phenotypes": {
            "initial_categorical": "oxygen_saturation_tertile",
            "initial_continuous": "oxygen_saturation_day0",
            "final": "final_severity_3",
            "outcome_window_days": OUTCOME_WINDOW_DAYS,
        },
        "phenotype_counts": {
            "oxygen_saturation_tertile": prepared["oxygen_saturation_tertile"]
            .value_counts()
            .to_dict(),
            "final_severity_3": prepared["final_severity_3"].value_counts().to_dict(),
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return output_path, metadata_path
