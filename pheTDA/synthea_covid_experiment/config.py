"""Shared configuration for the Synthea COVID-19 experiments."""

from dataclasses import dataclass

PREPARATION_VERSION = 1
COVID_CODE = "840539006"
COVID_HOSPITAL_CODE = "1505002"
ICU_CODE = "305351004"
VENTILATION_CODE = "26763009"
OUTCOME_WINDOW_DAYS = 90
MIN_CONDITION_PREVALENCE = 0.01

TRIAL_BUDGETS = ((50, 10), (100, 15), (250, 25), (500, 50))
UNSUPERVISED_BASELINE_NAME = "unsupervised_reduction_clustering"

VITAL_CODES = {
    "oxygen_saturation_day0": "2708-6",
    "temperature_day0": "8310-5",
    "respiratory_rate_day0": "9279-1",
    "heart_rate_day0": "8867-4",
    "diastolic_bp_day0": "8462-4",
    "systolic_bp_day0": "8480-6",
    "weight_day0": "29463-7",
}

# These are explicit SNOMED CT groupings. The source does not contain ICD-9
# codes, so grouping condition codes by their first digit would be invalid.
CONDITION_GROUPS = {
    "condition_obesity": {"162864005", "408512008"},
    "condition_prediabetes": {"15777000"},
    "condition_anemia": {"271737000"},
    "condition_hypertension": {"59621000"},
    "condition_chronic_sinusitis": {"40055000"},
    "condition_hyperlipidemia": {"55822004"},
    "condition_diabetes": {"44054006"},
    "condition_hypertriglyceridemia": {"302870006"},
    "condition_metabolic_syndrome": {"237602007"},
    "condition_osteoarthritis": {"239873007", "239872002", "201834006"},
    "condition_chronic_pain": {"82423001", "95417003"},
    "condition_osteoporosis": {"64859006"},
    "condition_stroke": {"230690007"},
    "condition_coronary_heart_disease": {"53741008"},
    "condition_cardiac_arrest_history": {"429007001"},
    "condition_atrial_fibrillation": {"49436004"},
    "condition_chronic_kidney_disease": {
        "431855005",
        "431856006",
        "127013003",
        "90781000119102",
    },
    "condition_seizure_disorder": {"128613002", "84757009", "703151001"},
    "condition_cancer_history": {"254837009", "126906006", "92691004"},
    "condition_asthma": {"233678006"},
    "condition_dementia": {"26929004", "230265002"},
    "condition_copd": {"87433001", "185086009"},
}

# Passive conjunctival congestion occurs in fewer than 1% of the cohort and
# is intentionally excluded. The remaining symptom indicators exceed 1%.
SYMPTOM_CODES = {
    "symptom_nasal_congestion": "68235000",
    "symptom_headache": "25064002",
    "symptom_cough": "49727002",
    "symptom_sore_throat": "267102003",
    "symptom_sputum": "248595008",
    "symptom_fatigue": "84229001",
    "symptom_hemoptysis": "66857006",
    "symptom_dyspnea": "267036007",
    "symptom_nausea": "422587007",
    "symptom_diarrhea": "267060006",
    "symptom_muscle_pain": "68962001",
    "symptom_joint_pain": "57676002",
    "symptom_chills": "43724002",
    "symptom_loss_of_taste": "36955009",
}

CONTINUOUS_FEATURES = [
    "age_at_covid",
    "temperature_day0",
    "respiratory_rate_day0",
    "heart_rate_day0",
    "diastolic_bp_day0",
    "systolic_bp_day0",
    "weight_day0",
    "prior_encounter_count",
    "prior_emergency_count",
    "prior_inpatient_count",
    "active_condition_count",
    "active_medication_count",
]
BINARY_FEATURES = [
    "gender_male",
    *CONDITION_GROUPS,
    *SYMPTOM_CODES,
]
CATEGORICAL_FEATURES = []
FEATURE_COLUMNS = [*CONTINUOUS_FEATURES, *BINARY_FEATURES]


@dataclass(frozen=True)
class Scenario:
    name: str
    initial_column: str = None
    initial_type: str = "categorical"
    initial_direction: str = "minimize"
    final_column: str = None
    final_direction: str = "minimize"


def _variant_scenarios(variant, initial_column, initial_type):
    return [
        Scenario(
            name=f"{variant}_initial_only",
            initial_column=initial_column,
            initial_type=initial_type,
        ),
        Scenario(
            name=f"{variant}_both_alignment",
            initial_column=initial_column,
            initial_type=initial_type,
            final_column="final_severity_3",
        ),
        Scenario(
            name=f"{variant}_initial_mixing_final_separation",
            initial_column=initial_column,
            initial_type=initial_type,
            initial_direction="maximize",
            final_column="final_severity_3",
        ),
    ]


SCENARIOS = [
    Scenario(
        name="final_only",
        final_column="final_severity_3",
    ),
    *_variant_scenarios("categorical", "oxygen_saturation_tertile", "categorical"),
    *_variant_scenarios("continuous", "oxygen_saturation_day0", "continuous"),
]
SCENARIOS_BY_NAME = {scenario.name: scenario for scenario in SCENARIOS}
