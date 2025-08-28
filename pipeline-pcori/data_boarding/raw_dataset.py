from enum import Enum
import pandas as pd
from pathlib import Path

class Label(Enum):
    POS = "POS"
    NEG = "NEG"


class DataCategory(Enum):
    DX = "dx"
    MED = "med"
    LAB = "lab"
    CLINICAL_EVENTS = "clinical_events"
    AGE = "age"


DX_TYPE = {
    "encounter_id": int,  # Encounter/session identifier
    "patient_sk": int,  # Surrogate key for patient (anonymized ID)
    "hospital_id": int,  # ID of the hospital
    "patient_type_id": int,  # Inpatient/outpatient etc. (likely coded integer)
    "admitted_dt_tm": pd.Timestamp,  # Timestamp of admission
    "diagnosis_id": int,  # Internal diagnosis row ID
    "diagnosis_code": str,  # ICD code or similar (textual)
}

MED_PARTIAL_TYPE = {
    "encounter_id": int,
    "hospital_id": int,
    "patient_id": int,
    "patient_type_id": int,
    "admitted_dt_id": int,
    "admitted_dt_tm": pd.Timestamp,
    "medication_id": int,
    "frequency_id": int,
    "med_started_dt_id": int,
    "med_entered_dt_id": int,
    "med_stopped_dt_id": int,
    "med_discontinued_dt_id": int,
    "total_dispensed_doses": float,
    "dose_quantity": float,
    "initial_dose_quantity": float,
    "dose_units_id": int,
    "charge_quantity": float,
    "credit_quantity": float,
    "infusion_rate": float,
    "infusion_time": float,
    "infusion_time_units_id": int,
    "order_strength": float,
    "order_strength_units_id": int,
    "order_volume": float,
    "order_volume_units_id": int,
    "total_volume": float,
    "unit_cost": float,
    "unit_price": float,
    "med_started_dt_tm": pd.Timestamp,
    "med_entered_dt_tm": pd.Timestamp,
    "med_stopped_dt_tm": pd.Timestamp,
    "med_discontinued_dt_tm": pd.Timestamp,
    "med_started_tm_valid_flg": bool,
    "med_entered_tm_valid_flg": bool,
    "med_stopped_tm_valid_flg": bool,
    "med_discontinued_tm_valid_flg": bool,
    "medication_id__hf_d_medication_": int,
    "ndc_code": str,
    "brand_name": str,
    "product_strength_description": str,
    "obsolete_dt_tm": pd.Timestamp,
}

CE_TYPE = {
    "encounter_id": int,
    "event_code_id": int,
    "event_source_id": int,
    "result_value_num": float,
    "event_code_desc": str,
    "event_code_display": str,
}

LAB_TYPE = {
    "encounter_id": int,
    "patient_sk": str,
    "lab_id": str,
    "result_indicator_desc": str,
}

AGE_TYPE = {
    "encounter_id": int,
    "age_in_years": int,
}

TYPE_DICT = {
    DataCategory.MED: MED_PARTIAL_TYPE,
    DataCategory.CLINICAL_EVENTS: CE_TYPE,
    DataCategory.LAB: LAB_TYPE,
    DataCategory.AGE: AGE_TYPE,
    DataCategory.DX: DX_TYPE,
}

FILENAME_MAP = {
    DataCategory.DX: "_dx_partial",
    DataCategory.MED: "_med_partial",
    DataCategory.LAB: "_lab_partial",
    DataCategory.CLINICAL_EVENTS: "_ce_partial",
    DataCategory.AGE: "_age",
}

PREFIX_MAP = {
    Label.POS: "pos",
    Label.NEG: "neg",
}


def convert_column(series, target_type):
    if target_type is pd.Timestamp:
        return pd.to_datetime(series, errors="coerce")
    elif target_type is bool:
        return series.astype("boolean")  # preserves NA
    elif target_type is int:
        return pd.to_numeric(series, errors="coerce").astype("Int64")  # NA-safe int
    elif target_type is float:
        return pd.to_numeric(series, errors="coerce")
    elif target_type is str:
        return series.astype(str)
    else:
        return series


def convert_dataframe(df, column_type_map):
    cols_to_keep = [col for col in column_type_map if col in df.columns]
    df_converted = df[cols_to_keep].copy()

    for col in cols_to_keep:
        target_type = column_type_map[col]
        df_converted[col] = convert_column(df_converted[col], target_type)

    return df_converted


def get_opioid_impl(dataset_dir: Path, label: Label, data_category: DataCategory) -> pd.DataFrame:
    """
    Get the opioid data.
    """

    filename = f"{dataset_dir}/{PREFIX_MAP[label]}{FILENAME_MAP[data_category]}.pqt"
    df = pd.read_parquet(filename)
    df = convert_dataframe(df, TYPE_DICT[data_category])
    index = ["encounter_id"]
    if "patient_id" in df.columns:
        index.append("patient_id")
    return df.set_index(index)