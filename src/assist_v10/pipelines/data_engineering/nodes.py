"""
Data Engineering nodes for Assist v10.

This module prepares the base datasets for:
- HIS-05: saturation / wait-time forecasting.
- HIS-10: no-show classification.

The goal of this pipeline is not to train models, but to clean raw hospital tables
and create modeling-ready datasets for the Data Science pipelines.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


# =========================================================
# Generic helpers
# =========================================================

def _strip_string_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Strip leading/trailing spaces from text columns and convert empty strings to NA.
    """

    df = df.copy()

    # Raw Assist tables contain many padded strings and blank values.
    # This standardizes text fields before joins, filtering, and feature creation.
    for col in df.columns:
        if df[col].dtype == "object" or str(df[col].dtype).startswith("string"):
            df[col] = df[col].astype("string").str.strip().replace("", pd.NA)

    return df


def _standardize_time_hhmmss(series: pd.Series) -> pd.Series:
    """
    Standardize time strings to HHMMSS.

    Examples:
    - '0900'   -> '090000'
    - '93000'  -> '093000'
    - '093000' -> '093000'
    """

    # Raw time columns are not always consistent.
    # Some values come as HHMM, others as HHMMSS, and some may lose leading zeros.
    s = series.astype("string").str.strip().replace("", pd.NA)
    s = s.str.replace(r"\.0$", "", regex=True)
    s = s.str.replace(r"\D", "", regex=True)

    def fix_time(value):
        """
        Normalize one raw time value into a six-character HHMMSS string.
        """
        if pd.isna(value):
            return pd.NA

        value = str(value)

        if len(value) == 6:
            return value
        if len(value) == 5:
            return "0" + value
        if len(value) == 4:
            return value + "00"
        if len(value) == 3:
            return "0" + value + "00"
        if len(value) == 2:
            return value + "0000"
        if len(value) == 1:
            return "0" + value + "0000"

        return value.zfill(6)[-6:]

    return s.map(fix_time).astype("string")


def _parse_datetime_from_date_time(
    df: pd.DataFrame,
    date_col: str,
    time_col: str,
) -> pd.Series:
    """
    Parse datetime from date column YYYYMMDD and time column HHMMSS/HHMM.
    Invalid values become NaT.
    """

    # If a table does not contain the expected date/time columns,
    # return NaT instead of breaking the pipeline.
    if date_col not in df.columns or time_col not in df.columns:
        return pd.Series(pd.NaT, index=df.index)

    date_s = df[date_col].astype("string").str.strip().replace("", pd.NA)
    date_s = date_s.str.replace(r"\.0$", "", regex=True)
    date_s = date_s.str.replace(r"\D", "", regex=True)

    time_s = _standardize_time_hhmmss(df[time_col])

    raw = date_s + time_s

    return pd.to_datetime(raw, format="%Y%m%d%H%M%S", errors="coerce")


def _safe_numeric(series: pd.Series) -> pd.Series:
    """
    Convert a column to numeric values.

    Invalid values are converted to NaN instead of breaking the pipeline.
    Numeric missing values are not imputed here because imputation should be
    decided later by the Data Science/modeling step.
    """
    return pd.to_numeric(series.astype("string").str.strip(), errors="coerce")


def _add_temporal_features(df: pd.DataFrame, datetime_col: str, prefix: str = "") -> pd.DataFrame:
    """
    Add calendar-based features from a datetime column.

    Created features:
    - hour
    - day_of_week
    - day
    - month
    - is_weekend

    These features help models capture hourly, daily, and weekly patterns.
    """
    df = df.copy()

    col_prefix = f"{prefix}_" if prefix else ""

    df[f"{col_prefix}hour"] = df[datetime_col].dt.hour
    df[f"{col_prefix}day_of_week"] = df[datetime_col].dt.dayofweek
    df[f"{col_prefix}day"] = df[datetime_col].dt.day
    df[f"{col_prefix}month"] = df[datetime_col].dt.month
    df[f"{col_prefix}is_weekend"] = df[f"{col_prefix}day_of_week"].isin([5, 6]).astype(int)

    return df


def _fill_categorical_missing(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """
    Fill missing categorical columns with 'UNKNOWN'.
    """
    df = df.copy()

    # For categorical variables, missing values are kept as an explicit category.
    # This is useful for tree-based models such as CatBoost.
    # Numeric imputation is intentionally not done here; that belongs to modeling decisions.
    for col in columns:
        if col in df.columns:
            df[col] = df[col].astype("string").fillna("UNKNOWN")

    return df


# =========================================================
# Cleaning nodes
# =========================================================

def clean_hospac(hospac: pd.DataFrame) -> pd.DataFrame:
    """
    Clean HOSPAC patient/encounter table.

    This table is used as a bridge between appointment keys and patient information.
    """
    df = _strip_string_columns(hospac)

    # Create normalized join keys.
    # These keys are later used to connect HOSPAC with HOSAGD without duplicating appointments.
    df["area_key"] = df["p_area"].astype("string").str.strip()
    df["cve_num_key"] = df["p_res_cve_num"].astype("string").str.strip()
    df["cve_mbo_key"] = df["p_res_cve_mbo"].astype("string").str.strip()
    df["p_num_exp_key"] = df["p_num_exp"].astype("string").str.strip()

    # Parse date/time fields into real datetime columns.
    # These are useful for lead-time features and operational timing analysis.
    df["reservation_datetime"] = _parse_datetime_from_date_time(df, "p_res_fec", "p_res_hra")
    df["arrival_datetime"] = _parse_datetime_from_date_time(df, "p_fec_lld", "p_hra_lld")
    df["registration_datetime"] = _parse_datetime_from_date_time(df, "p_fec_reg", "p_hra_reg")

    # Keep only columns required for joins, feature creation, or downstream modeling.
    # This reduces noise from raw operational tables with many unused fields.
    useful_cols = [
        "area_key",
        "cve_num_key",
        "cve_mbo_key",
        "p_num_exp_key",
        "p_area",
        "p_status",
        "p_sexo",
        "p_tpo_pac",
        "p_tpo_cita",
        "reservation_datetime",
        "arrival_datetime",
        "registration_datetime",
    ]

    useful_cols = [col for col in useful_cols if col in df.columns]

    return df[useful_cols].drop_duplicates()


def clean_hosagd(hosagd: pd.DataFrame) -> pd.DataFrame:
    """
    Clean HOSAGD appointment schedule table.

    This is the main table for HIS-10 No-Show Guard because it contains
    appointment scheduling information and the raw asistencia field used to
    build the no_show target.
    """
    df = _strip_string_columns(hosagd)

    df["area_key"] = df["area"].astype("string").str.strip()
    df["cve_num_key"] = df["cve_num"].astype("string").str.strip()
    df["cve_mbo_key"] = df["cve_mbo"].astype("string").str.strip()

    # Build the appointment timestamp and temporal features.
    # These features are available before the appointment and are safe for no-show prediction.
    df["appointment_datetime"] = _parse_datetime_from_date_time(df, "a_fecha", "hra_ini")
    df = _add_temporal_features(df, "appointment_datetime", prefix="appointment")

    df["duration_min"] = _safe_numeric(df["dur"])

    df["asistencia_clean"] = df["asistencia"].astype("string").str.strip().replace("", pd.NA)

    # HIS-10 target construction:
    # A = attended appointment.
    # I = no-show / missed appointment.
    # Missing asistencia values are NOT imputed because the real outcome is unknown.
    # Those rows are excluded later from the initial training dataset.
    df["no_show"] = df["asistencia_clean"].map({"A": 0, "I": 1})

    # Keep only columns required for joins, feature creation, or downstream modeling.
    # This reduces noise from raw operational tables with many unused fields.
    useful_cols = [
        "area_key",
        "cve_num_key",
        "cve_mbo_key",
        "area",
        "cve_num",
        "cve_mbo",
        "med",
        "esp",
        "a_fecha",
        "hra_ini",
        "hra_fin",
        "appointment_datetime",
        "appointment_hour",
        "appointment_day_of_week",
        "appointment_day",
        "appointment_month",
        "appointment_is_weekend",
        "duration_min",
        "tpo_cita",
        "conflicto",
        "agregada",
        "ultimahora",
        "buffer",
        "asistencia_clean",
        "no_show",
    ]

    useful_cols = [col for col in useful_cols if col in df.columns]

    return df[useful_cols].drop_duplicates()


def clean_hosmpi(hosmpi: pd.DataFrame) -> pd.DataFrame:
    """
    Clean HOSMPI master patient index table.

    This table provides demographic enrichment for HIS-10.
    """
    df = _strip_string_columns(hosmpi)

    # Normalize patient ID to join demographic information with HOSPAC encounters.
    df["m_num_exp_key"] = df["m_num_exp"].astype("string").str.strip()

    # Age is converted to numeric, but missing values are not imputed here.
    # The Data Science pipeline can decide whether to impute, filter, or let the model handle NaN.
    if "m_edad" in df.columns:
        df["m_edad_num"] = _safe_numeric(df["m_edad"])

    if "m_fec_nac" in df.columns:
        df["m_fec_nac_clean"] = pd.to_datetime(
            df["m_fec_nac"].astype("string").str.strip(),
            format="%Y%m%d",
            errors="coerce",
        )

    # Keep only columns required for joins, feature creation, or downstream modeling.
    # This reduces noise from raw operational tables with many unused fields.
    useful_cols = [
        "m_num_exp_key",
        "m_status",
        "m_sexo",
        "m_edad_num",
        "m_fec_nac_clean",
        "m_ciu",
        "m_col",
        "m_cp",
        "m_edo",
        "m_pai",
    ]

    useful_cols = [col for col in useful_cols if col in df.columns]

    return df[useful_cols].drop_duplicates(subset=["m_num_exp_key"])


def clean_triage(triage: pd.DataFrame) -> pd.DataFrame:
    """
    Clean TRIAGE emergency classification table.

    This table is used as optional enrichment for HIS-05 saturation monitoring.
    """
    df = _strip_string_columns(triage)

    df["expediente_key"] = df["Expediente"].astype("string").str.strip()
    df["clave_ingreso_key"] = df["ClaveIngreso"].astype("string").str.strip()

    # Triage records are used to enrich HIS-05 with emergency severity volume by hour.
    df["triage_datetime"] = _parse_datetime_from_date_time(df, "Fecha", "Hora")
    df["triage_clean"] = df["Triage"].astype("string").str.strip().replace("", pd.NA)

    # Keep only columns required for joins, feature creation, or downstream modeling.
    # This reduces noise from raw operational tables with many unused fields.
    useful_cols = [
        "expediente_key",
        "clave_ingreso_key",
        "triage_datetime",
        "Edad",
        "Sexo",
        "Area",
        "Departamento",
        "NomDepartamento",
        "LlegadaServicio",
        "MotivoConsulta",
        "triage_clean",
        "Destino",
        "TiempoEvolucion",
    ]

    useful_cols = [col for col in useful_cols if col in df.columns]

    return df[useful_cols].drop_duplicates()


def clean_notamedicaurg(notamedicaurg: pd.DataFrame) -> pd.DataFrame:
    """
    Clean NOTAMEDICAURG emergency medical notes table.

    This table is used for HIS-05 because it contains emergency arrival and note
    timestamps. Since AtMed_Hora is empty in the raw data, this node creates a
    wait-time proxy:
    wait_proxy_min = note_datetime - arrival_datetime.
    """

    df = _strip_string_columns(notamedicaurg)

    df["expediente_key"] = df["Expediente"].astype("string").str.strip()
    df["clave_ingreso_key"] = df["ClaveIngreso"].astype("string").str.strip()

    df["arrival_datetime"] = _parse_datetime_from_date_time(df, "Llegada_Fecha", "Llegada_Hora")
    df["note_datetime"] = _parse_datetime_from_date_time(df, "Fecha", "Hora")
    df["destination_datetime"] = _parse_datetime_from_date_time(df, "Destino_Fecha", "Destino_Hora")

    # AtMed_Hora is empty in the raw NOTAMEDICAURG table, so the official
    # medical attention timestamp is unavailable.
    #
    # For HIS-05, we create a wait-time proxy:
    # wait_proxy_min = note_datetime - arrival_datetime
    #
    # This proxy should be interpreted as an approximation of operational delay,
    # not as the official clinical waiting time.
    df["wait_proxy_min"] = (
        df["note_datetime"] - df["arrival_datetime"]
    ).dt.total_seconds() / 60

    # Keep only plausible proxy values for aggregation.
    # Negative values and waits longer than 24 hours are treated as invalid.
    df["valid_wait_proxy"] = (
        df["wait_proxy_min"].notna()
        & (df["wait_proxy_min"] >= 0)
        & (df["wait_proxy_min"] <= 24 * 60)
    ).astype(int)

    # Keep only columns required for joins, feature creation, or downstream modeling.
    # This reduces noise from raw operational tables with many unused fields.
    useful_cols = [
        "expediente_key",
        "clave_ingreso_key",
        "arrival_datetime",
        "note_datetime",
        "destination_datetime",
        "wait_proxy_min",
        "valid_wait_proxy",
        "Edad",
        "Sexo",
        "Especialidad",
        "LlegadaServicio",
        "MotivoConsulta",
        "Triage",
        "Salida",
    ]

    useful_cols = [col for col in useful_cols if col in df.columns]

    return df[useful_cols].drop_duplicates()


# =========================================================
# Feature table nodes
# =========================================================

def create_his10_base(
    processed_hosagd: pd.DataFrame,
    processed_hospac: pd.DataFrame,
    processed_hosmpi: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create HIS-10 base dataset for no-show classification.

    Output grain: one row per appointment with known attendance label.
    """
    appointments = processed_hosagd.copy()
    encounters = processed_hospac.copy()
    patients = processed_hosmpi.copy()

    # Correct join key between HOSAGD and HOSPAC:
    # area + cve_num + cve_mbo.
    #
    # During exploration, joining only by cve_num + cve_mbo duplicated appointments
    # because those keys repeat across hospital areas/sites.
    key_cols = ["area_key", "cve_num_key", "cve_mbo_key"]

    # Ensure one HOSPAC record per appointment key.
    # This protects the HIS-10 dataset from duplicated appointment rows.
    encounters_by_key = (
        encounters
        .sort_values(key_cols + ["registration_datetime"], na_position="last")
        .drop_duplicates(subset=key_cols, keep="first")
    )

    appointment_encounters = appointments.merge(
        encounters_by_key,
        on=key_cols,
        how="left",
        suffixes=("", "_hospac"),
    )

    dataset = appointment_encounters.merge(
        patients,
        left_on="p_num_exp_key",
        right_on="m_num_exp_key",
        how="left",
    )

    # Lead time: days between reservation creation and appointment date.
    # This is available before the appointment and can be useful for no-show prediction.
    dataset["lead_time_days"] = (
        dataset["appointment_datetime"] - dataset["reservation_datetime"]
    ).dt.total_seconds() / (60 * 60 * 24)

    dataset.loc[dataset["lead_time_days"] < 0, "lead_time_days"] = np.nan

    # Keep only appointments with known attendance labels.
    # Unknown labels are excluded instead of imputed to avoid creating artificial targets.
    dataset = dataset[dataset["no_show"].notna()].copy()
    dataset["no_show"] = dataset["no_show"].astype(int)

    # Feature selection for the initial HIS-10 MVP.
    #
    # We avoid post-event variables such as actual arrival or registration datetime,
    # because they would not be available before the appointment and could create data leakage.
    #
    # Note: p_status should be reviewed with the Data Science team. If it represents
    # a post-appointment state, it should be removed from the model features.
    #
    # CONSULTAEXTERNA was explored as an additional attendance confirmation source.
    # It is not included in the MVP because it only matched a small fraction of HOSAGD
    # appointments and recovered very few missing asistencia labels.
    feature_cols = [
        "area",
        "med",
        "esp",
        "tpo_cita",
        "conflicto",
        "agregada",
        "ultimahora",
        "buffer",
        "duration_min",
        "appointment_hour",
        "appointment_day_of_week",
        "appointment_day",
        "appointment_month",
        "appointment_is_weekend",
        "lead_time_days",
        "p_status",
        "p_sexo",
        "p_tpo_pac",
        "p_tpo_cita",
        "m_sexo",
        "m_edad_num",
        "m_ciu",
        "m_col",
        "m_cp",
        "m_edo",
        "m_pai",
        "no_show",
    ]

    feature_cols = [col for col in feature_cols if col in dataset.columns]
    dataset = dataset[feature_cols].copy()

    categorical_cols = [
        col
        for col in dataset.columns
        if col != "no_show" and (dataset[col].dtype == "object" or str(dataset[col].dtype).startswith("string"))
    ]

    # Fill only categorical missing values as UNKNOWN.
    # Numeric missing values are preserved for the modeling step.
    dataset = _fill_categorical_missing(dataset, categorical_cols)

    return dataset.reset_index(drop=True)


def create_his05_master_table(
    processed_notamedicaurg: pd.DataFrame,
    processed_triage: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create HIS-05 hourly master table for saturation / wait-time forecasting.

    Output grain: one row per hour.

    Main operational target:
    - pacientes_llegando: number of emergency arrivals per hour.

    Compatibility target:
    - tiempo_espera: average wait-time proxy per hour.

    Important: tiempo_espera is not the official waiting time. It is based on
    wait_proxy_min from NOTAMEDICAURG because AtMed_Hora is unavailable.
    """

    urg = processed_notamedicaurg.copy()
    triage = processed_triage.copy()

    urg = urg[urg["arrival_datetime"].notna()].copy()

    # Aggregate emergency arrivals at hourly level.
    # This is the modeling grain for the initial HIS-05 forecasting dataset.
    urg["timestamp"] = urg["arrival_datetime"].dt.floor("h")

    # Only valid wait-time proxy values are used when aggregating waiting-time metrics.
    # Invalid proxy values still allow arrivals to contribute to demand counts.
    valid_wait = urg["valid_wait_proxy"] == 1

    # Build hourly demand and wait-time proxy metrics from emergency notes.
    hourly_urg = (
        urg.groupby("timestamp")
        .agg(
            pacientes_llegando=("arrival_datetime", "size"),
            tiempo_espera=("wait_proxy_min", lambda s: s[valid_wait.loc[s.index]].mean()),
            tiempo_espera_mediana=("wait_proxy_min", lambda s: s[valid_wait.loc[s.index]].median()),
            wait_proxy_valid_count=("valid_wait_proxy", "sum"),
        )
        .reset_index()
    )

    # Add hourly triage counts as exogenous variables for saturation forecasting.
    if not triage.empty and "triage_datetime" in triage.columns:
        triage = triage[triage["triage_datetime"].notna()].copy()
        triage["timestamp"] = triage["triage_datetime"].dt.floor("h")

        hourly_triage = (
            triage.groupby("timestamp")
            .agg(
                triage_events=("triage_datetime", "size"),
            )
            .reset_index()
        )

        triage_pivot = (
            pd.crosstab(triage["timestamp"], triage["triage_clean"])
            .add_prefix("triage_")
            .reset_index()
        )

        hourly_triage = hourly_triage.merge(triage_pivot, on="timestamp", how="left")
    else:
        hourly_triage = pd.DataFrame({"timestamp": hourly_urg["timestamp"]})
        hourly_triage["triage_events"] = 0

    master = hourly_urg.merge(hourly_triage, on="timestamp", how="left")

    count_cols = [
        col
        for col in master.columns
        if col.startswith("triage_") or col in ["triage_events", "pacientes_llegando", "wait_proxy_valid_count"]
    ]

    for col in count_cols:
        master[col] = master[col].fillna(0)

    # Add calendar features that can help capture daily and weekly demand patterns.
    master = _add_temporal_features(master, "timestamp")

    master = master.sort_values("timestamp").reset_index(drop=True)

    return master


# =========================================================
# Backward-compatible wrappers
# =========================================================
# These functions keep compatibility with older placeholder code in the repo.
# The actual Data Engineering pipeline should use the clean_* nodes and the
# create_his10_base / create_his05_master_table functions above.

def preprocess_hospital_data(hospac: pd.DataFrame, hosmpi: pd.DataFrame) -> pd.DataFrame:
    """
    Backward-compatible wrapper from the original placeholder code.
    """
    processed_hospac = clean_hospac(hospac)
    processed_hosmpi = clean_hosmpi(hosmpi)

    return processed_hospac.merge(
        processed_hosmpi,
        left_on="p_num_exp_key",
        right_on="m_num_exp_key",
        how="left",
    )


def create_feature_table_his10(primary_df: pd.DataFrame, hosagd: pd.DataFrame) -> pd.DataFrame:
    """
    Backward-compatible placeholder wrapper.

    The primary_df argument is kept only to preserve the old function signature.
    This function does not create the full HIS-10 modeling dataset.
    Prefer create_his10_base() in the actual Data Engineering pipeline.
    """
    processed_hosagd = clean_hosagd(hosagd)
    return processed_hosagd[processed_hosagd["no_show"].notna()].copy()
