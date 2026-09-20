"""PTB clinical-summary parsing and the paper's twelve classes."""

from __future__ import annotations

import re

CLASS_ORDER = [
    "AMI", "ALMI", "ASMI", "ASLMI", "IMI", "ILMI",
    "IPMI", "IPLMI", "LMI", "PMI", "PLMI", "HC",
]

CLASS_NAMES = {
    "AMI": "Anterior MI",
    "ALMI": "Anterior lateral MI",
    "ASMI": "Anterior septal MI",
    "ASLMI": "Anterior septal lateral MI",
    "IMI": "Inferior MI",
    "ILMI": "Inferior lateral MI",
    "IPMI": "Inferior posterior MI",
    "IPLMI": "Inferior posterior lateral MI",
    "LMI": "Lateral MI",
    "PMI": "Posterior MI",
    "PLMI": "Posterior lateral MI",
    "HC": "Healthy control",
}


def clinical_fields(comments: list[str] | str) -> dict[str, str]:
    lines = comments.splitlines() if isinstance(comments, str) else comments
    fields: dict[str, str] = {}
    for raw in lines:
        line = raw.lstrip("# ").strip()
        if ":" in line:
            key, value = line.split(":", 1)
            fields[key.strip().lower()] = value.strip()
    return fields


def _normalize_location(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"\bseptum\b|\bsepto\b|\bsept\b", "septal", value)
    value = value.replace("antero", "anterior").replace("postero", "posterior")
    value = value.replace("infero", "inferior")
    value = re.sub(r"\bposter\b", "posterior", value)
    value = re.sub(r"\blatera\b", "lateral", value)
    return re.sub(r"[^a-z]+", " ", value).strip()


def location_to_class(value: str) -> str | None:
    """Map spelling variants in PTB headers to the paper's class abbreviations."""
    text = _normalize_location(value)
    if not text or text in {"no", "n a", "unknown", "none"}:
        return None
    tokens = set(text.split())
    anterior = "anterior" in tokens
    inferior = "inferior" in tokens
    posterior = "posterior" in tokens
    lateral = "lateral" in tokens
    septal = "septal" in tokens
    if anterior and septal and lateral:
        return "ASLMI"
    if inferior and posterior and lateral:
        return "IPLMI"
    if anterior and lateral:
        return "ALMI"
    if anterior and septal:
        return "ASMI"
    if inferior and lateral:
        return "ILMI"
    if inferior and posterior:
        return "IPMI"
    if posterior and lateral:
        return "PLMI"
    if anterior:
        return "AMI"
    if inferior:
        return "IMI"
    if posterior:
        return "PMI"
    if lateral:
        return "LMI"
    return None


def label_from_comments(comments: list[str] | str) -> str | None:
    fields = clinical_fields(comments)
    reason = fields.get("reason for admission", "").lower()
    diagnosis = fields.get("diagnose", "").lower()
    if "healthy control" in reason or "healthy control" in diagnosis:
        return "HC"
    if "myocardial infarction" not in reason and "myocardial infarction" not in diagnosis:
        return None
    # The paper's 126 MI subjects match the subset with an acute localization.
    # Falling back to former MI silently expands the cohort to 148 MI subjects.
    return location_to_class(fields.get("acute infarction (localization)", ""))
