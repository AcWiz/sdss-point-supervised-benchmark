from __future__ import annotations

from collections.abc import Mapping

SDSS_FLAG_BITS = {
    "BRIGHT": 1,
    "EDGE": 2,
    "BLENDED": 3,
    "CHILD": 5,
    "CR": 12,
    "INTERP": 17,
    "SATURATED": 18,
    "NOTCHECKED": 19,
    "SUBTRACTED": 20,
    "BADSKY": 22,
    "DEBLEND_NOPEAK": 24,
    "PSF_FLUX_INTERP": 25,
    "INTERP_CENTER": 28,
    "LOCAL_EDGE": 39,
    "PEAKS_TOO_CLOSE": 43,
    "BINNED1": 44,
    "BINNED2": 45,
    "BINNED4": 46,
    "MOVED": 47,
    "DEBLENDED_AS_PSF": 48,
}

FLAG_NAME_TO_BIT = {name: 1 << bit for name, bit in SDSS_FLAG_BITS.items()}

LABEL_QUALITY_TO_INT = {"suspect": 0, "weak": 1, "clean": 2}
QUALITY_TO_WEIGHT = {"suspect": 0.0, "weak": 0.5, "clean": 1.0}

HARD_SUSPECT_FLAGS = {
    "SATURATED",
    "EDGE",
    "LOCAL_EDGE",
    "BADSKY",
    "NOTCHECKED",
    "PEAKS_TOO_CLOSE",
    "DEBLEND_NOPEAK",
}
SOFT_WEAK_FLAGS = {
    "INTERP",
    "PSF_FLUX_INTERP",
    "BLENDED",
    "CHILD",
    "DEBLENDED_AS_PSF",
    "SUBTRACTED",
}


def flag_names(raw_flags: int | str | None) -> list[str]:
    """Return known SDSS PhotoObj flag names present in a numeric or name string."""

    if raw_flags is None or raw_flags == "":
        return []
    if isinstance(raw_flags, str) and not raw_flags.strip().lstrip("-").isdigit():
        tokens = raw_flags.replace(",", " ").replace("|", " ").replace(";", " ").split()
        return [token.upper() for token in tokens if token.upper() in FLAG_NAME_TO_BIT]
    mask = _flag_mask(raw_flags)
    return [name for name, bit_mask in FLAG_NAME_TO_BIT.items() if mask & bit_mask]


def source_quality_flags(metadata: Mapping[str, object] | None) -> list[str]:
    if metadata is None:
        return []
    return flag_names(metadata.get("flags"))


def source_label_quality(metadata: Mapping[str, object] | None) -> str:
    flags = set(source_quality_flags(metadata))
    if flags & HARD_SUSPECT_FLAGS:
        return "suspect"
    if {"INTERP_CENTER", "CR"} <= flags:
        return "suspect"
    if _clean_value(metadata):
        return "clean"
    return "weak"


def source_label_weight(metadata: Mapping[str, object] | None) -> float:
    return QUALITY_TO_WEIGHT[source_label_quality(metadata)]


def raw_flag_value(metadata: Mapping[str, object] | None) -> int:
    if metadata is None:
        return 0
    return _flag_mask(metadata.get("flags"))


def _flag_mask(raw_flags: int | str | object | None) -> int:
    if raw_flags is None or raw_flags == "":
        return 0
    try:
        return int(float(str(raw_flags).strip()))
    except ValueError:
        mask = 0
        for name in flag_names(str(raw_flags)):
            mask |= FLAG_NAME_TO_BIT[name]
        return mask


def _clean_value(metadata: Mapping[str, object] | None) -> bool:
    if metadata is None:
        return False
    value = metadata.get("clean")
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true"}
