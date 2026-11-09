"""Schema validation for incoming data frames."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from pandas.api import types as pdt

from driftwatch.config import DataSchema


class DataValidationError(ValueError):
    """The data does not satisfy the declared schema."""


@dataclass
class ValidationReport:
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def validate_frame(frame: pd.DataFrame, schema: DataSchema) -> ValidationReport:
    """Collect every problem instead of stopping at the first."""
    report = ValidationReport()
    for name, spec in schema.columns.items():
        if name not in frame.columns:
            report.errors.append(f"missing column {name!r}")
            continue
        column = frame[name]
        nulls = int(column.isna().sum())
        if nulls and not spec.nullable:
            report.errors.append(f"column {name!r} has {nulls} null value(s)")
        present = column.dropna()
        if spec.dtype == "category":
            continue
        if not pdt.is_numeric_dtype(column):
            report.errors.append(f"column {name!r} must be numeric")
            continue
        if spec.dtype == "int" and not (present == present.round()).all():
            report.errors.append(f"column {name!r} must contain whole numbers")
        if spec.min is not None and (present < spec.min).any():
            report.errors.append(f"column {name!r} has values below {spec.min}")
        if spec.max is not None and (present > spec.max).any():
            report.errors.append(f"column {name!r} has values above {spec.max}")
    return report


def require_valid(frame: pd.DataFrame, schema: DataSchema) -> None:
    report = validate_frame(frame, schema)
    if not report.ok:
        raise DataValidationError("; ".join(report.errors))


def load_csv(path: Path | str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise DataValidationError(f"cannot read {path}: {exc}") from exc
