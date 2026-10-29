"""Validated local OHLCV data loading."""

from dataclasses import dataclass
from datetime import date
import csv
from pathlib import Path

REQUIRED_COLUMNS = ("date", "open", "high", "low", "close", "volume")


@dataclass(frozen=True)
class OHLCVBar:
    date: date
    open: float
    high: float
    low: float
    close: float
    volume: float


def load_ohlcv(path: str | Path) -> list[OHLCVBar]:
    """Load chronological OHLCV bars and reject malformed or duplicate dates."""
    source = Path(path)
    with source.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("OHLCV CSV must include a header row")
        columns = {column.strip().lower() for column in reader.fieldnames}
        missing = [column for column in REQUIRED_COLUMNS if column not in columns]
        if missing:
            raise ValueError(f"OHLCV CSV is missing columns: {', '.join(missing)}")

        bars: list[OHLCVBar] = []
        seen_dates: set[date] = set()
        for line_number, row in enumerate(reader, start=2):
            try:
                current_date = date.fromisoformat(row["date"].strip())
                values = {
                    key: float(row[key])
                    for key in REQUIRED_COLUMNS[1:]
                }
            except (AttributeError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid OHLCV row at line {line_number}") from exc
            if current_date in seen_dates:
                raise ValueError(f"Duplicate date at line {line_number}: {current_date}")
            if not (values["low"] <= values["open"] <= values["high"] and
                    values["low"] <= values["close"] <= values["high"]):
                raise ValueError(f"Price range is invalid at line {line_number}")
            if values["volume"] < 0:
                raise ValueError(f"Volume cannot be negative at line {line_number}")
            seen_dates.add(current_date)
            bars.append(OHLCVBar(current_date, **values))

    if bars != sorted(bars, key=lambda bar: bar.date):
        raise ValueError("OHLCV rows must be sorted chronologically")
    if not bars:
        raise ValueError("OHLCV CSV contains no data rows")
    return bars
