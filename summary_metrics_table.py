"""Create a metric table from a comparison summary CSV.

The comparison scripts write wide summary files with one row of metrics.
This helper converts any such summary file into a readable long-form table
with one metric per row.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd


def resolve_summary_path(path: Path) -> Path:
    """Resolve either a summary file or a directory containing one."""

    if path.is_dir():
        candidate = path / "summary_metrics.csv"
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"No summary_metrics.csv found in directory: {path}")

    if not path.exists():
        raise FileNotFoundError(f"Summary file does not exist: {path}")

    return path


def load_summary_table(path: Path) -> pd.DataFrame:
    """Load a wide summary file and convert it into a long table."""

    summary_df = pd.read_csv(path)
    if summary_df.empty:
        return pd.DataFrame(columns=["metric", "value"])

    if len(summary_df) == 1:
        row = summary_df.iloc[0]
        return pd.DataFrame(
            {"metric": summary_df.columns, "value": [row[column] for column in summary_df.columns]}
        )

    records: list[dict[str, object]] = []
    for row_index, (_, row) in enumerate(summary_df.iterrows(), start=1):
        for column in summary_df.columns:
            records.append({"row": row_index, "metric": column, "value": row[column]})

    return pd.DataFrame(records, columns=["row", "metric", "value"])


def _column_widths(headers: list[str], rows: Iterable[Iterable[object]]) -> list[int]:
    widths = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(str(cell)))
    return widths


def format_ascii_table(df: pd.DataFrame) -> str:
    """Render a simple ASCII table that works without extra dependencies."""

    headers = list(df.columns)
    rows = [list(map(_stringify_cell, row)) for row in df.itertuples(index=False, name=None)]
    widths = _column_widths(headers, rows)

    def render_row(values: list[str]) -> str:
        return " | ".join(value.ljust(widths[index]) for index, value in enumerate(values))

    separator = "-+-".join("-" * width for width in widths)
    lines = [render_row(headers), separator]
    lines.extend(render_row(row) for row in rows)
    return "\n".join(lines)


def format_markdown_table(df: pd.DataFrame) -> str:
    """Render a markdown table for easy pasting into reports."""

    headers = list(df.columns)
    rows = [list(map(_stringify_cell, row)) for row in df.itertuples(index=False, name=None)]
    widths = _column_widths(headers, rows)

    def render_row(values: list[str]) -> str:
        return "| " + " | ".join(value.ljust(widths[index]) for index, value in enumerate(values)) + " |"

    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    lines = [render_row(headers), separator]
    lines.extend(render_row(row) for row in rows)
    return "\n".join(lines)


def _stringify_cell(value: object) -> str:
    if pd.isna(value):
        return "NaN"
    if isinstance(value, float):
        return f"{value:.15g}"
    return str(value)


def write_output(df: pd.DataFrame, output_path: Path) -> None:
    suffix = output_path.suffix.lower()

    if suffix == ".csv":
        df.to_csv(output_path, index=False)
        return

    if suffix == ".md":
        output_path.write_text(format_markdown_table(df) + "\n", encoding="utf-8")
        return

    output_path.write_text(format_ascii_table(df) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a comparison summary CSV into a readable metric table."
    )
    parser.add_argument(
        "summary_file",
        type=Path,
        help="Path to a summary_metrics.csv file or a directory containing one",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Optional output path (.csv, .md, or .txt). If omitted, prints to stdout.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_path = resolve_summary_path(args.summary_file)
    table = load_summary_table(summary_path)

    if args.output is not None:
        write_output(table, args.output)
        print(f"Wrote metric table to {args.output}")
        return

    print(f"Source: {summary_path}")
    print(format_ascii_table(table))


if __name__ == "__main__":
    main()