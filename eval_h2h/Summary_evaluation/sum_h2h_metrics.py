import argparse
import json
from pathlib import Path


DEFAULT_FIELDS = [
    "total_tokens",
    "prompt_tokens",
    "sample_time_seconds_sum",
    "api_time_seconds_sum",
]


def collect_summary_files(input_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in input_dir.rglob("*_summary.json")
        if path.is_file()
    )


def load_metrics(summary_path: Path) -> dict:
    with summary_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    aggregate_metrics = payload.get("aggregate_metrics")
    if not isinstance(aggregate_metrics, dict):
        raise ValueError(f"Missing aggregate_metrics in {summary_path}")

    return aggregate_metrics


def sum_fields(summary_files: list[Path], fields: list[str]) -> tuple[dict, list[str]]:
    totals = {field: 0 for field in fields}
    skipped_files: list[str] = []

    for summary_path in summary_files:
        try:
            metrics = load_metrics(summary_path)
        except Exception:
            skipped_files.append(str(summary_path))
            continue

        for field in fields:
            value = metrics.get(field, 0)
            if isinstance(value, (int, float)):
                totals[field] += value
            else:
                skipped_files.append(str(summary_path))
                break

    return totals, skipped_files


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sum selected aggregate metrics from head-to-head evaluation summary files."
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing *_summary.json files, such as eval_results_h2h_eval200/HippoRAG_vs_LGraphRAG",
    )
    parser.add_argument(
        "--fields",
        nargs="+",
        default=DEFAULT_FIELDS,
        help="Metric fields to sum. Defaults to total_tokens prompt_tokens sample_time_seconds_sum api_time_seconds_sum",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output.",
    )
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    if not input_dir.is_dir():
        raise SystemExit(f"Input directory does not exist: {input_dir}")

    summary_files = collect_summary_files(input_dir)
    if not summary_files:
        raise SystemExit(f"No *_summary.json files found under: {input_dir}")

    totals, skipped_files = sum_fields(summary_files, args.fields)

    result = {
        "input_dir": str(input_dir),
        "summary_file_count": len(summary_files),
        "fields": args.fields,
        "totals": totals,
        "skipped_file_count": len(skipped_files),
        "skipped_files": skipped_files,
    }

    if args.pretty:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
