import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Iterable

from openai import APIError, APITimeoutError, OpenAI, RateLimitError

DEFAULT_BASE_URL = ""
DEFAULT_MODEL = "gpt-4o"
DEFAULT_TIMEOUT = 180.0

SYSTEM_PROMPT = """You are a professional academic paper analysis expert.
You must generate high-quality comparison QA pairs strictly grounded in the provided paper text.
Do not introduce external knowledge.
Return JSON only.
"""

USER_PROMPT_TEMPLATE = """Input:
- Paper Content: {paper_content}
- Paper Title: {paper_title}

Task requirements are as follows:
Identify two different methods / techniques / models / methodologies. In the provided text, find two methods, techniques, systems, models, or research paths that are explicitly described or contrasted (e.g., "Government Agencies vs. Industry and AGI Labs" or "RAP vs. MCTS"). Then generate a single, specific, and directly comparative question. The question should focus on one of the following aspects: performance differences, efficiency comparison, applicability scenarios, limitations or risks, basic principles or design philosophy, or governance roles / functional divisions. The question must be expressed in a clear interrogative sentence and should encourage the revelation of differences, trade-offs, or similarities.

Provide a concise keyword-style answer. The answer should be a structured list of 3-10 keywords, formatted as:
[A: keyword1, keyword2, ...] [B: keyword1, keyword2, ...]
Keywords should be concise and accurate, reflecting the core functions, responsibilities, mechanisms, or characteristics in the text. Avoid using complete sentences.

Provide supporting evidence paragraphs. The evidence field needs to contain all relevant sentences from the original text used to support the Q&A, retaining all citation marks for subsequent supplementation of reference content.

Example:
"evidence": "Government Agencies oversee AI policies using legislative, judicial, and enforcement powers... (Anderljung et al., 2023). Industry and AGI Labs perform risk assessments throughout the lifecycle... [79]"

Output multiple Q&A pairs. Based on the text content, generate as many independent and meaningful comparative Q&A pairs as possible, {min_pairs} or more. Each Q&A pair is separated by a comma or output as a JSON list. All questions and answers must closely follow the text content, highlight technical comparisons, and avoid generalization or vague statements.

Output format (JSON list):
[
  {{
    "question": "...",
    "answer": "[A: ...] [B: ...]",
    "evidence": "..."
  }}
]

Notes:
- All information must originate from the provided text.
- Keywords should be short and powerful, reflecting core concepts.
- Do not generate repetitive or semantically similar questions.
- The evidence must contain the original sentences from the source text and must not be rewritten.
- Return only the JSON list, without markdown fences or extra commentary.
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate comparison QA pairs from paper text."
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--paper-file",
        type=Path,
        help="Single paper text/markdown file.",
    )
    input_group.add_argument(
        "--input-jsonl",
        type=Path,
        help="JSONL file. Each line should contain paper_title and paper_content.",
    )
    input_group.add_argument(
        "--input-json",
        type=Path,
        help="JSON file containing a dict or a list of dicts.",
    )
    parser.add_argument(
        "--paper-title",
        type=str,
        default=None,
        help="Paper title for --paper-file. Defaults to the file stem.",
    )
    parser.add_argument(
        "--title-field",
        type=str,
        default="paper_title",
        help="Title field name for JSON/JSONL input.",
    )
    parser.add_argument(
        "--content-field",
        type=str,
        default="paper_content",
        help="Content field name for JSON/JSONL input.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./outputs_compare_qa"),
        help="Directory used to save generated JSON results.",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=os.getenv("ADC_BASE_URL", DEFAULT_BASE_URL),
        help="OpenAI-compatible API base URL.",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.getenv("ADC_API_KEY") or os.getenv("OPENAI_API_KEY"),
        help="API key. Prefer ADC_API_KEY or OPENAI_API_KEY environment variables.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.getenv("ADC_MODEL", DEFAULT_MODEL),
        help="Model name.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--min-pairs",
        type=int,
        default=5,
        help="Minimum requested number of QA pairs per paper.",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=80000,
        help="Maximum number of input characters kept from each paper.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum retry attempts for one paper.",
    )
    parser.add_argument(
        "--retry-wait",
        type=float,
        default=3.0,
        help="Base seconds before retry. Later retries back off automatically.",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help="Request timeout in seconds.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )
    return parser


def load_papers(args: argparse.Namespace) -> list[dict[str, str]]:
    if args.paper_file:
        paper_content = read_text(args.paper_file)
        paper_title = (args.paper_title or args.paper_file.stem).strip()
        return [{"paper_title": paper_title, "paper_content": paper_content}]

    if args.input_jsonl:
        return load_jsonl_records(
            args.input_jsonl, args.title_field, args.content_field
        )

    if args.input_json:
        return load_json_records(
            args.input_json, args.title_field, args.content_field
        )

    raise ValueError("No valid input source provided.")


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def load_jsonl_records(
    path: Path, title_field: str, content_field: str
) -> list[dict[str, str]]:
    papers: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {line_no}: {exc}") from exc
            papers.append(extract_paper_record(record, title_field, content_field, line_no))
    return papers


def load_json_records(
    path: Path, title_field: str, content_field: str
) -> list[dict[str, str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        raise ValueError("JSON input must be a dict or a list of dicts.")

    papers: list[dict[str, str]] = []
    for idx, record in enumerate(data, start=1):
        papers.append(extract_paper_record(record, title_field, content_field, idx))
    return papers


def extract_paper_record(
    record: Any, title_field: str, content_field: str, index: int
) -> dict[str, str]:
    if not isinstance(record, dict):
        raise ValueError(f"Record #{index} is not a JSON object.")

    title = str(record.get(title_field, "")).strip()
    content = str(record.get(content_field, "")).strip()
    if not title:
        raise ValueError(f"Record #{index} is missing '{title_field}'.")
    if not content:
        raise ValueError(f"Record #{index} is missing '{content_field}'.")
    return {"paper_title": title, "paper_content": content}


def sanitize_title(title: str) -> str:
    cleaned = re.sub(r'[\\/:*?"<>|]+', "_", title).strip()
    cleaned = re.sub(r"\s+", "_", cleaned)
    return cleaned[:120] or "untitled_paper"


def build_user_prompt(paper_title: str, paper_content: str, min_pairs: int) -> str:
    return USER_PROMPT_TEMPLATE.format(
        paper_title=paper_title.strip(),
        paper_content=paper_content.strip(),
        min_pairs=min_pairs,
    )


def maybe_truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip()


def create_client(args: argparse.Namespace) -> OpenAI:
    if not args.api_key:
        raise ValueError(
            "Missing API key. Use --api-key or set ADC_API_KEY / OPENAI_API_KEY."
        )
    return OpenAI(
        api_key=args.api_key,
        base_url=args.base_url,
        timeout=args.request_timeout,
    )


def request_qa_pairs(
    client: OpenAI,
    *,
    paper_title: str,
    paper_content: str,
    model: str,
    temperature: float,
    min_pairs: int,
    max_retries: int,
    retry_wait: float,
) -> list[dict[str, str]]:
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": build_user_prompt(
                            paper_title=paper_title,
                            paper_content=paper_content,
                            min_pairs=min_pairs,
                        ),
                    },
                ],
            )
            content = response.choices[0].message.content or ""
            qa_pairs = normalize_qa_pairs(parse_json_payload(content))
            if not qa_pairs:
                raise ValueError("Model returned empty QA pairs.")
            return qa_pairs
        except (APIError, APITimeoutError, RateLimitError, ValueError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt == max_retries:
                break
            wait_seconds = retry_wait * attempt
            print(
                f"[retry {attempt}/{max_retries}] {paper_title}: {exc}. "
                f"Sleeping {wait_seconds:.1f}s...",
                file=sys.stderr,
            )
            time.sleep(wait_seconds)
    assert last_error is not None
    raise RuntimeError(f"Failed to generate QA pairs for '{paper_title}': {last_error}")


def parse_json_payload(raw_text: str) -> Any:
    text = raw_text.strip()
    if not text:
        raise ValueError("Empty model response.")

    fence_match = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fence_match:
        text = fence_match.group(1).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for match in re.finditer(r"[\[{]", text):
        start = match.start()
        try:
            payload, _ = decoder.raw_decode(text[start:])
            return payload
        except json.JSONDecodeError:
            continue

    raise ValueError("Could not find a valid JSON payload in the model response.")


def normalize_qa_pairs(payload: Any) -> list[dict[str, str]]:
    data = payload
    if isinstance(data, dict):
        for key in ("qa_pairs", "data", "results", "items"):
            if isinstance(data.get(key), list):
                data = data[key]
                break

    if not isinstance(data, list):
        raise ValueError("Model output JSON must be a list or a dict containing a list.")

    normalized: list[dict[str, str]] = []
    seen_questions: set[str] = set()
    for item in data:
        if not isinstance(item, dict):
            continue
        question = str(item.get("question", "")).strip()
        answer = normalize_answer(item.get("answer", ""))
        evidence = str(item.get("evidence", "")).strip()

        if not question or not answer or not evidence:
            continue
        if question in seen_questions:
            continue

        seen_questions.add(question)
        normalized.append(
            {
                "question": question,
                "answer": answer,
                "evidence": evidence,
            }
        )
    return normalized


def normalize_answer(answer: Any) -> str:
    if isinstance(answer, str):
        return answer.strip()
    if isinstance(answer, list):
        return " ".join(str(part).strip() for part in answer if str(part).strip())
    if isinstance(answer, dict):
        pieces: list[str] = []
        for key in ("A", "B", "a", "b"):
            value = answer.get(key)
            if value is None:
                continue
            if isinstance(value, list):
                joined = ", ".join(str(x).strip() for x in value if str(x).strip())
            else:
                joined = str(value).strip()
            if joined:
                pieces.append(f"[{key.upper()}: {joined}]")
        return " ".join(pieces).strip()
    return str(answer).strip()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        papers = load_papers(args)
        client = create_client(args)
    except Exception as exc:
        print(f"Initialization error: {exc}", file=sys.stderr)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)

    index_rows: list[dict[str, Any]] = []
    total = len(papers)
    for idx, paper in enumerate(papers, start=1):
        title = paper["paper_title"]
        raw_content = paper["paper_content"]
        content = maybe_truncate(raw_content, args.max_chars)
        output_path = args.output_dir / f"{sanitize_title(title)}.json"

        if output_path.exists() and not args.overwrite:
            print(f"[{idx}/{total}] Skip existing file: {output_path}")
            index_rows.append(
                {
                    "paper_title": title,
                    "output_file": str(output_path),
                    "status": "skipped_existing",
                }
            )
            continue

        print(f"[{idx}/{total}] Generating QA pairs for: {title}")
        if len(raw_content) > len(content):
            print(
                f"[{idx}/{total}] Truncated input from {len(raw_content)} to {len(content)} chars."
            )
        try:
            qa_pairs = request_qa_pairs(
                client,
                paper_title=title,
                paper_content=content,
                model=args.model,
                temperature=args.temperature,
                min_pairs=args.min_pairs,
                max_retries=args.max_retries,
                retry_wait=args.retry_wait,
            )
            write_json(output_path, qa_pairs)
            if len(qa_pairs) < args.min_pairs:
                print(
                    f"Warning: only generated {len(qa_pairs)} QA pairs for {title}.",
                    file=sys.stderr,
                )
            index_rows.append(
                {
                    "paper_title": title,
                    "output_file": str(output_path),
                    "status": "ok",
                    "qa_pair_count": len(qa_pairs),
                }
            )
            print(f"Saved {len(qa_pairs)} QA pairs -> {output_path}")
        except Exception as exc:
            error_path = args.output_dir / f"{sanitize_title(title)}.error.json"
            write_json(
                error_path,
                {
                    "paper_title": title,
                    "error": str(exc),
                },
            )
            index_rows.append(
                {
                    "paper_title": title,
                    "output_file": str(error_path),
                    "status": "error",
                    "error": str(exc),
                }
            )
            print(f"Failed: {title}: {exc}", file=sys.stderr)

    index_path = args.output_dir / "index.jsonl"
    write_jsonl(index_path, index_rows)
    print(f"Run finished. Index written to {index_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
