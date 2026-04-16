import re
import time
import json
import requests
import json
import argparse
import sys
from pathlib import Path

BASE_URL = "https://api.ai-gaochao.cn/v1"
API_KEY = "sk-SHiLrPzmuREecae9E29f4eA62fD84eA5A3E569026fAf33De"
MODEL = "gpt-5.2"

def eval(ground_truth, answer, common_errors) -> dict:

    sys_prompt = '''
You are an expert evaluator. Your task is to analyze an answer's topic coverage against a ground truth list and return **only** a detailed evaluation as a single, valid JSON object.

---

## Evaluation Logic

### Phase 1: Coverage Analysis (Recall)
- For each topic in the **Scoring Topics List** (ground truth), determine if it is semantically present in the **Answer to Evaluate**.
- Classify topics into:
  - **`covered_topics`**: Topics from ground truth that appear in the answer (as simple string list).
  - **`missed_from_ground_truth`**: Topics from ground truth that are absent (as simple string list).

### Phase 2: Spurious Analysis (Precision)
- Identify content in the answer that falls into these categories:

  **A. Harmful Spurious (penalize precision)**:
  - Statements claiming data is missing/insufficient
  - Apologies or refusal to answer
  - Completely off-topic explanations
  - Factually incorrect information

  **B. Supplementary Content (report but don't penalize)**:
  - Additional technical details related to ground truth topics
  - Examples or use cases that enhance understanding
  - Brief contextual background

### Phase 3: Common Errors Detection (Hallucination Analysis)
- **If a Common Errors List is provided**, check if the answer contains any of these known error patterns.
- For each error in the Common Errors List, determine if it appears in:
  - The `spurious_harmful` content identified in Phase 2
  - OR anywhere else in the answer text
- Classify detected errors into:
  - **`detected_common_errors`**: Errors from the Common Errors List that appear in the answer (as simple string list).
  - Include brief evidence for each detected error in `reason.error_details`

### Phase 4: Reasoning Documentation
- Provide a **comprehensive `reason` object** containing:
  - **`overall`**: A 2-3 sentence summary explaining:
    - How many topics were covered and why
    - Overall quality of the answer's alignment with ground truth
  - **`coverage_details`**: For each covered topic, explain:
    - Which part of the answer supports this classification
    - Key evidence or quotes (max 80 chars per quote)
  - **`miss_details`**: For each missed topic, explain:
    - Why it was not found (e.g., "Not mentioned", "Only tangentially referenced")
  - **`spurious_details`**: For each spurious element, explain:
    - Why it's considered off-topic or irrelevant
  - **`error_details`**: (Only if Common Errors List is provided) For each detected error, explain:
    - Where it appears in the answer
    - Why it matches the error pattern (max 80 chars evidence)

---

## Output Format

Return **only** this JSON structure (no markdown code blocks, no explanations):

{
  "covered_topics": ["topic1", "topic2"],
  "missed_from_ground_truth": ["topic3"],
  "spurious_harmful": ["Apology statement"],
  "spurious_supplementary": ["Additional technical detail"],
  "detected_common_errors": ["Error pattern 1", "Error pattern 2"],
  "reason": {
    "overall": "The answer successfully covers 2 out of 3 ground truth topics by...",
    "coverage_details": {
      "topic1": "Explicitly mentioned in paragraph 2: 'The algorithm uses...'",
      "topic2": "Demonstrated through code example in section 3"
    },
    "miss_details": {
      "topic3": "No discussion of space complexity optimization found"
    },
    "spurious_details": {
      "Apology statement": "Unnecessary disclaimer not requested in ground truth"
    },
    "error_details": {
      "Error pattern 1": "Found in sentence: 'The algorithm always runs in O(1)...'",
      "Error pattern 2": "Claim contradicts established theory"
    }
  }
}

---

## Requirements

- **Output**: Pure JSON only (no ```json wrapper)
- **String values**: Use concise, descriptive labels for topics
- **Empty arrays/objects**: Use `[]` or `{}` if a category has no items
- **Semantic matching**: Match topics by meaning, not exact wording
- **Evidence quotes**: Must be verbatim, max 80 characters
- **Overall reason**: Must be 2-3 sentences, focus on coverage rate and answer quality
'''

    # 格式化 ground_truth 为列表形式
    gt_formatted = "\n".join([f"- {topic}" for topic in ground_truth])

    prompt = f'''
Evaluate the following answer based on the provided topics list and return the result strictly in the JSON format defined in your instructions.

---

### Answer to Evaluate:
{answer}

### Scoring Topics List:
{gt_formatted}

### Common Errors List:
{common_errors if common_errors else "N/A"}
'''

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.3,
        "max_tokens": 2048,
        "response_format": { "type": "json_object" } 
    }
    
    max_retries = 3
    retries = 0
    success = False
    result = None
    res_dict = {}
    while not success and retries < max_retries:
        try:
            response = requests.post(
                f"{BASE_URL}/chat/completions",
                headers=headers,
                json=payload,
                timeout=120
            )
            print(f"Debug: Raw JSON response:\n{response.json()}\n")
            result = response.json()['choices'][0]['message']['content'].strip()
        except Exception as e:
            retries += 1
            print(f"OpenAI error, retrying... ({retries}/{max_retries})")
            time.sleep(2)
            
        try:
            
            # 清理多余换行符
            result = re.sub(r"\n+", "\n", result)
            
            # 解析 JSON
            json_res = json.loads(result)
            
            # 提取核心字段
            covered = json_res.get("covered_topics", [])
            missed = json_res.get("missed_from_ground_truth", [])
            spurious_harmful = json_res.get("spurious_harmful", [])
            spurious_supplementary = json_res.get("spurious_supplementary", [])
            detected_errors = json_res.get("detected_common_errors", [])
            reason_obj = json_res.get("reason", {})
            
            # 计算指标
            gt_total = len(ground_truth)
            covered_count = len(covered)
            spurious_count = len(spurious_harmful)
            
            recall = covered_count / gt_total if gt_total > 0 else 0.0
            precision = covered_count / (covered_count + spurious_count) if (covered_count + spurious_count) > 0 else 0.0
            f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

            hallucination_score = 0.0
            if common_errors:
                detected_count = len(detected_errors)
                total_errors = len(common_errors)
                hallucination_score = detected_count / total_errors if total_errors > 0 else 0.0

            res_dict = {
                "Recall": round(recall, 4),
                "Precision": round(precision, 4),
                "F1_Score": round(f1_score, 4),
                "Hallucination_Score": round(hallucination_score, 4),
                "Covered_Topics": covered,
                "Missed_Topics": missed,
                "Spurious_Harmful": spurious_harmful,
                "Spurious_Supplementary": spurious_supplementary,
                "Detected_Common_Errors": detected_errors,
                "Reason": reason_obj,
                "Raw_Response": json_res
            }
            
            return res_dict
        
        except Exception as e:
            print(f"Error parsing JSON response from OpenAI, Error: {e}.")
            retries += 1
            continue
            
    return {
        "Recall": "N/A",
        "Precision": "N/A",
        "F1_Score": "N/A",
        "Hallucination_Score": "N/A",
        "Covered_Topics": [],
        "Missed_Topics": ground_truth.copy(),
        "Spurious_Harmful": [],
        "Spurious_Supplementary": [],
        "Detected_Common_Errors": [],
        "Reason": {
            "overall": "Evaluation failed due to API or parsing errors",
            "coverage_details": {},
            "miss_details": {topic: "Evaluation not completed" for topic in ground_truth},
            "spurious_details": {}
        },
        "Raw_Response": None,
        "Error": "All retry attempts failed"
    }



def parse_arguments():
    parser = argparse.ArgumentParser(
        description='evaluate QA predictions against ground truth answers.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--qa_path',
        type=str,
        required=True,
        help='JSON file path for predicted answers (including question and answer fields)'
    )
    
    parser.add_argument(
        '--base_qa_path',
        type=str,
        required=True,
        help='JSON file path for ground truth answers (including question and answer fields)'
    )
    
    parser.add_argument(
        '--save_path',
        type=str,
        required=True,
        help='Evaluation result save path'
    )
    
    return parser.parse_args()

def parse_ground_truth(gt_string):
    gt_string = gt_string.strip()
    if gt_string.startswith('[') and gt_string.endswith(']'):
        gt_string = gt_string[1:-1]
    
    topics = []
    current_topic = ""
    bracket_depth = 0
    
    for char in gt_string:
        if char == ',' and bracket_depth == 0:
            topics.append(current_topic.strip())
            current_topic = ""
        else:
            if char in '([{':
                bracket_depth += 1
            elif char in ')]}':
                bracket_depth -= 1
            current_topic += char
    
    if current_topic.strip():
        topics.append(current_topic.strip())
    
    return topics

def load_json(file_path):
    # try:
    #     with open(file_path, 'r', encoding='utf-8') as f:
    #         data = json.load(f)
    # except FileNotFoundError:
    #     print(f"Error: File not found: '{file_path}'")
    #     exit(1) 
    # except json.JSONDecodeError:
    #     print(f"Error: '{file_path}' is not a valid json file.")
    #     exit(1)
    
    data = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            
            # 判断是否为 JSONL：包含换行符且每行都像 JSON
            if '\n' in content:
                lines = content.split('\n')
                # 检查前几行是否都是有效 JSON
                if all(line.strip().startswith(('{', '[')) for line in lines[:3] if line.strip()):
                    print("[Info] Detected JSONL format")
                    for i, line in enumerate(lines, 1):
                        if not line.strip():
                            continue
                        try:
                            data.append(json.loads(line))
                        except json.JSONDecodeError:
                            print(f"[Warning] Line {i} is invalid JSON, skipping")
                    if not data:
                        raise ValueError("No valid JSON lines found")
                else:
                    # 标准 JSON
                    data = json.loads(content)
            else:
                # 单行，标准 JSON
                data = json.loads(content)
                
    except FileNotFoundError:
        print(f"Error: File not found: '{file_path}'")
        exit(1)
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Error: '{file_path}' is not valid: {e}")
        exit(1)
        
    if isinstance(data, dict):
        for key in ['data', 'results', 'list', 'items', 'qa_list']:
            if key in data and isinstance(data[key], list):
                print(f"[Info] Detected container key '{key}', using its content as the list.")
                data = data[key]
                break
        else:
            if "question" in data and "output" in data:
                 print("[Info] Input looks like a raw result item with 'output', wrapping into list.")
                 data = [data]
            elif "question" in data and "answer" in data:
                print("[Info] Input is a single dict object, wrapping into list.")
                data = [data]
            elif "answer" in data and "question_index" in data:
                 print("[Info] Input looks like a raw result item, wrapping into list.")
                 data = [data]
            else:
                print(f"[Warning] JSON is a dict but could not auto-detect list format. Keys: {list(data.keys())}")
                
    if not isinstance(data, list):
        print(f"[Error] Expected a list of QA pairs, but got {type(data)}. Trying to force wrap...")
        data = [data]
        
    return data

def align_qa_data(qa_list, base_qa_list):
    base_qa_dict = {}
    
    for item in base_qa_list:
        question = item.get("question")
        answer = item.get("answer")
        
        if not question or not answer:
            continue
        
        if isinstance(answer, str):
            if answer.strip().startswith('[') and answer.strip().endswith(']'):
                print(f"ℹ️  Info: Converting string-formatted list to actual list for question '{question[:50]}...'")
                answer = parse_ground_truth(answer)
            else:
                answer = [answer]
        
        common_errors = item.get("com_err", [])
        
        if isinstance(common_errors, str):
            if common_errors.strip().startswith('[') and common_errors.strip().endswith(']'):
                print(f"ℹ️  Info: Converting string-formatted list to actual list for question '{question[:50]}...'")
                common_errors = parse_ground_truth(common_errors)
            else:
                common_errors = [common_errors]
        
        normalized_key = question.strip()
        base_qa_dict[normalized_key] = {
            "answer": answer,
            "common_errors": common_errors
        }
    
    aligned_data = []

    for i, item in enumerate(qa_list):
       
        question = item.get("question")
        predicted_answer = item.get("output") or item.get("answer", "")
        
        if not question:
            print(f"⚠ Warning: The {i}th data is missing the question field and has been skipped.")
            continue
        
        base_item = base_qa_dict.get(question)
        
        if base_item is None:
            print(f"⚠ Warning: The question '{question[:50]}...' was not found in the standard answers and has been skipped.")
            continue
        
        aligned_data.append({
            'question': question,
            'predicted_answer': predicted_answer,
            'ground_truth': base_item["answer"],
            'common_errors': base_item["common_errors"]
        })
    
    return aligned_data

if __name__ == "__main__":

    args = parse_arguments()

    qa_path = args.qa_path
    base_qa_path = args.base_qa_path
    save_path = args.save_path
    
    qa_list = load_json(qa_path)
    base_qa_list = load_json(base_qa_path)
    
    print(qa_list)
    
    aligned_data = align_qa_data(qa_list, base_qa_list)
    
    print(f"[DEBUG] aligned_data length: {len(aligned_data)}")  # ← 关键！
    print(f"[DEBUG] aligned_data sample: {aligned_data[:1] if aligned_data else 'EMPTY'}")

    results_for_json = []

    for i, item in enumerate(aligned_data):
        ground_truth=item['ground_truth']
        answer=item['predicted_answer']
        question=item['question']
        common_errors=item['common_errors']

        if ground_truth is None:
            print(f"Warning: No ground truth found, Skipping {i}th question.")
            continue
        
        res_dict = eval(
            ground_truth=ground_truth,
            answer=answer, 
            common_errors=common_errors
        )

        results_for_json.append({
            'question': question,
            'ground_truth': ground_truth,
            'answer': answer,
            'recall': res_dict.get("Recall", "N/A"),
            'precision': res_dict.get("Precision", "N/A"),
            'f1_score': res_dict.get("F1_Score", "N/A"),
            'hallucination_score': res_dict.get("Hallucination_Score", "N/A"),
            'covered_topics': res_dict.get("Covered_Topics", []),
            'missed_topics': res_dict.get("Missed_Topics", []),
            'spurious_harmful': res_dict.get("Spurious_Harmful", []),
            'spurious_supplementary': res_dict.get("Spurious_Supplementary", []),
            'detected_common_errors': res_dict.get("Detected_Common_Errors", []),
            'reason': res_dict.get("Reason", {})
        })

    if results_for_json:
        try:
            with open(save_path, 'w', encoding='utf-8') as jsonfile:
                json.dump(results_for_json, jsonfile, indent=4, ensure_ascii=False)
            
            print(f"\nSuccessfully saved results to {save_path}")

        except IOError as e:
            print(f"\nError writing to JSON file: {e}")


    