import json
import glob
import csv
from pathlib import Path
import re
from typing import List, Dict, Tuple, Set

# 定义一个结构来承载提取结果：(scores, skip_reason)
ExtractionResult = Tuple[Dict[str, float], str]

def get_expected_paths(reference_base: str, category: str, dataset_case_limits: dict = None) -> Set[str]:
    """
    从参考路径中获取所有应该存在的子目录结构
    返回格式: {relative_path1, relative_path2, ...}
    例如: {'1_data/simple_QA/S_00', '1_data/simple_QA/S_01', ...}
    """
    
    if dataset_case_limits is None:
        dataset_case_limits = {
            '1_data': (0, 599),  # 1_data 只读取 S_00 到 S_599
            # 其他数据集不在字典中，默认读取全部
        }
    
    # 路径模式：reference_base/dataset/category/case_id
    pattern = f"{reference_base}/*/{category}/*"
    
    all_dirs = glob.glob(pattern)
    expected_paths = set()
    reference_path = Path(reference_base)
    
    print(f"\n{'='*60}")
    print(f"Searching for expected paths in: {reference_base}")
    print(f"Category: {category}")
    print(f"Pattern: {pattern}")
    print(f"{'='*60}")
    for dataset, limits in dataset_case_limits.items():
        if limits:
            print(f"  - {dataset}: S_{limits[0]:03d} to S_{limits[1]:03d}")
        else:
            print(f"  - {dataset}: No limit")
    print(f"  - Other datasets: No limit (read all)")
    print(f"{'='*60}")
    
    # 检查 reference_base 是否存在
    if not reference_path.exists():
        print(f"ERROR: Reference base path does not exist: {reference_base}")
        return expected_paths
    
    for dir_path in all_dirs:
        if Path(dir_path).is_dir():
            try:
                # 提取相对路径（例如：1_data/simple_QA/S_00）
                rel_path = Path(dir_path).relative_to(reference_path)
                
                dataset_name = rel_path.parts[0]  # 例如：'1_data', '2_data'
                
                # 提取 Case ID（路径的最后一部分）
                case_id = rel_path.parts[-1]
                if dataset_name in dataset_case_limits and dataset_case_limits[dataset_name] is not None:
                    min_case, max_case = dataset_case_limits[dataset_name]
                    
                    # 验证 Case ID 格式
                    if not re.match(r'^S_\d{2,3}$', case_id):
                        continue
                    
                    # 提取数字并检查范围
                    try:
                        case_num = int(case_id.split('_')[1])
                    except (ValueError, IndexError):
                        continue
                    
                    # 应用范围过滤
                    if min_case <= case_num <= max_case:
                        expected_paths.add(str(rel_path))
                    else:
                        continue
                else:
                    expected_paths.add(str(rel_path))
            except ValueError as e:
                print(f"Warning: Could not get relative path for {dir_path}: {e}")
                continue
    
    print(f"\nFound {len(all_dirs)} directories")
    print(f"Extracted {len(expected_paths)} unique expected paths")
    
    if expected_paths:
        print(f"\nSample expected paths (first 5):")
        for sample in list(expected_paths)[:5]:
            print(f"  - {sample}")
    else:
        print("\n" + "!"*60)
        print("WARNING: No expected paths found!")
        print(f"Please verify that directories exist in: {reference_base}")
        print("!"*60)
    
    print(f"{'='*60}\n")
    
    return expected_paths

def check_path_existence(eval_base: str, expected_rel_path: str) -> Tuple[bool, str, List[str]]:
    """
    检查评估结果路径是否存在，以及是否包含JSON文件
    
    参数:
        eval_base: 评估结果基础路径 (例如: /mnt/data/xinyang/.../hipporag)
        expected_rel_path: 相对路径 (例如: 1_data/simple_QA/S_00)
    
    返回: (路径有效性, 完整路径, JSON文件列表)
    """
    # 构建完整路径：eval_base + expected_rel_path
    full_path = Path(eval_base) / expected_rel_path
    
    # 检查路径是否存在
    if not full_path.exists():
        return False, str(full_path), []
    
    # 检查是否包含JSON文件
    json_files = list(full_path.glob("hirag_*.json"))
    
    if not json_files:
        return False, str(full_path), []
    
    return True, str(full_path), [str(f) for f in json_files]

def extract_scores_from_file(file_path: str) -> Tuple[List[ExtractionResult], List[ExtractionResult]]:
    """
    从单个JSON文件中提取分数或跳过记录。
    返回 (有效分数列表, 跳过记录列表)
    """
    valid_scores: List[ExtractionResult] = []
    skipped_records: List[ExtractionResult] = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # 统一处理结构
        data_to_process = data if isinstance(data, list) else [data]
        
        for item in data_to_process:
            audit_data = {
                'source_file': file_path
            }
            
            # --- 1. 检查 answer 字段 ---
            answer = item.get('answer')
            if answer is None or (isinstance(answer, str) and not answer.strip()) or answer == "":
                skipped_records.append(
                    (audit_data, f"Answer is missing or empty. Value: {answer}")
                )
                continue
            
            if answer == "Insufficient information.":
                skipped_records.append(
                    (audit_data, "Insufficient information")
                )
                continue

            # --- 2. 检查 "N/A" 分数值 ---
            required_metrics = ['recall', 'precision', 'f1_score', 'hallucination_score']
            if any(item.get(key) == "N/A" for key in required_metrics):
                skipped_records.append(
                    (audit_data, "One or more scores are 'N/A'")
                )
                continue
            
            # --- 3. 检查 Reason 字段 ---
            reason_overall = item.get('Reason', {}).get('overall')
            if reason_overall == "Evaluation failed due to API or parsing errors":
                skipped_records.append(
                    (audit_data, "Evaluation failed due to API or parsing errors")
                )
                continue
            
            # --- 4. 验证所有必需分数指标存在 (有效记录) ---
            if all(key in item for key in required_metrics):
                score_data = {
                    'recall': item['recall'],
                    'precision': item['precision'],
                    'f1_score': item['f1_score'],
                    'hallucination_score': item['hallucination_score']
                }
                valid_scores.append((score_data, ""))
                
            else:
                skipped_records.append(
                    (audit_data, "Missing one or more required score metrics")
                )

        return valid_scores, skipped_records
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        skipped_records.append(
            ({'source_file': file_path}, f"Fatal file loading error: {e}")
        )
        return [], skipped_records

def calculate_average_scores(
    eval_base: str, 
    reference_base: str, 
    category: str,
    dataset_case_limits: dict = None
) -> Tuple[Dict[str, float], List[Dict[str, str]]]:
    """
    计算所有JSON文件的平均分数，并收集跳过记录
    返回: (平均分数, 所有异常记录列表)
    """
    
    # 1. 获取所有预期路径
    expected_paths = get_expected_paths(reference_base, category, dataset_case_limits)
    
    if not expected_paths:
        print("\n" + "!"*60)
        print("ERROR: No expected paths found! Cannot proceed.")
        print("!"*60 + "\n")
        return {}, []
    
    all_scores = {
        'recall': [],
        'precision': [],
        'f1_score': [],
        'hallucination_score': []
    }
    
    all_issues = []  # 统一的问题记录列表
    missing_paths_count = 0
    missing_files_count = 0
    processed_files_count = 0
    
    # 2. 遍历所有预期路径，检查评估结果
    print(f"Processing {len(expected_paths)} expected paths...\n")
    
    for idx, expected_rel_path in enumerate(expected_paths, 1):
        if idx % 10 == 0:
            print(f"Progress: {idx}/{len(expected_paths)} paths processed...")
        
        path_exists, full_eval_path, json_files = check_path_existence(
            eval_base, expected_rel_path
        )
        
        # --- 情况1: 路径不存在或无JSON文件 ---
        if not path_exists or not json_files:
            if not Path(full_eval_path).exists():
                reason = "Expected evaluation path does not exist"
                missing_paths_count += 1
            else:
                reason = "Path exists but contains no JSON files"
                missing_files_count += 1
            
            all_issues.append({
                'File_Path': full_eval_path,
                'Skip_Reason': reason
            })
            
            # 为每个缺失的路径/文件添加零分记录
            all_scores['recall'].append(0.0)
            all_scores['precision'].append(0.0)
            all_scores['f1_score'].append(0.0)
            all_scores['hallucination_score'].append(0.0)
            continue
        
        # --- 情况2: 路径存在，处理JSON文件 ---
        for json_file in json_files:
            processed_files_count += 1
            valid_results, skipped_results = extract_scores_from_file(json_file)
            
            # 收集有效分数
            for scores, _ in valid_results:
                all_scores['recall'].append(scores['recall'])
                all_scores['precision'].append(scores['precision'])
                all_scores['f1_score'].append(scores['f1_score'])
                all_scores['hallucination_score'].append(scores['hallucination_score'])
            
            # 收集跳过记录并添加零分
            for item_data, reason in skipped_results:
                file_path = item_data.get('source_file', 'UNKNOWN')
                
                all_issues.append({
                    'File_Path': file_path,
                    'Skip_Reason': reason
                })
                
                # 为每个跳过的记录添加零分
                all_scores['recall'].append(0.0)
                all_scores['precision'].append(0.0)
                all_scores['f1_score'].append(0.0)
                all_scores['hallucination_score'].append(0.0)
    
    # 3. 计算平均值
    avg_scores = {}
    total_samples = len(all_scores['recall'])
    
    print(f"\n{'='*60}")
    print(f"Processing Statistics:")
    print(f"{'='*60}")
    print(f"Total expected paths:          {len(expected_paths)}")
    print(f"Paths with missing directories: {missing_paths_count}")
    print(f"Paths with no JSON files:      {missing_files_count}")
    print(f"JSON files processed:          {processed_files_count}")
    print(f"Total samples (including 0s):  {total_samples}")
    print(f"Total issues recorded:         {len(all_issues)}")
    print(f"{'='*60}\n")
    
    for metric in all_scores:
        if total_samples > 0:
            avg_scores[metric] = sum(all_scores[metric]) / total_samples
        else:
            avg_scores[metric] = 0.0
    
    return avg_scores, all_issues

def save_issues_to_csv(base_path: str, category: str, issues: List[Dict[str, str]]):
    """将所有问题记录保存到CSV文件，只保存文件路径和问题原因"""
    if not issues:
        print("No issues found, skipping CSV generation.")
        return
    
    output_file = f"{base_path}/all_issues_audit_{category}.csv"
    print(f"Saving audit trail of {len(issues)} issues to: {output_file}")

    fieldnames = ['File_Path', 'Skip_Reason']
    
    try:
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(issues)
        print(f"CSV file saved successfully!")
                
    except Exception as e:
        print(f"Error writing CSV file: {e}")


def main():
    # 配置路径
    eval_base_path = "/mnt/data/xinyang/graphrag_benchmark/eval_results/Lightlow"
    reference_base_path = "/mnt/data/shansong/ADC/ADC/is_data_copy"
    category = "simple_QA"
    
    dataset_case_limits = {
        '1_data': (0, 599),   # 1_data 只读取 S_00 到 S_599
        # '2_data': None,     # 2_data 读取全部（可选，不写也默认读取全部）
        # '3_data': (0, 299), # 示例：3_data 限制到 S_00 到 S_299
    }
    
    print("="*60)
    print("Starting Enhanced Score Extraction and Calculation")
    print("="*60)
    print(f"Evaluation base path:  {eval_base_path}")
    print(f"Reference base path:   {reference_base_path}")
    print(f"Category:              {category}\n")
    
    # 计算平均分数并收集所有问题
    avg_scores, all_issues = calculate_average_scores(
        eval_base_path, 
        reference_base_path, 
        category,
        dataset_case_limits
    )
    
    # 保存问题记录到CSV
    save_issues_to_csv(eval_base_path, category, all_issues)
    
    # 打印和保存平均结果
    if avg_scores and any(avg_scores.values()):
        print("\n" + "="*60)
        print("Average Scores (including zero scores for missing/invalid data):")
        print("="*60)
        print(f"Recall:              {avg_scores['recall']:.4f}")
        print(f"Precision:           {avg_scores['precision']:.4f}")
        print(f"F1 Score:            {avg_scores['f1_score']:.4f}")
        print(f"Hallucination Score: {avg_scores['hallucination_score']:.4f}")
        print("="*60)
        
        output_file = f"{eval_base_path}/average_scores_{category}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(avg_scores, f, indent=4)
        print(f"\nAverage results saved to: {output_file}")
    else:
        print("\n" + "!"*60)
        print("No valid scores calculated - check configuration")
        print("!"*60)

if __name__ == "__main__":
    main()