import json

def load_json_file(file_path: str) -> any:
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json_file(data: any, file_path: str):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def generate_qa_keyword_mapping(errors_file_path: str, output_file_path: str):
    """根据错误关键词数据生成QA关键词映射文件"""
    errors_data = load_json_file(errors_file_path)
    
    errors_by_qa = errors_data.get("errors_by_qa", {})
    
    qa_keyword_mapping = []
    
    for qa_id, errors_list in errors_by_qa.items():
        if errors_list:  
            question_text = errors_list[0]["question"]
            
            keywords = [error["keyword"] for error in errors_list]
            
            mapping_item = {
                "question": question_text,
                "keywords": keywords
            }
            
            qa_keyword_mapping.append(mapping_item)
    
    print(f"正在保存映射数据到 {output_file_path}...")
    save_json_file(qa_keyword_mapping, output_file_path)
    
    print(f"生成完成共处理了 {len(qa_keyword_mapping)} 个QA对")
    
    # 显示一些统计信息
    total_keywords = sum(len(item["keywords"]) for item in qa_keyword_mapping)
    print(f"总共包含 {total_keywords} 个关键词")

def main():
    errors_file_path = "/mnt/data/shansong/ADC/ADC/4qa_keyword_validation_errors.json"
    output_file_path = "/mnt/data/shansong/ADC/ADC/4qa_errkeyword_mapping.json"
    
    generate_qa_keyword_mapping(errors_file_path, output_file_path)

if __name__ == "__main__":
    main()



