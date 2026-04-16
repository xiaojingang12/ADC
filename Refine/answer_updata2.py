import json

with open('/mnt/data/shansong/ADC/ADC/final_data/updated_4qa.json', 'r') as f:
    updated_2qa = json.load(f)

with open('/mnt/data/shansong/ADC/ADC/4qa_errkeyword_mapping.json', 'r') as f:
    keyword_mapping = json.load(f)

keyword_dict = {}
for item in keyword_mapping:
    truncated_question = item['question'].rstrip('.').strip()
    keyword_dict[truncated_question] = item['keywords']

for item in updated_2qa:
    question = item['question']
    
    if question in keyword_dict:
        item['common_errors'] = keyword_dict[question]
        continue
    found_match = False
    for truncated_q in keyword_dict:
        if question.startswith(truncated_q):
            item['common_errors'] = keyword_dict[truncated_q]
            found_match = True
            break
    if not found_match:
        best_match = None
        max_common_prefix = 0
        
        for truncated_q in keyword_dict:
            common_length = 0
            min_len = min(len(question), len(truncated_q))
            for i in range(min_len):
                if question[i] == truncated_q[i]:
                    common_length += 1
                else:
                    break
            if common_length > max_common_prefix and common_length >= len(truncated_q) * 0.8:
                max_common_prefix = common_length
                best_match = truncated_q
        
        if best_match:
            item['common_errors'] = keyword_dict[best_match]

output_file = '/mnt/data/shansong/ADC/ADC/final_data/updated_4qa_with_errors.json'
with open(output_file, 'w') as f:
    json.dump(updated_2qa, f, indent=2)

print(f"输出文件: {output_file}")

# 显示匹配统计
matched_count = sum(1 for item in updated_2qa if 'common_errors' in item)
print(f"成功匹配: {matched_count} 条")
print(f"未匹配: {len(updated_2qa) - matched_count} 条")

print("\n处理结果示例:")
for i, item in enumerate(updated_2qa[:3]): 
    print(f"\n问题 {i+1}: {item['question'][:80]}...")
    if 'common_errors' in item:
        print(f"common_errors: {item['common_errors']}")
    else:
        print("common_errors: 未找到对应关键词")