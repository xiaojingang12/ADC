import json
import requests
import time
import os

# --- 配置 ---
# API参数
BASE_URL = ""
API_KEY = ""
MODEL = "gpt-4o"

# 读取新闻数据
with open("news_api.json", "r", encoding="utf-8") as f:
    all_news_data = json.load(f)

# 输出文件名
OUTPUT_FILE = "newsqa_single_topic.json" # 更改文件名以区分

# 生成QA对的数量
NUM_QA_PAIRS_TO_GENERATE = 20 # 可根据需要调整

# --- 函数定义 ---

def generate_qa_single_topic(news_list):
    """
    使用大模型基于提供的新闻列表生成围绕单一主题的QA对。
    模型将被指示选择一个主题，并列举与该主题相关的信息。
    """
    # 构建提示词，包含所有新闻
    news_prompt_part = "Here is the list of news articles:\n\n"
    for i, news in enumerate(news_list):
        # 为每条新闻添加索引，方便模型引用
        news_prompt_part += (
            f"[News {i}]\n"
            f"Title: {news['title']}\n"
            f"Description: {news.get('description', 'N/A')}\n"
            f"Published At: {news['published_at']}\n"
            f"Source: {news['source']}\n"
            "---\n"
        )

    prompt = f"""
{news_prompt_part}

Task:
1.  Scan the list of news articles above.
2.  Identify a single, specific, and concrete topic or subject that is mentioned or discussed across multiple articles. This should be a clear focal point, like 'Retrieval-Augmented Generation (RAG) techniques', 'Apple's upcoming product features', 'Impacts of a specific new policy', or 'Performance of a particular company's recent quarter'. Avoid overly broad topics like 'Technology' or 'World News'.
3.  Formulate a question that asks for a list or enumeration of key points, methods, features, impacts, or other relevant details specifically related to the chosen topic. The question should prompt an answer that lists and briefly describes different aspects or examples. An example question format is: "What are the key features of the new iPhone as reported by various sources?" or "What are the main challenges faced by companies adapting to new tariff structures?".
4.  Provide the answer to your question as a list of concise keywords or short phrases, summarizing the core information from the selected articles related to the topic. Format the answer strictly as a list like this: [Point 1, brief description; Point 2, brief description; Another relevant detail]. Do not include any explanatory text or markdown in the answer.
5.  Briefly explain in 1-2 sentences why you chose this specific topic and how the selected articles contribute information to answer your question. This is the 'reason'.
6.  List the titles of the news articles that are relevant to the chosen topic and used to formulate your question and answer, separated by semicolons.

Please respond in JSON format with the following structure:
{{
  "question": "<Your generated question asking for a list related to the single topic>",
  "answer": "[<Point 1, brief description; Point 2, brief description; ...>]", # Strictly this format
  "reason": "<Your reason for choosing this topic and how the articles contribute>",
  "titles": "<Title of Relevant News 1; Title of Relevant News 2; ...>" # Titles of relevant articles
}}

Example Response Format (based on hypothetical news not in the list):
{{
  "question": "What are the different approaches companies are taking to diversify their supply chains away from China?",
  "answer": "[Aritzia, shifting sourcing to Vietnam and Bangladesh; Floor & Decor, increasing inventory from Mexico and India; Generic Tech Firm, investing in automation to reduce location dependency]",
  "reason": "I chose 'Supply Chain Diversification' as the topic because multiple articles discussed how different companies are responding to tariff tensions. These three articles provided specific examples of distinct strategies being employed.",
  "titles": "Aritzia Shifts Supply Chain Amid Tariff Tensions; Floor & Decor Expands Supplier Base Beyond China; Tech Firm Invests in Automation to Hedge Supply Risks"
}}

Remember: Only output the final JSON object. Do not include any other text, thoughts, or markdown code blocks in your response.
"""

    # 准备API请求
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an intelligent assistant tasked with analyzing news articles. "
                    "Your goal is to identify a single, clear topic discussed across multiple articles, "
                    "create a question that asks for a list of key points about that topic, "
                    "and provide a concise, keyword-based answer summarizing the information. "
                    "You must strictly follow the output format instructions."
                )
            },
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7, # 保持一定随机性
        "max_tokens": 1500,
        "response_format": { "type": "json_object" } # 强制JSON输出 (如果API支持)
    }

    max_retries = 3
    for attempt in range(max_retries):
        try:
            print(f"  Attempting API call (Try {attempt + 1}/{max_retries})...")
            response = requests.post(
                f"{BASE_URL}/chat/completions",
                headers=headers,
                json=payload,
                timeout=120 # 增加超时时间
            )
            response.raise_for_status()
            result_text = response.json()['choices'][0]['message']['content'].strip()

            # 尝试解析返回的JSON
            try:
                parsed_result = json.loads(result_text)
                # 基本验证返回的字段
                required_keys = ["question", "answer", "reason", "titles"]
                if all(key in parsed_result for key in required_keys):
                    # 简单检查问题是否像是列举式问题
                    q_lower = parsed_result["question"].lower()
                    if any(word in q_lower for word in ["what are", "list", "enumerate", "approaches", "methods", "types", "ways", "features", "challenges", "impacts", "strategies"]):
                        print("  API call successful and JSON parsed (looks like an enumeration question).")
                        return parsed_result
                    else:
                        print(f"  API response question doesn't seem like an enumeration: {parsed_result['question']}. Retrying...")
                        # 视为失败，进行重试
                else:
                    print(f"  API response missing required keys: {result_text}")
            except json.JSONDecodeError:
                print(f"  Failed to parse API response as JSON: {result_text}")

        except requests.exceptions.Timeout:
            print(f"  API call timed out on attempt {attempt + 1}.")
        except requests.exceptions.RequestException as e:
            print(f"  API request failed on attempt {attempt + 1}: {e}")
        except Exception as e:
             print(f"  An unexpected error occurred during API call {attempt + 1}: {e}")

        if attempt < max_retries - 1:
            wait_time = 2 ** attempt # 指数退避
            print(f"  Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
        else:
            print(f"  Failed to get a valid response after {max_retries} attempts.")
            return {
                "question": "Error generating question.",
                "answer": "[Error]",
                "reason": "Failed to receive or parse a valid response from the API after multiple attempts, or generated question was not of the expected enumeration type.",
                "titles": "N/A"
            }


# --- 主执行逻辑 ---

def main():
    print(f"Starting Single-Topic QA generation for {len(all_news_data)} news items...")
    print(f"Target number of QA pairs to generate: {NUM_QA_PAIRS_TO_GENERATE}")

    # 创建或加载结果文件
    if not os.path.exists(OUTPUT_FILE):
        existing_qa_list = []
        print(f"Creating new output file: {OUTPUT_FILE}")
    else:
        try:
            with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
                existing_qa_list = json.load(f)
            print(f"Loaded existing QA pairs from {OUTPUT_FILE}. Current count: {len(existing_qa_list)}")
        except (json.JSONDecodeError, FileNotFoundError):
            existing_qa_list = []
            print(f"Could not load {OUTPUT_FILE}, starting fresh.")

    initial_count = len(existing_qa_list)
    generated_count = 0

    for i in range(NUM_QA_PAIRS_TO_GENERATE):
        print(f"\n--- Generating Single-Topic QA Pair {i+1}/{NUM_QA_PAIRS_TO_GENERATE} ---")
        
        # 添加延迟以避免API限制
        if i > 0: # 第一次不需要延迟
             delay = 10 # 可根据API限制调整
             print(f"  Waiting for {delay} seconds before next API call...")
             time.sleep(delay)

        # 调用函数生成QA
        qa_dict = generate_qa_single_topic(all_news_data)

        # 保存到列表
        existing_qa_list.append(qa_dict)
        generated_count += 1

        # 定期保存，防止中途出错丢失数据
        if (i + 1) % 5 == 0 or (i + 1) == NUM_QA_PAIRS_TO_GENERATE:
             try:
                 with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                     json.dump(existing_qa_list, f, ensure_ascii=False, indent=2)
                 print(f"  Progress saved to {OUTPUT_FILE}. Total QA pairs: {len(existing_qa_list)}")
             except Exception as e:
                 print(f"  Error saving to file: {e}")


    print("\n" + "="*50)
    print("Single-Topic Generation Complete!")
    print(f"Newly generated QA pairs: {generated_count}")
    print(f"Total QA pairs in {OUTPUT_FILE}: {len(existing_qa_list)}")
    print("="*50)


if __name__ == "__main__":
    main()



