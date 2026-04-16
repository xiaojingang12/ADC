import json
import numpy as np
from collections import defaultdict
from typing import List, Dict, Any, Tuple
import requests
from sklearn.cluster import KMeans
import warnings
import re
import os


QA_FILE = "/mnt/data/shansong/ADC/ADC/final_data/updated_1qa.json"
CORPUS_FILE = "/mnt/data/shansong/ADC/ADC/1corpus_totalid.json"
OUTPUT_FILE = "/mnt/data/shansong/ADC/ADC/final_data/updated_1qa_with_topic_gaps_and_evidence.json"
CLUSTER_OUTPUT_FILE = "/mnt/data/shansong/ADC/ADC/final_data/necluster_to_1questions_mapping.json"
OLLAMA_BASE_URL = ""
OLLAMA_API_KEY = ""
OLLAMA_EMBED_MODEL = "nomic-embed-text:latest"
OLLAMA_CHAT_MODEL = ""
NUM_CLUSTERS = 10
TOP_K_EVIDENCE = 5
CHUNK_SIZE = 600


def get_ollama_embedding(text: str) -> List[float]:
    url = f"{OLLAMA_BASE_URL.rstrip('/')}/api/embed"
    headers = {"Content-Type": "application/json"}
    data = {
        "model": OLLAMA_EMBED_MODEL,
        "input": text
    }

    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        embeddings = result.get("embeddings", [])
        if embeddings and isinstance(embeddings[0], list):
            return embeddings[0]
        else:
            print(f"警告: Ollama embedding API 响应结构异常: {result}")
            raise ValueError("从 Ollama API 响应中获取嵌入失败")
    except requests.exceptions.RequestException as e:
        print(f"获取嵌入时发生错误: {e}")
        raise
    except Exception as e:
        print(f"处理 Ollama embedding 响应时发生意外错误: {e}")
        raise

def cluster_questions(qa_data: List[Dict], num_clusters: int) -> Tuple[np.ndarray, Dict[str, int], Dict[int, List[Dict]]]:

    
    questions = [item['question'] for item in qa_data]
    qa_ids = []
    for item in qa_data:
        ev_list = item.get('evidence_list', [])
        if ev_list and isinstance(ev_list, list) and len(ev_list) > 0:
            first_ev = ev_list[0]
            if isinstance(first_ev, dict):
                qid = first_ev.get('id')
                if qid is not None:
                    qa_ids.append(str(qid))
                else:
                    print(f"  警告: QA项的evidence_list中第一个证据缺少'id'字段，使用索引替代。")
                    qa_ids.append(f"qa_idx_{len(qa_ids)}")
            else:
                print(f"  警告: QA项的evidence_list中第一个元素不是字典，使用索引替代。")
                qa_ids.append(f"qa_idx_{len(qa_ids)}")
        else:
            print(f"  警告: QA项的evidence_list为空或不存在，使用索引替代。")
            qa_ids.append(f"qa_idx_{len(qa_ids)}")

    print(f"正在生成 {len(questions)} 个问题的嵌入...")
    embeddings_list = []
    for i, question in enumerate(questions):
        try:
            embedding = get_ollama_embedding(question)
            embeddings_list.append(embedding)
            if (i + 1) % 50 == 0 or (i + 1) == len(questions):
                 print(f"已编码 {i+1}/{len(questions)} 个问题...")
        except Exception as e:
             print(f"编码问题 {i+1} (id: {qa_ids[i]}) 时失败: {e}")
             raise

    embeddings_array = np.array(embeddings_list)
    print(f"嵌入生成完成。形状: {embeddings_array.shape}")

    print(f"执行标准 KMeans 聚类，簇数: {num_clusters}...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        kmeans_model = KMeans(n_clusters=num_clusters, random_state=42, n_init='auto')
        cluster_labels = kmeans_model.fit_predict(embeddings_array)
    
    qa_id_to_cluster = {qa_id: int(label) for qa_id, label in zip(qa_ids, cluster_labels)}
    cluster_to_qa_items = defaultdict(list)
    for qa_item, cluster_label, qid in zip(qa_data, cluster_labels, qa_ids):
        cluster_to_qa_items[int(cluster_label)].append(qa_item)
    
    unique, counts = np.unique(cluster_labels, return_counts=True)
    cluster_stats = dict(zip(unique, counts))
    print(f"聚类后各簇大小: {cluster_stats}")
    
    print("问题聚类完成。")
    return cluster_labels, qa_id_to_cluster, dict(cluster_to_qa_items)

def load_corpus(file_path: str) -> Dict[str, Any]:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            corpus_data = json.load(f)
        print(f"已加载语料库，包含 {len(corpus_data)} 个文档")
        
        # 构建 total_id -> 文档内容的映射
        total_id_to_doc = {}
        for doc in corpus_data:
            total_id = doc.get('total_id')
            if total_id is not None:
                total_id_to_doc[str(total_id)] = doc
            else:
                print(f"警告: 语料库中的文档缺少 'total_id' 字段: {doc.get('title', 'Unknown Title')[:50]}...")
                
        print(f"构建了 {len(total_id_to_doc)} 个 total_id 到文档的映射。")
        return total_id_to_doc
    except Exception as e:
        print(f"加载语料库文件 {file_path} 失败: {e}")
        raise

def find_relevant_evidence_for_cluster(cluster_qa_items: List[Dict], total_id_to_doc: Dict[str, Any]) -> List[Dict]:
    all_evidence_total_ids = set()
    for qa_item in cluster_qa_items:
        evidence_list = qa_item.get('evidence_list', [])
        for evidence in evidence_list:
            tid = evidence.get('total_id')
            if tid is not None:
                all_evidence_total_ids.add(str(tid))
    
    relevant_evidence = []
    for total_id in all_evidence_total_ids:
        doc = total_id_to_doc.get(total_id)
        if doc:
            text_content = doc.get('context', '') 
            if not text_content:
                 print(f"  警告: 语料库中 total_id 为 '{total_id}' 的文档 'context' 字段为空。")
                 continue
            if len(text_content) > CHUNK_SIZE:
                 # 简单切分，实际应用中可能需要更智能的分块
                 num_chunks = (len(text_content) + CHUNK_SIZE - 1) // CHUNK_SIZE
                 for i in range(num_chunks):
                     start_idx = i * CHUNK_SIZE
                     end_idx = min((i + 1) * CHUNK_SIZE, len(text_content))
                     chunk_text = text_content[start_idx:end_idx]
                     relevant_evidence.append({
                         'total_id': total_id,
                         'chunk_index': i,
                         'text': chunk_text,
                         'source_document': doc
                     })
            else:
                 relevant_evidence.append({
                     'total_id': total_id,
                     'text': text_content,
                     'source_document': doc
                 })
        else:
            print(f"  警告: 在语料库中找不到 total_id 为 '{total_id}' 的文档。")

    return relevant_evidence[:TOP_K_EVIDENCE]

def query_ollama_chat(messages: List[Dict[str, str]], model: str = OLLAMA_CHAT_MODEL) -> str:

    url = f"{OLLAMA_BASE_URL.rstrip('/')}/api/chat"
    headers = {"Content-Type": "application/json"}
    data = {
        "model": model,
        "messages": messages,
        "stream": False
    }
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        return result.get('message', {}).get('content', '')
    except Exception as e:
        print(f"调用 Ollama Chat API 失败: {e}")
        return ""

def detect_topic_gaps_using_cluster_and_evidence(
    qa_data: List[Dict], 
    qa_id_to_cluster: Dict[str, int], 
    cluster_to_qa_items: Dict[int, List[Dict]], 
    total_id_to_doc: Dict[str, Any]
) -> List[Dict]:
    results = []
    topic_patterns = {
        "asset_creation": [
            r"asset.*creation", r"3d.*asset", r"model.*creation", r"texturing", r"mesh.*generation",
            r"procedural.*generation", r"content.*creation", r"asset.*pipeline"
        ],
        "data_sourcing": [
            r"data.*sourc", r"real.*world.*scan", r"lidar.*data", r"sensor.*data", 
            r"photogrammetry", r"scanned.*data", r"synthetic.*data", r"real.*data"
        ],
        "scene_construction": [
            r"scene.*construction", r"scene.*building", r"environment.*creation", 
            r"world.*building", r"scene.*generation", r"level.*design"
        ],
        "simulation_approaches": [
            r"game.*based", r"world.*based", r"physics.*engine", r"simulator.*type",
            r"simulation.*method", r"approach.*simulation"
        ],
        "realism": [
            r"realism", r"fidelity", r"real.*world.*rep", r"visual.*quality", 
            r"photo.*realistic", r"photorealism", r"authenticity"
        ],
        "transfer": [
            r"sim.*to.*real", r"real.*to.*sim", r"domain.*transfer", r"sim2real",
            r"generalization", r"cross.*domain"
        ],
        "performance": [
            r"performance", r"efficiency", r"optimization", r"speed", r"latency",
            r"computational.*cost", r"resource.*usage", r"rendering.*time"
        ],
        "accuracy": [
            r"accuracy", r"precision", r"error.*rate", r"measurement.*accuracy",
            r"spatial.*accuracy", r"detection.*accuracy"
        ]
    }

    def extract_topics_from_text(text: str) -> List[str]:
        if isinstance(text, list):
            text_str = ' '.join([str(item) for item in text])
        elif not isinstance(text, str):
            text_str = str(text)
        else:
            text_str = text
        
        topics = set()
        text_lower = text_str.lower()
        for topic, patterns in topic_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    topics.add(topic)
                    break
        return list(topics)

    print(f"开始检测 {len(qa_data)} 个QA对的Topic缺失情况（基于聚类和证据）...")
    for i, qa in enumerate(qa_data):
        # 从evidence_list中提取当前QA的id
        current_qa_id = None
        ev_list = qa.get('evidence_list', [])
        if ev_list and isinstance(ev_list, list) and len(ev_list) > 0:
            first_ev = ev_list[0]
            if isinstance(first_ev, dict):
                current_qa_id = first_ev.get('id')
        
        if current_qa_id is None:
            print(f"处理第 {i+1}/{len(qa_data)} 个QA对... 跳过（无法确定ID）")
            continue # 如果无法确定ID，则跳过此QA项
            
        print(f"处理第 {i+1}/{len(qa_data)} 个QA对 (ID: {current_qa_id})...")
        
        target_question = qa['question']
        target_answer = qa['answer']
        
        cluster_label = qa_id_to_cluster.get(str(current_qa_id))
        if cluster_label is None:
            print(f"  警告: QA (ID: {current_qa_id}) 未找到对应的聚类标签，跳过。")
            continue

        cluster_qa_items = cluster_to_qa_items.get(cluster_label, [])
        if len(cluster_qa_items) <= 1:
            print(f"  警告: QA (ID: {current_qa_id}) 所属的聚类中只有自己，跳过。")
            continue
        target_topics = set(extract_topics_from_text(target_answer))

        all_cluster_topics = set()
        cluster_topic_info = defaultdict(list) 
        for cluster_qa in cluster_qa_items:

            cluster_qa_id_val = None
            cluster_ev_list = cluster_qa.get('evidence_list', [])
            if cluster_ev_list and isinstance(cluster_ev_list, list) and len(cluster_ev_list) > 0:
                first_cluster_ev = cluster_ev_list[0]
                if isinstance(first_cluster_ev, dict):
                    cluster_qa_id_val = first_cluster_ev.get('id')
            
            if cluster_qa_id_val is not None and str(cluster_qa_id_val) != str(current_qa_id)
                cluster_qa_answer = cluster_qa.get('answer', '')
                topics_in_cluster_qa = extract_topics_from_text(cluster_qa_answer)
                all_cluster_topics.update(topics_in_cluster_qa)
                
  
                if isinstance(cluster_qa_answer, list):
                    cluster_qa_answer_str = ' '.join([str(item) for item in cluster_qa_answer])
                elif not isinstance(cluster_qa_answer, str):
                    cluster_qa_answer_str = str(cluster_qa_answer)
                else:
                    cluster_qa_answer_str = cluster_qa_answer
                
                sentences = re.split(r'[.!?]+', cluster_qa_answer_str)
                for sentence in sentences:
                    sentence = sentence.strip()
                    for topic in topics_in_cluster_qa:
                         if any(re.search(pattern, sentence.lower()) for pattern in topic_patterns[topic]):
                             cluster_topic_info[topic].append(sentence)


        missing_topics = all_cluster_topics - target_topics
        
        if missing_topics:
            print(f"  发现缺失主题: {missing_topics}")
            

            relevant_evidence_for_cluster = find_relevant_evidence_for_cluster(cluster_qa_items, total_id_to_doc)
            

            prompt_parts = [
                """You are an AI assistant tasked with analyzing the informational completeness of a question-answer pair.
                  Current question: {target_question}
                  Current answer: {target_answer}
                  The evidence documents related to this question contain the following topics, which are not mentioned in the current answer: {list(missing_topics)}.
                  Relevant evidence excerpts from the corpus (extracted from the 'context' field based on the total_id linkage in middle_evidence_list): {json.dumps(all_evidence_content, ensure_ascii=False, indent=2)[:1000]}...
                  Please carefully analyze the current answer and the missing thematic information in the associated evidence.
                  Determine whether the current answer genuinely needs to be supplemented with important information related to these missing topics.
                  Consider the following points:
                  Are the missing topics directly relevant to the current question?
                  Does the current answer already address the core requirements of the question?
                  Are the missing topics essential components of the question, or are they merely related but non-essential information?
                  Is the current answer already sufficiently complete and accurate?
                  Respond with 'YES' if the current answer indeed requires supplementation with important information from the missing topics.
                  Otherwise, respond with 'NO' to indicate that the current answer is already sufficiently complete.
                  If you answer 'YES', please briefly explain why the information from these missing topics is important for the current answer.
                  If you answer 'NO', please briefly explain why these missing topics are not essential components of the current answer. """   
            ]
            prompt = "\n".join(prompt_parts)
            
            messages = [
                {"role": "user", "content": prompt}
            ]
            
            try:
                ollama_response = query_ollama_chat(messages)
                print(f"  Ollama 响应: {ollama_response[:100]}...")
                
                response_lower = ollama_response.lower()
                needs_addition = False
                if "yes" in response_lower and "no" not in response_lower:
                    needs_addition = True
                elif "no" in response_lower and "yes" not in response_lower:
                    needs_addition = False
                elif ("yes" in response_lower and "no" in response_lower) or ("yes" not in response_lower and "no" not in response_lower):
                    if any(keyword in response_lower for keyword in ["need", "important", "should", "indeed", "necessary"]):
                        needs_addition = True
                    elif any(keyword in response_lower for keyword in ["not need", "already", "sufficient", "not", "unnecessary"]):
                        needs_addition = False
                    else:
                        needs_addition = False
                
                result = {
                    'index': i,
                    'id': str(current_qa_id), 
                    'question': target_question,
                    'answer': target_answer,
                    'cluster': cluster_label,
                    'cluster_size': len(cluster_qa_items),
                    'missing_topics': list(missing_topics),
                    'cluster_topic_info': {topic: info_list[:3] for topic, info_list in cluster_topic_info.items()}, # 只取前3条
                    'relevant_evidence': relevant_evidence_for_cluster,
                    'ollama_response': ollama_response,
                    'needs_topic_addition': needs_addition
                }
                

                updated_answer = target_answer
                insertion_info = None
                if needs_addition:

                    update_prompt_parts = [
                       f"Current question: {target_question}",
f"Current answer: {target_answer}",
f"Missing topics: {list(missing_topics)}",
f"Relevant information from other answers in the same cluster: {json.dumps(cluster_topic_info, ensure_ascii=False, indent=2)[:1000]}...",
f"Relevant evidence from the corpus: {json.dumps(relevant_evidence_for_cluster, ensure_ascii=False, indent=2)[:1000]}...",
f"Please analyze the structure of the current answer and determine where the missing thematic information should be inserted.",
f"Provide the following information:",
f"1. Insertion position: Which answer list(s) should the new content be added to (e.g., one or multiple lists)?",
f"2. Insertion reason: Briefly explain why this position was chosen.",
f"3. New content: Generate the exact snippet to insert, based on the missing topics and relevant information.",
f"4. Updated full answer: The complete answer after inserting the new content into the appropriate position(s).",
f"Return the result strictly in the following JSON format:",
f'{{"insertion_position": "...", "reason": "...", "new_content": "...", "updated_answer": "..."}}'
                    ]
                    update_prompt = "\n".join(update_prompt_parts)
                    
                    update_messages = [
                        {"role": "user", "content": update_prompt}
                    ]
                    
                    update_response = query_ollama_chat(update_messages)
                    print(f"  答案更新响应: {update_response[:100]}...")
                    

                    try:
                        json_start = update_response.find('{')
                        json_end = update_response.rfind('}') + 1
                        if json_start != -1 and json_end != 0:
                            json_str = update_response[json_start:json_end]
                            insertion_info = json.loads(json_str)
 
                            updated_answer = insertion_info.get('updated_answer', target_answer)
                        else:
                            print(f"  无法解析答案更新响应中的JSON: {update_response}")
                            # 如果无法解析JSON，保持原答案不变
                    except json.JSONDecodeError as e:
                        print(f"  解析答案更新响应JSON失败: {e}")
                        print(f"  响应内容: {update_response}")

                result['updated_answer'] = updated_answer
                result['insertion_info'] = insertion_info
                
                results.append(result)
            except Exception as e:
                print(f"  调用 Ollama 判断时发生错误: {e}")
                result = {
                    'index': i,
                    'id': str(current_qa_id), 
                    'question': target_question,
                    'answer': target_answer,
                    'cluster': cluster_label,
                    'cluster_size': len(cluster_qa_items),
                    'missing_topics': list(missing_topics),
                    'cluster_topic_info': {topic: info_list[:3] for topic, info_list in cluster_topic_info.items()},
                    'relevant_evidence': relevant_evidence_for_cluster,
                    'ollama_response': f"Error calling Ollama: {e}",
                    'needs_topic_addition': False, 
                    'updated_answer': target_answer,
                    'insertion_info': None 
                }
                results.append(result)
        else:
            print(f"  没有发现缺失主题，跳过")
            continue

    return results

def save_results(results: List[Dict], output_path: str):
    try:
        def convert(o):
            if isinstance(o, np.integer):
                return int(o)
            if isinstance(o, np.floating):
                return float(o)
            if isinstance(o, np.ndarray):
                return o.tolist()
            if isinstance(o, np.bool_):
                return bool(o)
            raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=convert)
        print(f"结果已保存到 {output_path}")
    except Exception as e:
        print(f"保存结果到 {output_path} 失败: {e}")
        raise

def save_cluster_mapping(cluster_to_qa_items: Dict[int, List[Dict]], output_file: str):
    try:
        cluster_to_qa_indices = {} 
        for cluster_id, qa_items in cluster_to_qa_items.items():
            qa_indices = []
            for qa_item in qa_items:
                qa_indices.append(len(qa_indices)) 

            print(f"注意: 聚类到问题的映射逻辑已更新，但当前实现可能不够精确。")
            cluster_to_qa_indices[int(cluster_id)] = len(qa_items)
        

        print(f"注意: 聚类到问题的映射逻辑已更新，但当前实现可能不够精确。")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(cluster_to_qa_indices, f, indent=2, ensure_ascii=False)
        print(f"聚类大小信息已保存到 {output_file}")
    except Exception as e:
        print(f"保存聚类映射到 {output_file} 失败: {e}")
        raise


def main():

    print("1. 加载QA数据...")
    with open(QA_FILE, 'r', encoding='utf-8') as f:
        qa_data = json.load(f)
    print(f"   已加载 {len(qa_data)} 个QA对")

    print("\n2. 对问题进行聚类...")
    cluster_labels, qa_id_to_cluster, cluster_to_qa_items = cluster_questions(qa_data, NUM_CLUSTERS)
    save_cluster_mapping(cluster_to_qa_items, CLUSTER_OUTPUT_FILE)

    print("\n3. 加载语料库并建立映射...")
    total_id_to_doc = load_corpus(CORPUS_FILE)

    print("\n4. 检测Topic缺失并关联证据...")
    results = detect_topic_gaps_using_cluster_and_evidence(qa_data, qa_id_to_cluster, cluster_to_qa_items, total_id_to_doc)

    print(f"\n5. 保存结果到 {OUTPUT_FILE}...")
    save_results(results, OUTPUT_FILE)

    total_qas = len(qa_data)
    gap_qas = len(results)  
    need_addition_count = sum(1 for res in results if res.get('needs_topic_addition', False))

    print(f"\n--- 检测完成 ---")
    print(f"总QA对数量: {total_qas}")
    print(f"存在潜在Topic缺失的QA对数量: {gap_qas}")
    print(f"其中需要补充信息的QA对数量: {need_addition_count}")
    print(f"结果详细信息已保存到: {OUTPUT_FILE}")
    print(f"聚类信息已保存到: {CLUSTER_OUTPUT_FILE}")


    if results:
        print(f"\n--- 示例结果 (前3个) ---")
        for i, result in enumerate(results[:3]):
            print(f"\nQA对 {i+1} (索引: {result['index']}, ID: {result['id']}):")
            print(f"  问题: {result['question'][:100]}...")
            print(f"  所属聚类: {result['cluster']} (大小: {result['cluster_size']})")
            print(f"  缺失主题: {result['missing_topics']}")
            print(f"  需要补充: {result['needs_topic_addition']}")
            print(f"  原始答案: {result['answer'][:100]}...")
            print(f"  更新答案: {result['updated_answer'][:100]}...")
            print(f"  插入信息: {result['insertion_info']}")
            print(f"  Ollama 判断: {result['ollama_response'][:200]}...")
            print(f"  相关证据 (前2个): {result['relevant_evidence'][:2]}")
    else:
        print(f"\n没有发现任何QA对存在Topic缺失或需要补充信息。")

if __name__ == "__main__":
    main()



