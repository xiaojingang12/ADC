import json
import logging
import time
import random
import requests
from openai import OpenAI
import os
from magic_pdf.data.data_reader_writer import FileBasedDataWriter, FileBasedDataReader
from magic_pdf.data.dataset import PymuDocDataset
from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
from magic_pdf.config.enums import SupportedPdfParseMethod
from typing import Optional
import re
from contextlib import suppress
import torch
import gc
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"  
torch.cuda.empty_cache()

def fetch_papers(year=2023):
    api_url = "https://api.openreview.net/notes"
    params = {
        "invitation": f"ICLR.cc/{year}/Conference/-/Blind_Submission",
        "details": "replyCount"
    }

    response = requests.get(api_url, params=params)
    if response.status_code != 200:
        print(f"API 请求失败，状态码: {response.status_code}")
        return []

    papers = response.json()
    results = []

    for paper in papers.get("notes", []):
        paper_id = paper.get("id")
        title = paper.get("content", {}).get("title", "无标题")
        authors = paper.get("content", {}).get("authors", [])
        pdf_link = f"https://openreview.net/pdf?id={paper_id}"
        results.append({"title": title, "authors":authors, "pdf_link": pdf_link, "id": paper_id})

    return results

import requests
import time
import random

def get_sum(id):
    api_base_url = "https://api.openreview.net/notes"
    
    
    params = {
        "forum": id,
        "trash": "true",
        "details": "replyCount,writable,revisions,original,overwriting,invitation,tags",
        "limit": 1000,
        "offset": 0
    }
    
    # 指数退避参数
    max_retries = 5  # 最大重试次数
    base_delay = 1   # 初始延迟时间（秒）
    
    for attempt in range(max_retries + 1):
        try:
            response = requests.get(api_base_url, params=params)
            
            # 处理429错误 - Too Many Requests
            if response.status_code == 429:
                if "Retry-After" in response.headers:
                    # 使用服务器建议的等待时间
                    wait_time = int(response.headers["Retry-After"])
                else:
                    # 指数退避算法
                    wait_time = base_delay * (2 ** attempt) + random.uniform(0, 0.1)
                
                print(f"请求过频繁，等待 {wait_time:.2f} 秒后重试... (尝试: {attempt+1}/{max_retries})")
                time.sleep(wait_time)
                continue
                
            # 处理其他错误状态
            if response.status_code != 200:
                print(f"API 请求失败，状态码: {response.status_code}")
                return []
                
            # 如果成功，处理响应
            reviews = []
            papers = response.json()
            for paper in papers.get("notes", []):
                try:
                    review = {
                        "summary": paper["content"]["summary_of_the_paper"].strip(),
                        "strength_and_weaknesses": paper["content"]["strength_and_weaknesses"].strip(),
                    }
                    # 过滤空评论
                    if any(review.values()):
                        reviews.append(review)
                except KeyError:
                    continue
                    
            return reviews
            
        except Exception as e:
            print(f"请求异常: {str(e)}")
            # 随机指数退避
            wait_time = base_delay * (2 ** attempt) + random.uniform(0, 0.5)
            time.sleep(wait_time)



"""
pdf_response = requests.get("https://openreview.net/pdf?id=bbVH40jy7f")
pdf = pdf_response.content
with open("bbVH40jy7f.pdf","wb") as f:
    f.write(pdf)
# 输入PDF文件路径
pdf_file_name = "bbVH40jy7f.pdf"
local_image_dir, local_md_dir = "output/images", "output"

# 创建输出目录
os.makedirs(local_image_dir, exist_ok=True)
image_writer, md_writer = FileBasedDataWriter(local_image_dir), FileBasedDataWriter(local_md_dir)

# 读取PDF内容
reader = FileBasedDataReader("")
pdf_bytes = reader.read(pdf_file_name)

# 创建数据集实例并进行分类
ds = PymuDocDataset(pdf_bytes)
if ds.classify() == SupportedPdfParseMethod.OCR:
    infer_result = ds.apply(doc_analyze, ocr=True)
    pipe_result = infer_result.pipe_ocr_mode(image_writer)
else:
    infer_result = ds.apply(doc_analyze, ocr=False)
    pipe_result = infer_result.pipe_txt_mode(image_writer)

# 输出Markdown文件
pipe_result.dump_md(md_writer, f"{os.path.splitext(pdf_file_name)[0]}.md", local_image_dir)
paper_content=open("/home/shansong/output/bbVH40jy7f.md",'r',encoding='utf-8').read()
"""
# 获取论文内容并转换为MARKDOWN格式
def pdf_content(link):
   pdf_response = requests.get(link)
   pdf = pdf_response.content
   with open("link.pdf","wb") as f:
       f.write(pdf)
   # 输入PDF文件路径
   pdf_file_name = "link.pdf"
   local_image_dir, local_md_dir = "output/images", "output"

   # 创建输出目录
   os.makedirs(local_image_dir, exist_ok=True)
   image_writer, md_writer = FileBasedDataWriter(local_image_dir), FileBasedDataWriter(local_md_dir)

   # 读取PDF内容
   reader = FileBasedDataReader("")
   pdf_bytes = reader.read(pdf_file_name)

   # 创建数据集实例并进行分类
   ds = PymuDocDataset(pdf_bytes)
   if ds.classify() == SupportedPdfParseMethod.OCR:
     infer_result = ds.apply(doc_analyze, ocr=True)
     pipe_result = infer_result.pipe_ocr_mode(image_writer)
   else:
     infer_result = ds.apply(doc_analyze, ocr=False)
     pipe_result = infer_result.pipe_txt_mode(image_writer)

    # 输出Markdown文件
   md_filename = f"{os.path.splitext(pdf_file_name)[0]}.md"
   pipe_result.dump_md(md_writer, md_filename, local_image_dir)
   md_file_path = os.path.join(local_md_dir, md_filename)
    
    # 读取Markdown内容
   with open(md_file_path, 'r', encoding='utf-8') as f:
        paper_content = f.read()
    
    # 删除生成的文件
   with suppress(Exception):
        os.remove(pdf_file_name)       # 删除PDF文件
        os.remove(md_file_path)        # 删除Markdown文件
    
   return paper_content 

class TopicExtractor:
    def __init__(self, base_url="1", api_key="", model="gpt-4o"):    
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.logger = logging.getLogger(__name__)
        
    def _call_llm(self, prompt, max_retries=5):
        retries = 0
        while retries < max_retries:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=100,
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                self.logger.warning(f"API调用失败 ({retries+1}/{max_retries}): {str(e)}")
                time.sleep(2)
                retries += 1
        raise RuntimeError(f"超过最大重试次数 ({max_retries})")

    def extract_topic(self, reviews_data):
    # 将评论内容拼接成字符串作为输入
     combined_text = ""
     for review in reviews_data:
        combined_text += review.get("summary", "") + " "
        combined_text += review.get("strength_and_weaknesses", "") + " "
        combined_text += "\n\n"  # 添加分隔符
    
    # 使用安全的字符串拼接代替 f-string
     prompt = """
Please analyze the provided academic paper reviews and extract 10-15 technical terms or key phrases that best represent the paper's core contributions and methodological focus.Output Requirements:Directly list extracted terms/phrases
Use only technical terminology (no explanations) 
For each review, a topic is returned. Then, the topics returned by each review are summarized. The importance is determined by the number of occurrences, and the important ones are ranked at the top
Prioritize compound terms over single words
No explanations or introductory phrases
Pure technical terms formatted as a comma-separated list
No markdown or numbering
Exclude general academic verbs/adjectives Format as:
[term1], [term2], [term3], [...]

#example:

#input:"reviews": [
        {
            "summary": "This work proposes a novel dual-layer retrieval RAG framework that seamlessly integrates graph structures into text indexing, enabling efficient and complex information retrieval. The system leverages multi-hop subgraphs to extract global information, excelling in complex, multi-domain queries. Integrating graph and vector-based methods, it reduces retrieval time and computational costs compared to traditional text chunk traversal. Incremental update algorithms ensure seamless integration of new data, maintaining real-time accuracy and relevance.",
            "strengths": "1. This paper is well-presented and easy to follow. The authors provide a clear motivation and a good introduction to the problem.\n2. The proposed framework can be easily plugged into existing LLMs and corpus.\n3. Extensive and solid experiments.",
            "weaknesses": "1. It is recommended to include performance results and comparative analysis on some Closed QA benchmarks to demonstrate the system's improvements and advantages in practical applications.\n2. Coreference resolution and disambiguation were not performed during the graph construction process."
        },
        {
            "summary": "This paper introduces LightRAG, a novel Retrieval-Augmented Generation (RAG) system designed to enhance the performance of large language models by integrating external knowledge sources more effectively. \n\nThe key contributions of LightRAG include:\n\n1.  LightRAG incorporates graph structures into text indexing and retrieval processes, allowing for better representation of complex interdependencies among entities.\n2. The system employs Dual-level retrieval paradigm where both low-level and high-level retrieval strategies to capture detailed information about specific entities as well as broader topics and themes.\n3. By combining graph structures with vector representations, LightRAG enables fast retrieval of related entities and relationships. It also features an incremental update algorithm for quick adaptation to new data, which is an added advantage of this novel approach.\n4. The graph-based approach allows for extraction of global information from multi-hop subgraphs, enhancing the system's ability to handle complex queries spanning multiple document chunks.\n5. The paper presents experimental results are good  and well presented in comparing LightRAG to existing RAG baseline methods across four datasets from different domains. The evaluation focuses on comprehensiveness, diversity, empowerment, and overall performance. \n6. Results show that LightRAG consistently outperforms chunk-based retrieval methods, especially for larger datasets and complex queries requiring comprehensive consideration of the dataset's background.\n7. The authors argue that LightRAG addresses key limitations of existing RAG systems, such as reliance on flat data representations and inadequate contextual awareness. \n8. The authors also compares LightRAG approach with GraphRAG which is implemented on similar lines and how LightRAG outperforms GraphRAG in retrieval speed and complexity by retrieving entities and relationships rather than community-based traversal retrieval method.\n9. By incorporating graph structures and employing a dual-level retrieval paradigm, LightRAG aims to provide more coherent and contextually rich responses to user queries.",
            "strengths": "Originality:\n\n1. It introduces a novel graph-based text indexing paradigm for RAG systems, moving beyond flat document representations and community based entities to capture complex entity relationships by utilizing a  comprehensive knowledge graph to facilitate rapid\nand relevant document retrieval, enabling a deeper understanding of complex queries. \n\n2. The dual-level retrieval paradigm, combining low-level and high-level retrieval strategies, is an innovative approach to handling diverse query types.\n\n3. The integration of graph structures with vector representations for efficient retrieval is a creative combination of existing techniques\n\nQuality:\n\n1. Comprehensive experimental evaluation across multiple datasets and baselines is well presented with clarity to understand how much the proposed approach outperforms other existing RAG approaches' baselines. \n\n2. Detailed complexity analysis of the proposed framework gives a clear idea on how GraphRAG works in two parts compared to conventional and GraphRAG approaches. \n\n3. Thorough ablation studies to validate the contributions of different components such as Low-level-only Retrieval, high-level-only Retrieval and hybrid mode leading to the finding of the fact that resulting variant does not show significant performance declines across\nall four datasets when use of original text is eliminated in our retrieval process.\n\n\nClarity:\n\n1. The paper is generally well-structured and clearly written with clear architecture, specific data set selection, baseline definition and experimentation results showcasing the improvement compared to baseline. \n\n2. The introduction clearly outlines the motivation, challenges, and contributions and the gaps that are existing in current approavhes. \n\n3. The methodology is explained in detail with formal definitions and algorithms\n\n4. Experimental settings and results are presented systematically and figures/tables effectively illustrate and correlate with the framework and results\n\nSignificance:\n\n1. It addresses key limitations of existing RAG systems, particularly in handlig complex queries and large scale datasets.\n\n2. The proposed framework shows consistent performance improvements over state-of-the-art baselines across multiple datasets and evalution dimensions\n\n3. The approach is scalable and adaptable to new data, making it relevant for real-world applications\n\n4. By enhancing the capabilities of RAG systems, this work has potential impacts on various domains requring advanced information retrieval and generation and large data sets.",
            "weaknesses": "1. The paper compares LightRAG primarily to basic RAG approaches like Naive RAG and RQ-RAG. However, it lacks comparison to more recent and advanced RAG methods such as:\n\n           a. Self-RAG (Asai et al., 2023)\n           b. FLARE (Zhang et al., 2023)\n           c. Chain-of-Note (Wu et al., 2023)\n\nIncluding these comparisons would provide a more comprehensive evaluation of LightRAG's performance relative to the current state-of-the-art.\n\n2. While the paper provides a brief complexity analysis in Section 3.4, it lacks detailed quantitative comparisons of computational costs between LightRAG and baseline methods. Specifically, the paper should include concrete time and space complexity analysis for both indexing and retrieval phases. Empirical runtime comparisons across different dataset sizes would help illustrate LightRAG's efficiency claims.\n\n3. The paper focuses primarily on the strengths of LightRAG but does not adequately address potential limitations or failure cases. A more balanced discussion could include \n\n\n          a. Scenarios where graph-based indexing might underperform traditional methods\n          b. Potential scalability challenges for extremely large datasets\n          c. Discussion of how LightRAG handles queries that don't align well with the graph structure\n          d. Future directions on how any potential limitations on LightRAG can be further researched and improved. \n\n4. Insufficient analysis of graph update efficiency: The paper claims that LightRAG can efficiently adapt to new data, but it lacks detailed empirical evidence supporting this claim. Including experiments that measure update times for incrementally adding new documents would strengthen this argument.\n\n5. The experimentation uses a wide range of datasets covering multiple domains , however it could  be more diverse if more LLMs with different sizes have been included in the experiment instead of defaulting and relying only on GPT-4o-mini."
        },
        {
            "summary": "This paper introduces LightRAG, a Retrieval-Augmented Generation system designed to improve retrieval accuracy, contextual relevance, and processing efficiency. Compared to GraphRAG, LightRAG removes the community detection part and saves computational overhead for inference queries The authors present a dual-level retrieval mechanism, combining low-level, entity-focused retrieval with high-level, topic-focused retrieval, allowing for both detailed and abstract query handling. Additionally, LightRAG features an incremental update algorithm to integrate new data dynamically, aimed at making it more responsive to changing datasets.",
            "strengths": "1.\tThe dual-level retrieval approach is well-conceived, offering flexibility to handle both specific and abstract queries. This strategy appears useful for adapting responses to varied user intents, a valuable feature for broad applications.\n2.\tThe incremental update algorithm proposed is promising in ensuring LightRAG’s adaptability to new information, enabling it to stay current in dynamic environments without needing a full index rebuild.\n3.\tThe study evaluates LightRAG on multiple datasets across diverse domains, which provides a general understanding of its performance across different types of content.",
            "weaknesses": "1.\tThe paper does not clearly articulate LightRAG’s advancements over prior graph-based RAG systems, particularly GraphRAG. It is unclear whether LightRAG merely omits multi-layer and community-based construction and retrieval as used in GraphRAG, or if it introduces other distinctive enhancements.\n2.\tThe paper omits critical details on the graph construction process, such as specific techniques for entity disambiguation.\n3.\tHow local and global query keywords are matched within the graph is not described in detail, which reduces clarity on how dual-level retrieval is operationalized.\n4.\tThe paper lacks a discussion comparing the performance of different LLMs, missing a comparative analysis of their effectiveness within LightRAG.\n5.\tAlthough LightRAG is described as efficient, the complexity analysis provided is superficial and does not include actual runtime.\n6.\tSubjective metrics and limited statistical analysis reduce the rigor of the evaluation. LLM-based judgments could be supplemented with human evaluations or standardized benchmarks for improved reliability.\n7.\tWhen new data is incrementally added to the knowledge graph, the method for ensuring consistency with the pre-existing structure is not specified. Details on version control, conflict resolution, and synchronization strategies are necessary to understand how the system maintains the accuracy and coherence of the knowledge graph over time.\n8.\tThe keyword matching approach, crucial to the dual-level retrieval paradigm, is not clearly described.\n9.\tWhile the paper claims that incremental updates maintain efficiency, it lacks quantitative data on how this approach compares to a full re-indexing in terms of accuracy and computational cost. Specifically, the paper should include benchmarks demonstrating the performance impact of incremental updates versus complete recalculations, as well as any observed trade-offs in accuracy.\n10.\tLine 173 contains a typographical error: \"LightRAGcombines\" should be corrected to \"LightRAG combines.\""
        },
        {
            "summary": "This paper introduces LightRAG, which addresses the limitations of existing RAG systems, such as reliance on flat data representations and inadequate contextual awareness, by incorporating graph structures into text indexing and retrieval processes. This dual-level retrieval system improves information retrieval from both low-level and high-level knowledge discovery and efficiently retrieves related entities and their relationships. LightRAG also features an incremental update algorithm for timely integration of new data, ensuring the system remains effective in rapidly changing data environments.",
            "strengths": "The paper proposes LightRAG, which effectively balances performance and efficiency in tasks need global information through knowledge graph construction.",
            "weaknesses": "1. The evaluation datasets are limited to college textbooks, raising questions about generalizability to other types of corpora. Additionally, the use of only one LLM makes it difficult to assess LightRAG's dependency on LLM capabilities.\n\n2. The query construction methodology lacks sufficient justification. The authors fail to adequately explain the rationale behind their approach and its effectiveness in evaluating LightRAG. Particularly in paragraph of line 273,  it's unclear how global information is incorporated into the query construction process, which is crucial for assessing the method's effectiveness.\n\n3. Merely using the win rates metric is not solid enough. Especially taking GraphRAG as an example, it is not clear whether it can be stated that there is a 45.56% probability of being superior to LightRAG. Moreover, in the mixed dataset, GraphRAG is even slightly higher in three dimensions, which weakens the conclusion of the experiment.\n\n4. The use of Diversity as a metric for evaluating RAG responses is questionable. There's significant doubt whether higher diversity necessarily indicates better answers, as it might introduce excessive noise. In Table 3, the LLM's interpretation of diversity appears to align more with comprehensiveness rather than true diversity. This is particularly concerning as Diversity is presented as LightRAG's main advantage over baselines, significantly undermining confidence in the experimental results.\n\n5. The practical implementation of low-level and high-level retrieval mechanisms lacks clarity. While Section 3.2 outlines their principles and describes a three-step process for \"Integrating Graph and Vectors for Efficient Retrieval,\" it's unclear whether these steps represent low-level retrieval, high-level retrieval, or a combination thereof.\n\n6. The Cost Analysis section is incomplete. While token consumption is analyzed, temporal performance analysis is missing. Although token consumption correlates with time, they're not equivalent, as input and output processing times can differ significantly. Furthermore, the analysis omits the most resource-intensive component - indexing construction - focusing only on retrieval and incremental updates.\n\n7. Disappointingly, overall, the experimental and analysis sections of the article do not well correspond to the three challenges that the author proposed to solve in paragraph line 053.\n    1. Comprehensive Information Retrieval: The query construction method is not elaborated on how to ensure globality, and the types of experimental datasets and settings are not sufficient.\n    2. Enhanced Retrieval Efficiency: The author mentioned a significant reduction in response time. However, in the subsequent experiments, only the analysis of token consumption was carried out, and there was no mention of the improvement in time at all.\n    3. Rapid Adaptation to New Data: No relevant experiments were provided to support this conclusion."
        }

#output:
 [
        "Dual-level retrieval paradigm",
        " GraphRAG",
        " Retrieval-Augmented Generation (RAG)",
        " LightRAG",
        " Dual-layer retrieval framework",
        " Incremental update algorithms",
        " Multi-hop subgraphs",
        " Graph structures",
        " Vector representations",
        " Efficient retrieval",
        " Complex information retrieval",
        " Knowledge graph construction",
        " Entity disambiguation",
        " Keyword matching approach",
        " Low-level and high-level query handling",
        " Contextual relevance",
        " Processing efficiency",
        " Adaptive knowledge graph",
        " Dynamic environments",
        " Scalability challenges",
        " Version control strategies",
        " Conflict resolution techniques",
        " Synchronization"
    ]
""" + "\n\nReviews:\n" + combined_text
    
     self.logger.info(f"发送给LLM的提示语长度: {len(prompt)}字符")
     raw_response = self._call_llm(prompt)
    
    # 清理响应中的标点符号并分割成列表
     cleaned_response = raw_response.translate(str.maketrans('', '', '“”"\'。！？'))
     topics = [t.strip() for t in cleaned_response.split(',') if t.strip()]
    
     self.logger.info(f"提取到 {len(topics)} 个主题")
     return topics

# 配置日志和模型
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

CONFIG = {
    #"model_name": "llama3.1:8b4k",  # 模型名称
    "model_name": "gpt-4o",  # 模型名称
}

# 创建主题提取器
extractor = TopicExtractor(model=CONFIG["model_name"])

# 修改后的主处理循环
paper_inf = fetch_papers()
for i in paper_inf[910:]:
    print(f"正在处理论文: {i['title']} by {', '.join(i['authors'])}")
    review = get_sum(i["id"])
    if review:

        
        print("获取论文信息，开始生成")
        # 提取主题
        topic = extractor.extract_topic(review)

        
        paper_content = pdf_content(i["pdf_link"])
        
        with open("/mnt/data/shansong/ADC/simple.json", 'a', encoding="utf-8") as f:
            json.dump(
                {
                    "title": i["title"],
                    "authors": i["authors"],
                    "link": i["pdf_link"],
                    "reviews": review,
                    "content": paper_content,
                    "question": "Please summarize the paper %s" % i["title"],
                    "question_type": "summarization", 
                    "topic": topic,  
                    "answer": "The paper discusses the following topics: %s" % ', '.join(topic)
                }, 
                f, 
                indent=4
            )
            # 添加分隔符以确保JSON可读性
            f.write(",\n")
    else:
        print(f"未找到论文 {i['title']} 的评论信息，跳过处理。")        
