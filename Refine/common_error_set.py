import json
import re
import time
from typing import List, Dict, Set
from openai import OpenAI

class EvidenceValidator:
    def __init__(
        self,
        qa_file_path: str,
        context_file_path: str,
        api_key: str = "",
        base_url: str = "",
        model_name: str = "gpt-4o"
    ):
        self.qa_file_path = qa_file_path
        self.context_file_path = context_file_path
        self.qa_data = []
        self.context_data = {}
        self.common_errors = set()
        
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name

    def load_data(self):
        with open(self.qa_file_path, 'r', encoding='utf-8') as f:
            self.qa_data = json.load(f)
        
        with open(self.context_file_path, 'r', encoding='utf-8') as f:
            context_list = json.load(f)
        
        for item in context_list:
            total_id = item.get('total_id')
            if total_id is not None:
                self.context_data[total_id] = item.get('context', "")

    def _get_evidence_texts(self, evidence_list: List[Dict], max_length: int = 100000) -> str:
        evidence_parts = []
        total_len = 0

        for ev in evidence_list:
            total_id = ev.get('total_id')
            if total_id not in self.context_data:
                continue

            context = self.context_data[total_id]
            if isinstance(context, str):
                text = context.strip()
            elif isinstance(context, list):
                text = "\n\n".join(str(p).strip() for p in context if p and str(p).strip())
            else:
                text = str(context)

            if not text:
                continue

            part = f"【证据ID: {total_id}】\n{text}"
            if total_len + len(part) > max_length:
                remaining = max_length - total_len
                if remaining > 50:
                    part = part[:remaining] + "……"
                else:
                    break
            evidence_parts.append(part)
            total_len += len(part)
            if total_len >= max_length:
                break

        if not evidence_parts:
            return "无可用证据。"
        return "\n\n---\n\n".join(evidence_parts)

    def _llm_verify(self, claim: str, evidence_text: str, max_retries: int = 3) -> bool:
        prompt = f"""You are a rigorous academic fact-checking expert. The following evidence comes from one or more academic papers, each starting with 【Evidence ID: ...】 and separated by "---".
Based strictly on this evidence, determine whether the following statement is explicitly supported or logically inferable from at least one paper.
Rules:
Answer "Supported" only if the evidence contains sufficient information to support the statement.
If none of the evidence mentions, clearly implies, or logically leads to the statement—or if the evidence is vague or contradictory—answer "Not Supported".
Do not use external knowledge or make assumptions.
Evidence:
{evidence_text}
Statement:
{claim}
Please respond only with "Supported" or "Not Supported"."""

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=10,
                    timeout=30
                )
                answer = response.choices[0].message.content.strip()
                if "Supported" in answer:
                    return True
                elif "Not Supported" in answer:
                    return False
                else:
                    print(f"模型返回意外结果: '{answer}'，重试中... (尝试 {attempt + 1})")
                    time.sleep(1)
            except Exception as e:
                print(f"调用模型出错: {e}，重试中... (尝试 {attempt + 1})")
                time.sleep(2)
        return False  # 默认不支持

    def validate_answer_against_evidence(self, answer: str, evidence_list: List[Dict]) -> List[str]:
        if not answer.strip():
            return []

        answer_parts = re.findall(r'\[(.*?)\]', answer)
        if not answer_parts:
            sentences = [s.strip() for s in re.split(r'[。！？；\n]', answer) if s.strip()]
            answer_parts = sentences[:5]

        evidence_text = self._get_evidence_texts(evidence_list)
        errors = []

        for part in answer_parts:
            part = part.strip()
            if not part:
                continue
            if not self._llm_verify(part, evidence_text):
                errors.append(part)
        return errors

    def process_all_qa_pairs(self):
        for idx, qa_pair in enumerate(self.qa_data):
            question = qa_pair.get('question', '')
            answer = qa_pair.get('answer', '')
            evidence_list = qa_pair.get('evidence_list', [])
            print(f"正在验证第 {idx + 1}/{len(self.qa_data)} 个 QA 对...")
            errors = self.validate_answer_against_evidence(answer, evidence_list)
            for error in errors:
                self.common_errors.add(error)

    def save_errors_to_file(self, output_file_path: str):
        errors_list = sorted(self.common_errors)  # 排序便于阅读
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump({
                "total_errors_count": len(errors_list),
                "common_errors": errors_list,
                "description": "通过大模型（gpt-4o）验证识别出的不被证据支持的信息片段"
            }, f, ensure_ascii=False, indent=2)

    def run_validation(self, output_file_path: str = "validation_errors.json"):
        print("开始加载数据...")
        self.load_data()
        print(f"加载了 {len(self.qa_data)} 个 QA 对 和 {len(self.context_data)} 篇论文上下文")

        print("开始使用大模型验证答案（每条信息片段独立验证）...")
        self.process_all_qa_pairs()

        print(f"验证完成，识别出 {len(self.common_errors)} 个不被支持的信息片段")
        self.save_errors_to_file(output_file_path)
        print(f"结果已保存到: {output_file_path}")
        return self.common_errors


def main():
    qa_file_path = "/mnt/data/shansong/ADC/ADC/final_data/updated_2qa.json"
    context_file_path = "/mnt/data/shansong/ADC/ADC/2pdf_id.json"
    output_file_path = "/mnt/data/shansong/ADC/ADC/2qa_validation_errors.json"

    validator = EvidenceValidator(
        qa_file_path=qa_file_path,
        context_file_path=context_file_path,
        api_key="sk-SHiLrPzmuREecae9E29f4eA62fD84eA5A3E569026fAf33De",
        base_url="https://api.ai-gaochao.cn/v1",
        model_name="gpt-4o"
    )

    common_errors = validator.run_validation(output_file_path)

    print("\n" + "="*60)
    print("验证结果摘要")
    print("="*60)
    print(f"共发现 {len(common_errors)} 个不被证据支持的信息片段：")
    for i, error in enumerate(sorted(common_errors), 1):
        print(f"{i}. {error}")


if __name__ == "__main__":
    main()

