import os
import json
from pathlib import Path
from magic_pdf.data.data_reader_writer import FileBasedDataWriter, FileBasedDataReader
from magic_pdf.data.dataset import PymuDocDataset
from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
from magic_pdf.config.enums import SupportedPdfParseMethod
from contextlib import suppress
import torch

def extract_title_from_content(content):
    """
    从PDF内容中提取标题，优先查找以#开头的行
    """
    lines = content.split('\n')
    
    for line in lines[:20]:  # 检查前20行
        line = line.strip()
        if line.startswith('#'):
            # 移除#符号和可能的额外空格
            title = line[1:].strip()
            return title
    
    # 如果没找到#开头的标题，返回文件名（去除扩展名）
    return "Unknown Title"

def process_pdf_with_magic_pdf(pdf_path):
    """
    使用magic_pdf库处理单个PDF文件
    """
    # 定义输出目录
    local_image_dir, local_md_dir = "output/images", "output"
    
    # 创建输出目录
    os.makedirs(local_image_dir, exist_ok=True)
    image_writer, md_writer = FileBasedDataWriter(local_image_dir), FileBasedDataWriter(local_md_dir)
    
    # 读取PDF内容
    reader = FileBasedDataReader("")
    pdf_bytes = reader.read(pdf_path)
    
    # 创建数据集实例并进行分类
    ds = PymuDocDataset(pdf_bytes)
    if ds.classify() == SupportedPdfParseMethod.OCR:
        # 如果需要OCR处理（扫描版PDF），强制使用GPU
        infer_result = ds.apply(doc_analyze, ocr=True)
        pipe_result = infer_result.pipe_ocr_mode(image_writer)
    else:
        # 文字版PDF处理，强制使用GPU
        infer_result = ds.apply(doc_analyze, ocr=False)
        pipe_result = infer_result.pipe_txt_mode(image_writer)
    
    # 输出Markdown文件
    md_filename = f"{os.path.splitext(pdf_path)[0]}.md"
    pipe_result.dump_md(md_writer, md_filename, local_image_dir)
    md_file_path = os.path.join(local_md_dir, md_filename)
    
    # 读取生成的Markdown内容
    with open(md_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 清理临时文件
    with suppress(Exception):
        os.remove(md_file_path)  # 删除Markdown文件
    
    return content

def check_gpu_availability():
    """
    检查GPU可用性并显示相关信息
    """
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("警告: 没有检测到可用的GPU，将使用CPU运行")
    return torch.cuda.is_available()

def process_pdfs_to_json_with_magic_pdf(folder_path, output_json_path):
    """
    使用magic_pdf处理指定文件夹中的所有PDF文件，将其内容提取并保存为JSON格式
    """
    # 检查GPU可用性
    gpu_available = check_gpu_availability()
    
    if gpu_available:
        print("正在使用GPU运行...")
        # 设置PyTorch使用GPU
        device = torch.device('cuda')
        print(f"当前设备: {device}")
    else:
        print("正在使用CPU运行...")
        device = torch.device('cpu')
    
    results = []
    
    # 遍历文件夹中的所有PDF文件
    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.pdf'):
            pdf_path = os.path.join(folder_path, filename)
            
            try:
                print(f"正在处理: {filename}")
                
                # 使用magic_pdf处理PDF
                content = process_pdf_with_magic_pdf(pdf_path)
                
                # 提取标题
                title = extract_title_from_content(content)
                
                # 创建JSON对象
                result = {
                    "title": title,
                    "context": content
                }
                
                results.append(result)
                print(f"已处理: {filename}，标题: {title[:50]}...")
                
            except Exception as e:
                print(f"处理文件 {filename} 时出错: {e}")
    
    # 保存为JSON文件
    with open(output_json_path, 'a', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"处理完成！共处理 {len(results)} 个PDF文件")
    print(f"结果已保存到: {output_json_path}")
    
    return results

# 使用示例
if __name__ == "__main__":
    # 指定输入文件夹路径和输出JSON文件路径
    input_folder = "/home/shansong/pac/4类pdf添加"
    #input_folder = "/home/shansong/pac/test"
    output_json = "/mnt/data/shansong/ADC/ADC/4pdf.json"
    
    # 检查输入文件夹是否存在
    if not os.path.exists(input_folder):
        print(f"错误：文件夹 {input_folder} 不存在")
    else:
        # 处理PDF文件
        processed_data = process_pdfs_to_json_with_magic_pdf(input_folder, output_json)
        
        # 显示前几个结果作为示例
        print("\n前3个PDF的处理结果:")
        for i, item in enumerate(processed_data[:3]):
            print(f"\nPDF {i+1}:")
            print(f"  标题: {item['title'][:100]}...")  # 只显示前100个字符
            print(f"  内容长度: {len(item['context'])} 字符")



