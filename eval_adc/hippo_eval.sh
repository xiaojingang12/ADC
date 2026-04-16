#!/bin/bash

# ==================== 配置区域 ====================
# 基础路径
HIPPO_BASE_DIR="/mnt/data/shansong/GraphRAG/2_data_wor/Collected/RAPTOR"
BASE_QA_ROOT="/mnt/data/shansong/ADC/ADC/is_data_copy"
SAVE_ROOT="/mnt/data/xinyang/graphrag_benchmark/eval_results/RAPTOR"
PYTHON_SCRIPT="./eval.py"

# 日志配置
LOG_DIR="/home/xinyang/graphrag_benchmark/logs/RAPTOR"
BATCH_LOG_FILE="$LOG_DIR/RAPTOR_evaluation_$(date +%Y%m%d_%H%M%S).log"

# 目标文件名
TARGET_FILE="results.json"
QA_TYPE="middle_QA"

# ==================== 颜色输出函数 ====================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m'

log_info()    { echo -e "${BLUE}[INFO]${NC} $1" | tee -a "$BATCH_LOG_FILE"; }
log_success() { echo -e "${GREEN}[✓]${NC} $1" | tee -a "$BATCH_LOG_FILE"; }
log_warning() { echo -e "${YELLOW}[⚠]${NC} $1" | tee -a "$BATCH_LOG_FILE"; }
log_error()   { echo -e "${RED}[✗]${NC} $1" | tee -a "$BATCH_LOG_FILE"; }
log_step()    { echo -e "${CYAN}[STEP]${NC} $1" | tee -a "$BATCH_LOG_FILE"; }
log_section() { echo -e "${MAGENTA}[SECTION]${NC} $1" | tee -a "$BATCH_LOG_FILE"; }

# ==================== 初始化检查 ====================
mkdir -p "$LOG_DIR" "$SAVE_ROOT"

log_info "=========================================="
log_info "批量评估任务启动"
log_info "时间: $(date '+%Y-%m-%d %H:%M:%S')"
log_info "=========================================="

# 检查必要文件和目录
if [ ! -f "$PYTHON_SCRIPT" ]; then
    log_error "评估脚本不存在: $PYTHON_SCRIPT"
    exit 1
fi
if [ ! -d "$HIPPO_BASE_DIR" ]; then
    log_error "HippoRAG基础目录不存在: $HIPPO_BASE_DIR"
    exit 1
fi
if [ ! -d "$BASE_QA_ROOT" ]; then
    log_error "标准答案根目录不存在: $BASE_QA_ROOT"
    exit 1
fi

log_success "预检查通过"
log_info "配置信息:"
log_info "  HippoRAG目录: $HIPPO_BASE_DIR"
log_info "  标准答案根目录: $BASE_QA_ROOT"
log_info "  输出根目录: $SAVE_ROOT"
log_info "  目标文件: $TARGET_FILE"
log_info "  问答类型: $QA_TYPE"

# ==================== 主循环 ====================
TOTAL_QUESTION_TYPES=0
TOTAL_DIRS=0
PROCESSED_DIRS=0
SKIPPED_DIRS=0
SUCCESS_FILES=0
FAIL_FILES=0

BATCH_START_TIME=$(date +%s)

# 第一层循环: 遍历 QUESTION_TYPE (例如: 1_data, 2_data, 5_data, ...)
for question_type_dir in "$HIPPO_BASE_DIR"/*; do
    # 只处理目录
    if [ ! -d "$question_type_dir" ]; then
        continue
    fi
    
    QUESTION_TYPE=$(basename "$question_type_dir")
    
    # 检查是否为数据目录 (可选: 只处理特定模式的目录)
    if [[ ! "$QUESTION_TYPE" =~ ^[0-9]+_data$ ]]; then
        log_warning "跳过非数据目录: $QUESTION_TYPE"
        continue
    fi
    
    ((TOTAL_QUESTION_TYPES++))
    
    log_section "=========================================="
    log_section "处理问题类型 [$TOTAL_QUESTION_TYPES]: $QUESTION_TYPE"
    log_section "=========================================="
    
    # 构建该类型下的 simple_QA 路径
    QA_TYPE_DIR="$question_type_dir/$QA_TYPE"
    
    if [ ! -d "$QA_TYPE_DIR" ]; then
        log_warning "$QA_TYPE 目录不存在: $QA_TYPE_DIR"
        continue
    fi
    
    # 对应的标准答案基础目录
    BASE_QA_BASE_DIR="$BASE_QA_ROOT/$QUESTION_TYPE/$QA_TYPE"
    
    if [ ! -d "$BASE_QA_BASE_DIR" ]; then
        log_warning "标准答案目录不存在: $BASE_QA_BASE_DIR"
        continue
    fi
    
    # 对应的输出基础目录
    SAVE_BASE_DIR="$SAVE_ROOT/$QUESTION_TYPE/$QA_TYPE"
    mkdir -p "$SAVE_BASE_DIR"
    
    log_info "  QA类型目录: $QA_TYPE_DIR"
    log_info "  标准答案基础: $BASE_QA_BASE_DIR"
    log_info "  输出基础: $SAVE_BASE_DIR"
    
    # 第二层循环: 遍历 QUESTION_ID (例如: S_00, S_01, M_00, ...)
    for question_id_dir in "$QA_TYPE_DIR"/*; do
        # 只处理目录
        if [ ! -d "$question_id_dir" ]; then
            continue
        fi
        
        ((TOTAL_DIRS++))
        
        QUESTION_ID=$(basename "$question_id_dir")
        
        log_section "  ------------------------------------------"
        log_section "  处理问题 [$TOTAL_DIRS]: $QUESTION_TYPE/$QUESTION_ID"
        log_section "  ------------------------------------------"
        
        # 构建完整路径
        QA_FILE="$question_id_dir/$TARGET_FILE"
        BASE_QA_PATH="$BASE_QA_BASE_DIR/$QUESTION_ID/Question.json"
        SAVE_DIR="$SAVE_BASE_DIR/$QUESTION_ID"
        OUTPUT_FILE="$SAVE_DIR/${TARGET_FILE%.json}_evaluation.json"
        
        # 检查 results.json 是否存在
        if [ ! -f "$QA_FILE" ]; then
            log_warning "    $TARGET_FILE 不存在: $QA_FILE"
            ((SKIPPED_DIRS++))
            continue
        fi
        
        # 检查标准答案文件是否存在
        if [ ! -f "$BASE_QA_PATH" ]; then
            log_warning "    标准答案不存在: $BASE_QA_PATH"
            ((SKIPPED_DIRS++))
            continue
        fi
        
        # 创建输出目录
        mkdir -p "$SAVE_DIR"
        
        log_info "    输入: $QA_FILE"
        log_info "    标准: $BASE_QA_PATH"
        log_info "    输出: $OUTPUT_FILE"
        
        # 检查是否已存在评估结果
        if [ -f "$OUTPUT_FILE" ]; then
            log_warning "    ⊙ 已存在评估结果,跳过"
            ((SKIPPED_DIRS++))
            continue
        fi
        
        ((PROCESSED_DIRS++))
        
        log_step "    开始评估..."
        
        file_log="$LOG_DIR/${QUESTION_TYPE}_${QUESTION_ID}_$(date +%Y%m%d_%H%M%S).log"
        
        # 执行评估
        file_start=$(date +%s)
        
        if python3 "$PYTHON_SCRIPT" \
            --qa_path "$QA_FILE" \
            --base_qa_path "$BASE_QA_PATH" \
            --save_path "$OUTPUT_FILE" 2>&1 | tee -a "$file_log"; then
            
            file_end=$(date +%s)
            elapsed=$((file_end - file_start))
            
            log_success "    ✓ 评估完成 (耗时: ${elapsed}s)"
            ((SUCCESS_FILES++))
        else
            log_error "    ✗ 评估失败"
            log_error "    详细日志: $file_log"
            ((FAIL_FILES++))
        fi
        
    done
    
    log_info "  =========================================="
    
done

# ==================== 统计报告 ====================
BATCH_END_TIME=$(date +%s)
TOTAL_ELAPSED=$((BATCH_END_TIME - BATCH_START_TIME))

log_info ""
log_info "=========================================="
log_info "批量评估完成"
log_info "=========================================="
log_info "问题类型统计:"
log_info "  总类型数: $TOTAL_QUESTION_TYPES"
log_info ""
log_info "目录统计:"
log_info "  总目录数: $TOTAL_DIRS"
log_info "  已处理: $PROCESSED_DIRS"
log_warning "  跳过: $SKIPPED_DIRS"
log_info ""
log_info "文件统计:"
log_success "  成功: $SUCCESS_FILES"
log_error "  失败: $FAIL_FILES"
log_info ""
log_info "  总耗时: $TOTAL_ELAPSED 秒 ($(($TOTAL_ELAPSED / 60)) 分钟)"

if [ "$PROCESSED_DIRS" -gt 0 ] && [ "$TOTAL_ELAPSED" -gt 0 ]; then
    AVG_TIME=$((TOTAL_ELAPSED / PROCESSED_DIRS))
    log_info "  平均每个问题: ${AVG_TIME}s"
fi

log_info "----------------------------------------"
log_info "完整批量日志: $BATCH_LOG_FILE"
log_info "=========================================="

# 根据执行结果设置退出码
if [ "$FAIL_FILES" -gt 0 ]; then
    exit 1
else
    exit 0
fi