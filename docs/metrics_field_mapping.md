# RAG 评估指标与字段关系映射

本文档说明每个评估指标评估的是 `EvalCase` 中哪些字段之间的关系。

## EvalCase 字段概览

```python
class EvalCase:
    # 问题
    question: str
    
    # Ground Truth（标准答案和参考上下文）
    ground_truth_answer: str          # 标准答案
    ground_truth_contexts: list[str]  # 标准参考上下文
    
    # RAG 系统输出（评估时填充）
    answer: str | None                # RAG 系统生成的答案
    retrieved_contexts: list[str] | None  # RAG 系统检索到的上下文
    
    # 溯源信息
    source_units: list[dict]          # 来源 units
    persona: dict | None              # 使用的 persona
    generation_params: dict           # 生成参数
```

---

## 核心评估指标

### 1. Faithfulness（忠实度）

**评估关系**: `answer` ↔ `retrieved_contexts`

**评估目标**: RAG 系统生成的答案是否忠实于检索到的上下文

**具体做法**:
1. 将 `answer` 分解为原子陈述句（atomic statements）
2. 判断每个陈述句是否能从 `retrieved_contexts` 中找到支持
3. 计算：忠实度 = 被支持的陈述数 / 总陈述数

**关键特点**:
- ✅ 检测幻觉（hallucination）：答案是否编造了不在上下文中的信息
- ❌ 不关心 `retrieved_contexts` 本身是否正确
- ❌ 不关心 `answer` 是否回答了 `question`

**示例**:
```python
EvalCase(
    question="什么是 FHA 贷款的最低首付？",
    answer="FHA 贷款最低首付为 3.5%",  # 忠实于 context
    retrieved_contexts=[
        "FHA 贷款允许最低 3.5% 的首付..."
    ]
)
# Faithfulness: 1.0（完全忠实）

EvalCase(
    question="什么是 FHA 贷款的最低首付？",
    answer="FHA 贷款最低首付为 3.5%，且联邦政府强制要求不得超过此比例",  # 后半句幻觉
    retrieved_contexts=[
        "FHA 贷款允许最低 3.5% 的首付..."
    ]
)
# Faithfulness: 0.5（部分幻觉）
```

---

### 2. Context Relevance（上下文相关性）

**评估关系**: `retrieved_contexts` ↔ `question`

**评估目标**: 检索到的上下文是否与问题相关

**具体做法**:
1. 评估每个 `retrieved_contexts` 中的片段是否与 `question` 相关
2. 计算：相关性 = 相关片段数 / 总片段数

**关键特点**:
- ✅ 检测检索质量：是否检索到了相关内容
- ✅ 检测噪声：是否检索到了无关内容
- ❌ 不关心检索到的内容是否包含正确答案

**示例**:
```python
EvalCase(
    question="FHA 贷款的最低首付是多少？",
    retrieved_contexts=[
        "FHA 贷款允许最低 3.5% 的首付...",  # 相关
        "ARMs 通常以较低利率开始...",      # 不相关（无关检索）
    ]
)
# Context Relevance: 0.5（一半相关）
```

---

### 3. Context Recall（上下文召回率）

**评估关系**: `ground_truth_answer` ↔ `retrieved_contexts`

**评估目标**: 检索到的上下文是否包含回答问题所需的信息

**具体做法**:
1. 将 `ground_truth_answer` 分解为原子陈述句
2. 判断每个陈述句是否能从 `retrieved_contexts` 中找到支持
3. 计算：召回率 = 被支持的陈述数 / 总陈述数

**关键特点**:
- ✅ 检测检索完整性：是否检索到了足够的信息
- ✅ 评估检索系统的召回能力
- ⚠️ 需要 `ground_truth_answer` 才能评估

**示例**:
```python
EvalCase(
    question="FHA 贷款有哪些要求？",
    ground_truth_answer="FHA 贷款最低首付 3.5%，信用分数至少 580",
    retrieved_contexts=[
        "FHA 贷款允许最低 3.5% 的首付...",  # 包含首付信息
        # 缺少信用分数信息
    ]
)
# Context Recall: 0.5（只召回了一半所需信息）
```

---

### 4. Context Precision（上下文精确度）

**评估关系**: `ground_truth_contexts` ↔ `retrieved_contexts`

**评估目标**: 检索到的上下文是否精确（相关的排在前面）

**具体做法**:
1. 评估 `retrieved_contexts` 中每个片段是否在 `ground_truth_contexts` 中
2. 计算排序质量（相关的应该排在前面）
3. 使用 Precision@K 或类似指标

**关键特点**:
- ✅ 评估检索排序质量
- ✅ 检测是否检索到了无关内容
- ⚠️ 需要 `ground_truth_contexts` 才能评估

**示例**:
```python
EvalCase(
    question="FHA 贷款的最低首付是多少？",
    ground_truth_contexts=[
        "FHA 贷款允许最低 3.5% 的首付..."
    ],
    retrieved_contexts=[
        "ARMs 通常以较低利率开始...",      # 不相关（排在前面，差）
        "FHA 贷款允许最低 3.5% 的首付...",  # 相关（排在后面，差）
    ]
)
# Context Precision: 低（相关内容排序靠后）
```

---

### 5. Answer Relevancy（答案相关性）

**评估关系**: `answer` ↔ `question`

**评估目标**: 答案是否回答了问题

**具体做法**:
1. 判断 `answer` 是否直接回答了 `question`
2. 检测答案是否包含无关信息
3. 评估答案的针对性

**关键特点**:
- ✅ 检测答案是否切题
- ✅ 检测答案是否包含过多无关信息
- ❌ 不关心答案是否正确

**示例**:
```python
EvalCase(
    question="FHA 贷款的最低首付是多少？",
    answer="FHA 贷款允许最低 3.5% 的首付",  # 直接回答
)
# Answer Relevancy: 1.0

EvalCase(
    question="FHA 贷款的最低首付是多少？",
    answer="FHA 贷款有很多优势，包括较低的首付要求、灵活的信用分数标准...",  # 啰嗦，不直接
)
# Answer Relevancy: 0.6（包含答案但不直接）
```

---

### 6. Answer Correctness（答案正确性）

**评估关系**: `answer` ↔ `ground_truth_answer`

**评估目标**: 答案是否正确

**具体做法**:
1. 比较 `answer` 和 `ground_truth_answer` 的语义相似度
2. 评估事实一致性
3. 计算综合正确性得分

**关键特点**:
- ✅ 评估答案的正确性
- ✅ 端到端评估 RAG 系统的最终效果
- ⚠️ 需要 `ground_truth_answer` 才能评估

**示例**:
```python
EvalCase(
    question="FHA 贷款的最低首付是多少？",
    ground_truth_answer="FHA 贷款允许最低 3.5% 的首付",
    answer="FHA 贷款最低首付为 3.5%",  # 语义一致
)
# Answer Correctness: 1.0

EvalCase(
    question="FHA 贷款的最低首付是多少？",
    ground_truth_answer="FHA 贷款允许最低 3.5% 的首付",
    answer="FHA 贷款最低首付为 5%",  # 错误
)
# Answer Correctness: 0.0
```

---

## 指标关系总结表

| 指标 | 评估字段关系 | 需要 Ground Truth | 评估维度 | 检测问题 |
|-----|------------|------------------|---------|---------|
| **Faithfulness** | `answer` ↔ `retrieved_contexts` | ❌ | 忠实度 | 幻觉 |
| **Context Relevance** | `retrieved_contexts` ↔ `question` | ❌ | 检索相关性 | 检索噪声 |
| **Context Recall** | `ground_truth_answer` ↔ `retrieved_contexts` | ✅ | 检索完整性 | 检索遗漏 |
| **Context Precision** | `ground_truth_contexts` ↔ `retrieved_contexts` | ✅ | 检索精确度 | 检索排序 |
| **Answer Relevancy** | `answer` ↔ `question` | ❌ | 答案针对性 | 答非所问 |
| **Answer Correctness** | `answer` ↔ `ground_truth_answer` | ✅ | 答案正确性 | 错误答案 |

---

## 评估场景

### 场景 1: 无 Ground Truth 评估（生产环境）

**可用指标**:
- Faithfulness（检测幻觉）
- Context Relevance（检测检索质量）
- Answer Relevancy（检测答案相关性）

**不可用指标**:
- Context Recall（需要 `ground_truth_answer`）
- Context Precision（需要 `ground_truth_contexts`）
- Answer Correctness（需要 `ground_truth_answer`）

### 场景 2: 有 Ground Truth 评估（测试评估集）

**可用指标**: 全部 6 个指标

**典型流程**:
1. 使用 `synthetic_data` 模块生成评估集（包含 ground truth）
2. 使用 RAG 系统填充 `answer` 和 `retrieved_contexts`
3. 运行所有指标评估
4. 生成综合评估报告

---

## 字段填充时机

```python
# 生成时（synthetic_data 模块）
EvalCase(
    question="...",                    # ✅ 生成
    ground_truth_answer="...",         # ✅ 生成
    ground_truth_contexts=["..."],     # ✅ 生成
    answer=None,                       # ❌ 待填充
    retrieved_contexts=None,           # ❌ 待填充
)

# 评估前（RAG 系统）
case.answer = rag_system.generate(case.question)
case.retrieved_contexts = rag_system.retrieve(case.question)

# 评估后（evaluation 模块）
case.results["faithfulness"] = EvalResult(...)
case.results["context_relevance"] = EvalResult(...)
```

---

## 注意事项

1. **Faithfulness 不保证 retrieved_contexts 正确**
   - 即使 context 是错的，只要 answer 忠实于 context，分数就高
   - 需要配合 Context Recall 等指标一起使用

2. **部分指标需要 Ground Truth**
   - Context Recall、Context Precision、Answer Correctness 需要 ground truth
   - 生产环境无法使用这些指标

3. **指标互补性**
   - 单个指标无法全面评估 RAG 系统
   - 需要组合多个指标从不同维度评估
   - 建议至少使用：Faithfulness + Context Relevance + Answer Relevancy

4. **字段命名区分**
   - `answer`: RAG 系统生成的答案（待评估）
   - `ground_truth_answer`: 标准答案（用于对比）
   - `retrieved_contexts`: RAG 检索的上下文（待评估）
   - `ground_truth_contexts`: 标准上下文（用于对比）
