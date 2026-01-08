# 理解 F1 Score - 通俗教程

## 目录
1. [基础概念](#基础概念)
2. [TP/FP/FN 详解](#tpfpfn-详解)
3. [Precision 和 Recall](#precision-和-recall)
4. [F1 Score 综合评分](#f1-score-综合评分)
5. [实际例子](#实际例子)
6. [为什么没有 TN](#为什么没有-tn)
7. [F-beta Score 可调节版本](#f-beta-score-可调节版本)

---

## 基础概念

### 在 Answer Correctness 中的术语对应

| 术语 | 含义 | 在我们代码中 |
|------|------|--------------|
| **预测** | 系统给出的答案 | `answer` 字段 |
| **实际** | 标准答案 | `ground_truth_answer` 字段 |
| **陈述** | 分解后的原子信息 | 通过 LLM 分解得到 |

---

## TP/FP/FN 详解

### 具体例子

**问题**: "FHA 贷款有什么要求？"

**Ground Truth（标准答案）**:
- "FHA 需要 3.5% 首付"
- "FHA 需要 580 信用分数"  
- "FHA 需要保险"

**Answer（系统给出的答案）**:
- "FHA 需要 3.5% 首付" ✓
- "FHA 需要 580 信用分数" ✓
- "FHA 需要 5% 首付" ✗ (这是错的！)

### TP (True Positive) - 真阳性

**定义**: Answer 里说的，而且 Ground Truth 也支持的

**理解**: 你答对了的内容

**例子**:
- "FHA 需要 3.5% 首付" ✓ (答案里有，真实答案也确认了)
- "FHA 需要 580 信用分数" ✓ (答案里有，真实答案也确认了)

**TP = 2**

### FP (False Positive) - 假阳性  

**定义**: Answer 里说的，但 Ground Truth **不支持**的（错误信息）

**理解**: 你答错了的内容，或者瞎编的内容

**例子**:
- "FHA 需要 5% 首付" ✗ (答案里有，但真实答案说是 3.5%，这是错的！)

**FP = 1**

### FN (False Negative) - 假阴性

**定义**: Ground Truth 里有的，但 Answer **没说**的（遗漏信息）

**理解**: 你该说但没说的内容

**例子**:
- "FHA 需要保险" ⚠️ (真实答案里有，但答案里没提到，漏了！)

**FN = 1**

---

## Precision 和 Recall

### Precision（精确率）- "你说的有多少是对的"

```
公式: Precision = TP / (TP + FP)
含义: 答对的 / 答案里所有的
```

**通俗理解**: 你的答案里的"质量"如何

**例子计算**:
```
Precision = 2 / (2 + 1) = 2/3 = 0.67 = 67%
```

**解读**: 你的答案里有 3 条信息，但只有 2 条是对的，1 条是错的
→ 精确率 = 67%（说明你答案的"质量"一般）

### Recall（召回率）- "该说的你说了多少"

```
公式: Recall = TP / (TP + FN)
含义: 答对的 / 真实答案里所有的
```

**通俗理解**: 你的答案的"完整性"如何

**例子计算**:
```
Recall = 2 / (2 + 1) = 2/3 = 0.67 = 67%
```

**解读**: 真实答案有 3 条信息，你只答对了 2 条，漏了 1 条
→ 召回率 = 67%（说明你答案的"完整性"一般）

---

## F1 Score 综合评分

### 公式

```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

这是 **调和平均数**（Harmonic Mean），不是算术平均数！

### 为什么用调和平均？

调和平均数对较小的值更敏感，能更好地惩罚极端不平衡的情况。

**例子**:
- Precision = 1.0, Recall = 0.1
- 算术平均 = (1.0 + 0.1) / 2 = **0.55** (看起来还行？)
- 调和平均 (F1) = 2 × 1.0 × 0.1 / (1.0 + 0.1) = **0.18** (更真实！)

### 例子计算

```
F1 = 2 × 0.67 × 0.67 / (0.67 + 0.67)
   = 2 × 0.4489 / 1.34
   = 0.67
```

**通俗理解**: 既不够准确（有错误），也不够完整（有遗漏），综合得分 67%

---

## 实际例子

### 例子 1: 完美答案

**Ground Truth**: "FHA 需要 3.5% 首付"
**Answer**: "FHA 需要 3.5% 首付"

```
TP = 1, FP = 0, FN = 0

Precision = 1/(1+0) = 1.0  ✓ 完全正确
Recall = 1/(1+0) = 1.0     ✓ 完全完整
F1 = 1.0                   ✓ 完美！
```

### 例子 2: 高 Precision，低 Recall

**Ground Truth**: "太阳由氢聚变、释放能量、提供光和热"
**Answer**: "太阳由氢聚变"

```
TP = 1 (氢聚变说对了)
FP = 0 (没有错误信息)
FN = 2 (漏了能量和光热)

Precision = 1/(1+0) = 1.0  ✓ 说的都对！
Recall = 1/(1+2) = 0.33    ✗ 但说的太少了！
F1 = 0.50                  ⚠️ 综合评分不高
```

**问题**: 虽然没错，但太简略了

### 例子 3: 低 Precision，高 Recall（但实际不可能这样）

**Ground Truth**: "FHA 需要 3.5% 首付"
**Answer**: "FHA 需要 3.5% 首付，还需要 10% 首付，以及 20% 首付"

```
TP = 1 (3.5% 说对了)
FP = 2 (10% 和 20% 是错的)
FN = 0 (没有遗漏)

Precision = 1/(1+2) = 0.33 ✗ 3 条里只有 1 条对，乱说！
Recall = 1/(1+0) = 1.0     ✓ 但该说的都说了
F1 = 0.50                  ⚠️ 综合评分不高
```

**问题**: 虽然完整，但有很多错误

### 例子 4: 部分正确，部分遗漏

**Ground Truth**: "FHA 需要 3.5% 首付、580 信用分数、保险"
**Answer**: "FHA 需要 3.5% 首付、580 信用分数"

```
TP = 2 (首付✓、信用分数✓)
FP = 0 (没有错误信息)
FN = 1 (漏了保险)

Precision = 2/(2+0) = 1.0   ✓ 说的都对（100%准确）
Recall = 2/(2+1) = 0.67     ⚠️ 但不完整（只说了67%）
F1 = 2 × 1.0 × 0.67 / 1.67 = 0.80  ⚠️ 综合得分80分
```

**问题**: 质量很好，但不够完整

### 例子 5: 完全错误

**Ground Truth**: "太阳由核聚变驱动"
**Answer**: "太阳由核裂变驱动"

```
TP = 0 (说的都不对)
FP = 1 (核裂变是错的)
FN = 1 (漏了核聚变)

Precision = 0/(0+1) = 0.0   ✗ 完全错误
Recall = 0/(0+1) = 0.0      ✗ 完全遗漏
F1 = 0.0                    ✗ 零分！
```

---

## 为什么没有 TN

### TN (True Negative) 是什么？

**TN (True Negative)** = 预测为负，实际也是负

### 在传统分类问题中

**例子: 垃圾邮件分类**

| 实际\预测 | 预测是垃圾 | 预测是正常 |
|-----------|-----------|-----------|
| 实际是垃圾 | TP | FN |
| 实际是正常 | FP | **TN** |

- TP: 正确识别为垃圾邮件 ✓
- FP: 误判为垃圾邮件（正常邮件被当垃圾）✗
- FN: 漏掉的垃圾邮件（垃圾邮件没识别出来）⚠️
- **TN**: 正确识别为正常邮件 ✓ ← 这个很重要！

在这个场景中，TN 和 TP 一样重要，因为你有明确的"负类"。

### 在我们的场景中

我们评估的是**答案中说了什么**：

**Ground Truth**: "FHA 需要 3.5% 首付、580 信用分数、保险"
**Answer**: "FHA 需要 3.5% 首付"

TN 应该是什么？**"没说错的、也不该说的陈述"**？

比如：
- "FHA 不需要 10% 首付" ← Answer 没说，Ground Truth 也没说
- "FHA 不需要飞行执照" ← Answer 没说，Ground Truth 也没说
- "FHA 不需要外星人批准" ← Answer 没说，Ground Truth 也没说
- ... 这样的"没说的"有**无穷多个**！

### 结论

在**开放式文本生成**的场景中：
- **TN 是无限大且无意义的**
- 我们只关心：
  - 答案说了什么（TP）
  - 说错了什么（FP）
  - 该说但没说什么（FN）
- 不关心"无穷多个没说也不该说的事情"（TN）

---

## F-beta Score 可调节版本

### 公式

```
F_beta = (1 + β²) × (Precision × Recall) / (β² × Precision + Recall)
```

### Beta 参数的作用

#### β = 1 (标准 F1)
平等对待 Precision 和 Recall

#### β > 1 (如 β=2)
**更重视 Recall（召回率/完整性）**

**适用场景**: 宁可多找，不能漏找
- 医疗诊断：宁可误诊，不能漏诊
- 搜索引擎：宁可多返回，不能漏掉重要结果
- 问答系统：宁可答案长一点，不能遗漏关键信息

#### β < 1 (如 β=0.5)
**更重视 Precision（精确率/准确性）**

**适用场景**: 宁可少找，不能找错
- 垃圾邮件过滤：宁可放过垃圾邮件，不能误删重要邮件
- 推荐系统：宁可少推荐，不能推荐不相关的
- 问答系统：宁可答案简短，不能包含错误信息

### 实际例子对比

假设 Precision = 0.8, Recall = 0.6：

```
F1 (β=1):    2 × 0.8 × 0.6 / (0.8 + 0.6) = 0.686
F2 (β=2):    5 × 0.8 × 0.6 / (4×0.8 + 0.6) = 0.652  (更接近 Recall)
F0.5 (β=0.5): 1.25 × 0.8 × 0.6 / (0.25×0.8 + 0.6) = 0.732  (更接近 Precision)
```

### 在代码中使用

```python
# 当前实现（F1）
f1_score = 2 * (precision * recall) / (precision + recall)

# F-beta 实现
beta = 2.0  # 可调节，更重视完整性
beta_squared = beta ** 2
f_beta = (1 + beta_squared) * (precision * recall) / (beta_squared * precision + recall)
```

### 何时调节 Beta？

**需要更完整的答案时 (β>1)**:
```python
metric = AnswerCorrectness(
    llm_uri=llm_uri,
    api_key=api_key,
    beta=2.0  # 更重视召回率（完整性）
)
```

**需要更准确的答案时 (β<1)**:
```python
metric = AnswerCorrectness(
    llm_uri=llm_uri,
    api_key=api_key,
    beta=0.5  # 更重视精确率（准确性）
)
```

---

## 总结

### 核心理解

1. **TP/FP/FN** 是从答案的陈述角度分类
   - TP = 答对的
   - FP = 答错的
   - FN = 该答但没答的

2. **Precision** = 你说的有多少是对的（质量）
3. **Recall** = 该说的你说了多少（完整性）
4. **F1** = 综合质量和完整性的分数
5. **F-beta** = 可以调节重视 Precision 还是 Recall

### 记忆技巧

想象你是老师批改试卷：
- **Precision**: 学生写的答案中，有多少是对的？
- **Recall**: 标准答案中的要点，学生答对了多少？
- **F1**: 给个综合分数，既要对，也要全

### 何时用什么？

- **只看 Precision**: 你在乎"别说错"，不在乎"说不全"
- **只看 Recall**: 你在乎"别漏掉"，不在乎"说多了"
- **看 F1**: 你想要平衡，既要对，又要全
- **调节 Beta**: 你想偏向某一边，但还是要综合考虑

---

## 附录：代码中的实际计算

### 来自 test_answer_correctness.py 的真实输出

#### Case 1: 完美答案
```
Question: What is the minimum down payment for FHA loans?
Ground Truth: FHA loans require a minimum down payment of 3.5%.
Answer: FHA loans require a minimum down payment of 3.5%.

TP=1, FP=0, FN=0
Precision: 1.00
Recall: 1.00
F1 Score: 1.00
```

#### Case 3: 部分遗漏
```
Question: What are the requirements for FHA loans?
Ground Truth: FHA loans require a minimum down payment of 3.5%, 
              a credit score of at least 580, and mortgage insurance premiums.
Answer: FHA loans require a minimum down payment of 3.5% 
        and a credit score of at least 580.

TP=2 (首付✓、信用分数✓)
FP=0 (没有错误)
FN=1 (遗漏了保险)

Precision: 1.00 (说的都对)
Recall: 0.67 (但不完整)
F1 Score: 0.80
```

#### Case 4: 完全错误
```
Question: What is the minimum down payment for FHA loans?
Ground Truth: FHA loans require a minimum down payment of 3.5%.
Answer: FHA loans require a minimum down payment of 5% or 10% 
        depending on your credit score.

TP=0 (没有答对的)
FP=2 (5%和10%都是错的)
FN=1 (漏了正确的3.5%)

Precision: 0.00 (完全错误)
Recall: 0.00 (完全遗漏)
F1 Score: 0.00
```
