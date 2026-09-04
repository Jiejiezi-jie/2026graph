# Adaptive GraphRAG 阶段一实验记录

## 1. 实验目标

复用 Adaptive-RAG 的核心思想：先让多个候选策略在训练问题上运行，根据实际结果自动产生查询—策略银标，再训练一个只读取问题文本的轻量路由器。我们的动作空间改为：

1. Vector proxy：直接文本块检索；
2. Light graph proxy：文本块图上的一跳扩散检索；
3. Path graph proxy：查询分解、种子定位与路径流检索。

本轮不加入不确定性检测。

## 2. 重要实验边界

当前执行环境没有LLM API凭据、Ollama、GPU、PyTorch或Transformers。因此，本轮完成的是**真实benchmark上的检索层协议验证**，不是官方LightRAG/PathRAG端到端复现：

- 数据、问题、答案、证据来自官方GraphRAG-Bench Medical；
- 路由器经过真实训练和留出测试；
- 后端是确定性的离线代理，不使用LLM抽取实体与关系；
- Evidence Coverage是词项级确定性代理指标，不是官方LLM裁判指标；
- 抽取式`answer_proxy`只证明流程能够输出答案，不能作为医疗问答效果结论。

正式答辩前接入官方后端后，需要重新生成银标和实验表，不能将本表写成LightRAG、PathRAG论文复现结果。

## 3. 数据与设置

| 项目 | 数值 |
|---|---:|
| Medical语料规模 | 174,610词 |
| 文本块 | 1,255 |
| Fact Retrieval | 1,098题 |
| Complex Reasoning | 509题 |
| 总问题数 | 1,607题 |
| 训练集 | 963题 |
| 验证集 | 322题 |
| 测试集 | 322题 |
| 固定随机种子 | 42 |

切分参数为180词/块、40词重叠。三种方法分别返回3、5、7个文本块，表示逐步增加的检索预算。

## 4. 银标构造

每个问题分别运行三种后端。若某个方法的Evidence Coverage满足：

\[
Q(r,q) \geq \max_{r'}Q(r',q)-0.02
\]

则视为近似最优；在候选方法中按照Vector、Light、Path的顺序选择成本最低者。

得到的1607个银标：

| 路由 | 数量 | 比例 |
|---|---:|---:|
| Vector | 870 | 54.14% |
| Light proxy | 330 | 20.54% |
| Path proxy | 407 | 25.33% |

标签不是简单由题型决定：事实题也可能需要图检索，复杂题也可能被Vector检索充分覆盖。

## 5. 路由器

验证集选择出的模型为：

- Word TF-IDF + Character TF-IDF；
- Logistic Regression；
- `class_weight=balanced`；
- `C=0.5`。

测试集表现：

| 指标 | 轻量路由器 | 只预测多数类 |
|---|---:|---:|
| Accuracy | 50.31% | 54.04% |
| Macro-F1 | 44.94% | 23.39% |

路由器的Accuracy没有超过多数类基线，但Macro-F1明显更高，说明它能够识别Light和Path样本，而不是只预测Vector。该结果与Adaptive-RAG论文中“复杂度分类器仍是主要瓶颈”的观察一致。

将题型先验重复加入训练集并没有改善验证集Macro-F1：

| 训练方式 | 验证Accuracy | 验证Macro-F1 |
|---|---:|---:|
| 仅银标 | 48.45% | 42.63% |
| 银标+1份题型先验 | 47.52% | 38.87% |
| 银标+2份题型先验 | 48.76% | 39.28% |

因此主结果保留“仅银标”模型，没有根据测试结果反向选择模型。

## 6. 留出测试集结果

| 策略 | Evidence Coverage | Evidence Hit Rate | 平均上下文词数 | 平均检索延迟/ms |
|---|---:|---:|---:|---:|
| 固定Vector | 0.7333 | 0.5563 | 536.9 | 1.84 |
| 固定Light proxy | 0.7681 | 0.6140 | 894.8 | 2.50 |
| 固定Path proxy | 0.8109 | 0.6914 | 1252.8 | 4.19 |
| Adaptive router | 0.7746 | 0.6262 | 811.2 | 2.71 |
| Oracle router | 0.8113 | 0.6922 | 792.7 | 2.64 |

Adaptive router在322道测试题中选择：

- Vector：162题；
- Light proxy：73题；
- Path proxy：87题。

### 与固定Path方案比较

- 上下文减少35.25%；
- 离线检索延迟减少35.33%；
- Evidence Coverage绝对下降0.0362，相对下降4.47%；
- Evidence Hit Rate绝对下降0.0653。

### 与固定Light方案比较

- 上下文减少9.35%；
- Evidence Coverage绝对提高0.0066；
- Evidence Hit Rate绝对提高0.0122；
- 因部分问题调用Path，离线延迟增加8.46%。

### Oracle结果说明什么

Oracle以792.7个平均上下文词获得0.8113覆盖率，几乎以自适应路由的成本达到固定Path的质量。这说明动作空间与银标协议具有明显的路由潜力；当前Adaptive与Oracle之间的差距主要来自分类器，而不是三后端不存在互补性。

## 7. 不同题型结果

| 题型 | Vector | Light proxy | Path proxy | Adaptive |
|---|---:|---:|---:|---:|
| Fact Retrieval | 0.7711 | 0.8018 | 0.8468 | 0.8084 |
| Complex Reasoning | 0.6516 | 0.6953 | 0.7334 | 0.7017 |

表中数值为Evidence Coverage。复杂推理对所有方法都更难；增加图和路径预算能够提高覆盖率，但成本也同步增加，正好构成Adaptive-RAG式路由的实验动机。

## 8. 代表案例

### Vector足够

问题：`What is the most common type of skin cancer?`

- 三种方法Coverage均为1.0；
- Vector只返回约538词；
- 银标和路由器均选择Vector。

### Light路线近似最优

问题：`What are common symptoms of CSCC?`

- Vector Coverage：0.417；
- Light Coverage：0.639；
- Path Coverage：0.639；
- Light与Path质量相同，因此银标选择更便宜的Light，路由器预测正确。

### Path路线明显更好

问题：`Which diagnostic methods are used for BCC?`

- Vector Coverage：0.242；
- Light Coverage：0.242；
- Path Coverage：0.483；
- 银标和路由器均选择Path。

### 路由失败案例

问题：`Which combination of risk factors should prompt earlier and more frequent skin exams for BCC?`

- Vector Coverage：0.401；
- Light Coverage：0.556；
- Path Coverage：0.738；
- 银标为Path，但路由器预测为Light。

该案例说明，仅依赖浅层词项特征难以稳定识别“需要组合多个证据”的问题。下一版更换为T5-Small或其他预训练编码器是合理方向。

## 9. 当前结论

本轮已经验证：

1. GraphRAG-Bench Medical可以直接用于Adaptive-RAG式银标构造；
2. 三种不同成本的检索策略具有互补性；
3. 自适应路由能在Path方案和Light方案之间获得实际质量—成本折中；
4. 简单TF-IDF分类器能跑通架构，但能力不足，是下一轮首先应替换的模块；
5. 题型先验没有在验证集带来收益，因此不应为了提高测试Accuracy强行加入。

## 10. 下一步

1. 接入官方LightRAG `naive`和`hybrid`模式；
2. 接入官方PathRAG；
3. 使用同一LLM、embedding、温度和上下文预算重新生成答案；
4. 运行GraphRAG-Bench官方生成与检索评价；
5. 在新的银标上训练T5-Small/T5-Base路由器；
6. 最后再开发网页，不在模型未稳定前做界面包装。
