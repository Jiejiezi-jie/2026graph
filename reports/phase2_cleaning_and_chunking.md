# 阶段 2：《Dandy Dick》确定性清洗与结构感知切块

本阶段只读取阶段 1 的 `indexing_input.json` 正文，不读取训练题、测试题、
参考答案、evidence 或 evidence_triple。未调用 LLM/Embedding，未建图，
未运行检索，也未安装 LightRAG/PathRAG。

## 输入与清洗

- 输入：`data/processed/phase1/indexing_input.json`
- 输入文件 SHA-256：`526bf7453bd9a59ab4df479b77257427d82d4839a56dcc0811fff787a1968ed6`
- 原始正文：137,923 字符
- 清洗后语义正文：135,316 字符
- 写入结构换行后的 `clean_corpus.txt`：136,720 字符
- 文本保留比例：98.1098%
- 删除区间：2 个，共 2,607 字符

删除类别：

| 原因 | 区间数 | 字符数 |
|---|---:|---:|
| 删除正文标题前的 Project Gutenberg 制作者信息 | 1 | 91 |
| 删除作品结束后的转录说明和 Gutenberg 页脚 | 1 | 2,516 |

清洗器只允许删除标题前制作者信息、作品结束后的转录/Gutenberg 页脚和重复空白；
本次实际删除仅为前两类，原始正文没有需要压缩的连续空白。
本输入未发现 HTML 标签或独立 `[Illustration]` 占位，因此没有伪造零长度删除记录。
弯引号、破折号、拼写、人物口音、历史措辞和斜体下划线均保持原样。

## 戏剧结构

- 标题：*Dandy Dick*
- 作者：Arthur W. Pinero
- 体裁：三幕喜剧／闹剧
- 识别结构：front matter、两段 production history、cast/首演节目、
  Act 1、Act 2、Act 3 Scene 1、Act 3 Scene 2。
- 人物数：11；集体说话标签另有 `ALL`。
- 人物：BLORE、DARBEY、GEORGIANA、HANNAH、HATCHAM、NOAH、SALOME、SHEBA、SIR TRISTRAM、TARVER、THE DEAN
- 演出历史、Royal Court Theatre、首演演员信息和舞台说明均已保留。

说话人不是由代码中的固定列表直接猜测：脚本先屏蔽方括号舞台说明，
从正文抽取稳定大写说话标记，再与演员表内容标记及配置中的最终集合交叉验证。

## Unit 与 Chunk 统计

- Unit：1404
- Unit 类型：`{'cast_entry': 11, 'dialogue': 1320, 'heading': 12, 'prose': 57, 'stage_direction': 4}`
- Chunk：62
- 各结构段 Chunk：`{'act_1': 17, 'act_2': 17, 'act_3_scene_1': 13, 'act_3_scene_2': 11, 'cast': 1, 'front_matter': 1, 'production_history': 2}`
- 词数：min=60，p25=446.0，median=457.0，p75=472.5，max=515。

戏剧中的说话人、舞台动作、出入场和幕/场边界共同承载人物关系与事件链。
因此切块以完整结构单元为最小单位，只复制完整 unit 形成重叠，且禁止跨幕或跨场。

## 三组清洗前后对照

### 1. 标题与制作信息

- 清洗前：`Produced by Paul Haxo from page images generously made available by the Internet Archive. DANDY DICK A PLAY IN THREE ACTS By ARTHUR W. PINERO AUTHOR OF "SWEET LAVENDER," "THE TIMES," "THE CABINET MINISTER," "LADY BOUNTIFUL," ETC. All rights reserved. Performance forbidden, and right of representation reserved. Application for the right of performin`
- 清洗后：`DANDY DICK A PLAY IN THREE ACTS By ARTHUR W. PINERO AUTHOR OF "SWEET LAVENDER," "THE TIMES," "THE CABINET MINISTER," "LADY BOUNTIFUL," ETC. All rights reserved. Performance forbidden, and right of representation reserved. Application for the right of performing this piece must be made to the publishers. BOSTON _WALTER H. BAKER_ Copyright, 1893, by ARTHUR W. PINERO _All rights reserved._ INTRODUCTORY NOTE. "Dandy Dick…`

### 2. 人物台词与舞台说明

- 清洗前：`ing on her knees, staring wildly into vacancy. SHEBA, a fair little girl of about seventeen, wearing short petticoats, shares her despondency, and lies prostrate upon the settee._ SALOME. Oh! oh my! oh my! oh my! SHEBA. [_Sitting upright._] Oh, my gracious goodness, goodness gracious me! [_They both walk about excitedly._ SALOME. There's only one terrible word for it--it's a fix! SHEBA. It's worse than that! It's a s…`
- 清洗后：`ng on her knees, staring wildly into vacancy. SHEBA, a fair little girl of about seventeen, wearing short petticoats, shares her despondency, and lies prostrate upon the settee._ SALOME. Oh! oh my! oh my! oh my! SHEBA. [_Sitting upright._] Oh, my gracious goodness, goodness gracious me! [_They both walk about excitedly._ SALOME. There's only one terrible word for it--it's a fix! SHEBA. It's worse than that! It's a sc…`

### 3. 幕边界

- 清洗前：`constable's collared him, Sir--he's taken him in a cart to the lock-up! GEORGIANA _and_ SIR TRISTRAM. Oh! BLORE. [_In agony._] They've got the Dean! END OF THE SECOND ACT. THE THIRD ACT. The first scene is the interior of a country Police Station, a quaint old room with plaster walls, oaken beams, and a gothic mullioned window looking on to the street. A massive door, with a small sliding wicket and an iro`
- 清洗后：`nstable's collared him, Sir--he's taken him in a cart to the lock-up! GEORGIANA _and_ SIR TRISTRAM. Oh! BLORE. [_In agony._] They've got the Dean! END OF THE SECOND ACT. THE THIRD ACT. The first scene is the interior of a country Police Station, a quaint old room with plaster walls, oaken beams, and a gothic mullioned window looking on to the street. A massive door, with a small sliding wicket and an i`

## 五类切块抽样

### Production history：`dd_prod_c0001`

- 结构：`production_history`；词数：354；说话人：`[]`
- 文本：`INTRODUCTORY NOTE. "Dandy Dick" was the third of the farces which Mr. Pinero wrote for the old Court Theatre--a series of plays which, besides giving playgoers a fresh source of laughter, and the English stage a new order of comic play, brought plentiful prosperity to the joint management of Mr. Arthur Cecil and the late Mr. John Clayton. But a kind of melancholy interest attaches to "Dandy Dick," for this play was, as it were, the swan-song of the old theatre and of the Clayton and Cecil partnership; and it was the piece in which Mr. Clayton was acting when death overtook him, to the general grief. The production of "Dandy Dick" may be consi…`

### Cast：`dd_cast_c0001`

- 结构：`cast`；词数：200；说话人：`['THE DEAN', 'SIR TRISTRAM', 'TARVER', 'DARBEY', 'BLORE', 'NOAH', 'HATCHAM', 'GEORGIANA', 'SALOME', 'SHEBA', 'HANNAH']`
- 文本：`ROYAL COURT THEATRE, SLOANE SQUARE, S.W. _Lessees and Managers:_ Mr. John Clayton and Mr. Arthur Cecil. Programme THIS EVENING, THURSDAY, JANUARY 27, _At_ 8.30 _punctually_, DANDY DICK. AN ORIGINAL FARCE, IN THREE ACTS, BY A. W. PINERO. THE VERY REV. AUGUSTIN JEDD, D.D. MR. JOHN CLAYTON. (Dean of St. Marvell's) SIR TRISTRAM MARDON, Bart MR. EDMUND MAURICE. --th Hussars, MAJOR TARVER { quartered at } MR. F. KERR. MR. DARBEY { Durnstone, near } MR. H. EVERSFIELD. St. Marvell's BLORE (Butler at the Deanery) MR. ARTHUR CECIL. NOAH TOPPING (Constable at MR. W. H. DENNY. St. Marvell's) HATCHAM (Sir Tristram's groom) MR. W. LUGG. GEORGIANA TIDMAN (a…`

### 第一幕对话：`dd_act01_c0005`

- 结构：`act_1`；词数：500；说话人：`['SALOME', 'DARBEY', 'SHEBA', 'TARVER', 'THE DEAN']`
- 文本：`SALOME. Major Tarver! [_She leads him to a chair into which he sinks in a ghastly state. DARBEY strolls in from the Library with SHEBA._ DARBEY. [_To SHEBA._] Your remarks about the army are extremely complimentary. On behalf of the army I thank you. We fellows are not a bad sort, take us all round. SHEBA. There's a grand future before you, isn't there? DARBEY. Well, I suppose there is if I go on as I'm going now. TARVER. [_To SALOME._] Thanks, the attack has passed. Now about to-night; at what time is the house entirely quiet? SALOME. Poor dear Papa goes round with Blore at half-past nine--after that all is rest and peacefulness. TARVER. The…`

### 舞台说明所在块：`dd_act01_c0001`

- 结构：`act_1`；词数：390；说话人：`['SALOME', 'SHEBA']`
- 文本：`DANDY DICK. THE FIRST ACT. _The morning-room in the Deanery of St. Marvells, with a large arched opening leading to the library on the right, and a deeply-recessed window opening out to the garden on the left. It is a bright spring morning, and an air of comfort and serenity pervades the place._ _SALOME, a tall, handsome, dark girl, of about three-and-twenty, is sitting with her elbows resting on her knees, staring wildly into vacancy. SHEBA, a fair little girl of about seventeen, wearing short petticoats, shares her despondency, and lies prostrate upon the settee._ SALOME. Oh! oh my! oh my! oh my! SHEBA. [_Sitting upright._] Oh, my gracious …`

### 第三幕第二场多人场景：`dd_act03s02_c0009`

- 结构：`act_3_scene_2`；词数：448；说话人：`['THE DEAN', 'BLORE', 'SALOME', 'SHEBA', 'NOAH', 'HANNAH', 'SIR TRISTRAM', 'GEORGIANA']`
- 文本：`THE DEAN. Why not? In the name of that tottering Spire, why not? BLORE. Oh, sir, thinking as you'd given some of the mixture to Dandy I put your cheerful little offering on to Bonny Betsy. [_SALOME and SHEBA disappear._ THE DEAN. Oh! [_To BLORE._] I could have pardoned everything but this last act of disobedience. You are unworthy of the Deanery. Leave it for some ordinary household. BLORE. If I leave the Deanery, I shall give my reasons, and then what'll folks think of you and me in our old age? THE DEAN. You wouldn't spread this tale in St. Marvells? BLORE. Not if sober, sir--but suppose grief drove me to my cups? THE DEAN. I must save you …`

## 已记录的非阻塞结构例外

- 原文最终标记实际为 `THE END`（无句点），已按真实文本配置，未补造标点。
- 原文舞台说明采用 Gutenberg 排版约定，部分只有起始 `[`，由成对斜体下划线
  标出结束；脚本同时保护方括号区间与斜体区间，未把其中人物名识别成台词。
- front matter 只有 60 词，因禁止跨结构边界合并而保留为小块。
- 两个相邻块因完整末尾台词加入后会超过 520 词，未强行制造重叠，原因已写入 chunk。
- 以上均有自动检查或显式例外记录，不构成阻塞；仍建议提交前人工抽查报告中的五类样例。

## 质量结论与边界

- 自动质量检查：pass。
- 同一进程双构建核心摘要：`d5dca418f41e54e49d5b139ed0b9597a81f2c4814248c568c75d753980cd34aa`，完全一致。
- 所有 unit 均被覆盖；chunk ID、unit ID 和链接均有效。
- 没有 chunk 跨 section、幕或场；没有空块或完全重复块。
- 小尾块例外：`[{'chunk_id': 'dd_front_c0001', 'reasons': ['entire_section_below_min_words']}]`。
- 超长块例外：`[]`。
- 无重叠例外：`[{'chunk_id': 'dd_act02_c0012', 'reasons': ['overlap_omitted_to_preserve_complete_unit_and_max_words']}, {'chunk_id': 'dd_act02_c0013', 'reasons': ['overlap_omitted_to_preserve_complete_unit_and_max_words']}]`。
- 阶段 1 输入文件保持不变。

**下一步是构建共享知识图谱；本阶段没有自行开始。**
