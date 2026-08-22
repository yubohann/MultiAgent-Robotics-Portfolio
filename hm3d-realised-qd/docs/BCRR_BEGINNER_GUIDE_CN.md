
# BCRR 零基础说明：它是什么，以及它和旧方法有什么区别

日期：2026-07-29

状态：`BEGINNER GUIDE / BCRR IS A RESEARCH HYPOTHESIS / NOT A PROVEN METHOD`

这份文档只负责把 BCRR 讲明白。它不新增科研主张，也不表示 BCRR 已经完成实现、定理证明或正式实验。

## 先用一句话说明

旧方法解决的是：

> 已经有多套完整的多无人机飞行路线，团队下一步应该选哪一套？

BCRR 解决的是：

> 当某些无人机可能已经失效，但我们又不能直接知道谁真的坏了时，剩下的无人机应该怎样重新分担区域搜索、楼层观察和通信中继等责任？

最重要的变化是：

> **旧方法主要选择路线；BCRR 主要重组责任。**

路线是“具体怎样飞”，责任是“必须完成什么工作、由谁负责”。BCRR 先处理责任，再把责任解码成路线和可执行命令。

---

## 1. 先理解我们面对的任务

假设有一支无人机队伍在城市中搜索目标。它们需要完成的不只是“到处飞”，而是许多不同类型的工作，例如：

- 搜索低层入口；
- 检查建筑立面；
- 查看屋顶；
- 搜索某片街区；
- 保持不同无人机之间的通信；
- 充当通信中继，也就是 relay；
- 避免多架无人机重复搜索同一位置；
- 在电量、时间和安全约束内完成任务。

这里的每架无人机都可以称为一个“智能体”。智能体就是一架无人机和控制它的软件的组合，不是什么神秘概念。

### 1.1 什么是路线

路线描述的是具体运动：

```text
从坐标 A 起飞
→ 沿道路向北飞 30 米
→ 上升到 18 米
→ 在建筑东立面悬停 3 秒
→ 转向下一个航点
```

路线非常具体，通常和当前地图、无人机数量、起点及障碍物绑定。

### 1.2 什么是责任

责任描述的是团队必须完成的工作：

```text
负责东区低层入口搜索
负责屋顶观察
负责维持通信中继
负责接管失联成员留下的中层立面
```

同一项责任可以通过多条不同路线完成。责任比路线更抽象，也更容易在无人机数量变化后重新分配。

### 1.3 为什么不能只研究路线

假设原来有六架无人机，一套完整联合路线同时规定了六架无人机怎样飞。如果其中一架突然失效，原来的六机联合路线就可能整体不再适用。

真正需要解决的问题不是简单地删除一条轨迹，而是：

- 失效无人机原来负责的区域由谁接管？
- 如果它是通信中继，团队通信会不会断开？
- 接管工作的无人机是否有足够电量和时间？
- 接管后会不会放弃它原来的工作？
- 重新分工需要多少通信、规划和转向成本？

这就是为什么 BCRR 把“责任”放在“路线”前面。

---

## 2. BCRR 这个名字是什么意思

`BCRR` 是 `Belief-Conditioned Responsibility Repertoire` 的缩写。

可以把它拆成四部分理解。

### 2.1 Belief：信念

这里的 belief 不是宗教意义上的信念，而是“根据有限证据形成的概率判断”。

例如 UAV-5 连续没有心跳，也没有回复消息：

```text
UAV-5 确实损坏了？
UAV-5 只是暂时进入通信盲区？
UAV-5 的通信模块损坏，但仍能飞？
UAV-5 的相机坏了，但还能充当 relay？
```

BCRR 不能偷偷读取仿真器中的真实故障 ID。它只能根据合法观测形成类似下面的判断：

```text
UAV-5 正常工作的概率：10%
UAV-5 暂时失联的概率：30%
UAV-5 已经无法继续任务的概率：60%
```

这个判断就是 health belief。

### 2.2 Conditioned：以当前判断为条件

BCRR 的责任选择不能只看地图，还要根据当前 belief、通信图、剩余电量、无人机能力和时间条件进行调整。

“belief-conditioned”的意思就是：

> 在我们当前能够掌握的健康证据条件下做决定，而不是假装已经知道真实故障答案。

### 2.3 Responsibility：责任

BCRR 分配的主要对象不是完整轨迹，而是可解释的三维搜索责任，例如：

- 区域责任；
- 高度带责任；
- 入口、立面、屋顶等观察责任；
- 通信中继责任；
- 工作负载；
- 临时 coalition，也就是由多架无人机共同承担的一项责任；
- 重复覆盖和暂时无人负责的任务。

责任描述符不能包含目标坐标、真实故障 ID 或其他 evaluator-private 真值。

### 2.4 Repertoire：经过准备的一组不同分工方案

repertoire 可以理解为“方案库”或“技能库”。

它不是只保存一个最优方案，而是保存多种高质量、彼此不同的责任组合，例如：

- 屋顶优先型；
- 入口优先型；
- 通信中继优先型；
- 左右分区型；
- 高度分层型；
- 高冗余、抗单机失效型；
- 低能耗型。

发生故障后，算法不必每次都从零开始搜索所有可能的团队分工。它可以从 repertoire 中选择或组合适合当前情况的责任 code，再做局部修复。

但是，repertoire 是否真的比从头 greedy 或 auction 重分配更好，必须通过实验验证，不能预设。

---

## 3. 用一个完整例子理解 BCRR

假设六架无人机原来的分工如下：

| 无人机 | 原责任 |
| --- | --- |
| UAV-1 | 搜索低层入口 |
| UAV-2 | 搜索中层建筑立面 |
| UAV-3 | 搜索屋顶 |
| UAV-4 | 搜索东区街道 |
| UAV-5 | 充当通信 relay |
| UAV-6 | 负责高层观察 |

### 3.1 故障发生

UAV-5 突然停止发送心跳。

BCRR 不能直接从仿真器读取：

```text
failed_agent_id = UAV-5
```

它只能看到：

- 心跳中断；
- 消息没有回复；
- 最近一次位置；
- 最近的运动、能量和传感器反馈；
- 其他无人机是否还能通过 UAV-5 转发消息。

### 3.2 更新 health belief

算法根据这些证据判断 UAV-5 可能已经不能继续承担 relay 责任，但仍保留“暂时通信丢失”的可能性。

### 3.3 删除投影

BCRR 暂时把 UAV-5 从候选 active set 中删除，并把原分工投影到幸存团队。

其他无人机的责任仍然保留；UAV-5 的 relay 责任失去了负责人。

### 3.4 产生 orphan responsibility

没有负责人接管的 relay 责任被明确标记为：

```text
orphan responsibility
```

中文就是“孤儿责任”或“暂时无人负责的任务”。

算法不能假装这项任务自动消失，因为 relay 的消失可能造成通信图分裂。

### 3.5 从 repertoire 中选择合适的分工结构

BCRR 发现方案库中有一种“中继优先、局部压缩搜索范围”的责任 code，适合当前情况。

它不是一套写死的五机路线，而是一种可解码到当前幸存团队的分工结构。

### 3.6 约束修复

算法在容量、通信、安全和截止时间约束下进行修复，例如：

```text
UAV-4 暂时接管 relay
UAV-1 与 UAV-2 分担 UAV-4 原来的部分街区
UAV-3 继续负责屋顶
UAV-6 缩小高层观察范围，减少切换成本
```

这里必须计算代价：

- UAV-4 改做 relay 后损失多少街区搜索能力；
- UAV-1 和 UAV-2 是否有足够电量接管；
- 新分工能否恢复通信；
- 需要发送多少消息；
- 需要重新规划多少轨迹；
- 有多少任务仍然无人负责；
- 恢复需要多长时间。

### 3.7 解码并执行

责任分配确定后，共享策略才把责任解码成每架无人机的路线和 ActionToken。

无人机执行后产生 receipt。receipt 说明实际执行了什么、观察来自哪一帧、是否失败以及真实承担了哪些责任。下一轮决策再使用这些公开证据更新 belief 和分工。

---

## 4. BCRR 的完整流程

![BCRR 从观测到责任修复再到执行的流程](assets/bcrr-beginner-flow.svg)

对应的纯文字版本是：

```text
合法公开观测
  ↓
估计哪些无人机仍可能正常工作
  ↓
编码当前无人机、三维责任和通信关系
  ↓
从责任 repertoire 中选择或组合责任 code
  ↓
把责任投影到疑似幸存成员
  ↓
找出 orphan responsibilities
  ↓
按容量、通信、安全和 deadline 修复责任
  ↓
解码成路线与 ActionToken
  ↓
真实执行并记录 receipt
  ↓
更新 belief 和下一轮责任选择
```

图示源文件位于 `docs/assets/bcrr-beginner-flow.mmd`，同时提供 SVG 和 PNG 版本。

---

## 5. BCRR 的六个必要组成

| 组成 | 用最简单的话解释 | 它回答的问题 |
| --- | --- | --- |
| Set/graph encoder | 把数量可变的无人机、责任和通信关系读懂 | 当前有谁、能做什么、谁能和谁通信？ |
| Fragment responsibility descriptor | 记录短时间内真实承担了哪些责任 | 刚才到底是谁搜索了哪里、谁做 relay、哪里重复了？ |
| Compact responsibility repertoire | 保存多种紧凑且不同的分工 code | 遇到不同删员和通信情况时，有哪些可快速调用的分工？ |
| Health-belief filter | 根据合法证据估计成员健康状态 | 谁可能失效，判断有多确定？ |
| Deletion projection + constrained repair | 删除疑似失效成员，并重新分配遗留责任 | 谁留下了 orphan responsibility，应该交给谁？ |
| Online selection/composition | 根据当前状态选择、组合并执行责任方案 | 此刻用哪个 code，怎样落成真实任务和动作？ |

六个部分的地位并不完全相同：

- repertoire 的删除稳定性是候选理论核心；
- health belief 是信息条件，不是第二个并列创新；
- set/graph encoder 是支持变量团队的标准表示工具；
- ActionToken、receipt 和 safety guard 是可靠执行设施；
- descriptor-conditioned critic 是从 MIQD 等工作继承的学习设施；
- successor features 只有在消融证明有独立价值时才保留。

模块多不等于创新强。BCRR 必须围绕一个核心问题，而不是把模块名称堆在一起。

---

## 6. QD 在 BCRR 中到底做什么

QD 是 Quality-Diversity，中文常译为“质量-多样性”。

普通优化通常只寻找一个最高分方案。QD 希望同时得到：

- 质量不错；
- 行为或分工方式彼此不同。

在 BCRR 中，QD 的候选作用是发现一组互补的责任方案，而不是保存很多几乎相同的路线。

### 6.1 旧式 archive 可能保存什么

```text
方案 A：一整套四机网络或完整四机路线
方案 B：另一整套四机网络或完整四机路线
方案 C：又一整套四机网络或完整四机路线
```

这种 archive 很难直接用于三机、五机或某个成员被删除后的团队。

### 6.2 BCRR 希望保存什么

```text
code A：入口优先 + 一个 relay + 低重复覆盖
code B：屋顶优先 + 高度分层 + 两机协作
code C：通信脆弱时的双 relay + 压缩搜索范围
```

共享 set/graph policy 把 code 解码到当前无人机集合，而不是每个 cell 保存一整套固定 N 的团队网络。

### 6.3 为什么 QD 可能没有必要

如果发生故障后，下面这个简单方法已经足够好：

```text
一个鲁棒 graph policy
+
一次 greedy 或 auction 任务重分配
```

而且它比构建、维护和选择 repertoire 更快、更省，那么 BCRR 就没有理由硬保留 QD。

所以 QD 是待验证机制，不是信仰。

---

## 7. 旧方法到底是什么

旧方法可以概括为：

> `realised-QD + RB-SF-SAC + 安全执行与证据链`

旧方法的基本流程是：

```text
生成多套完整联合路线候选
  ↓
安全 mask 删除明显非法候选
  ↓
RB-SF-SAC 评价每套候选
  ↓
选择一套完整联合路线
  ↓
Isaac 中执行
  ↓
receipt 记录真实执行结果
  ↓
计算 realised descriptor
  ↓
更新 realised-QD archive 和 critics
```

其中：

- actor 负责选择候选；
- task critic 估计任务收益；
- SF critic 估计行为特征；
- cost critic 估计碰撞、能量和通信等成本；
- realised-QD 按真实执行行为而不是计划行为归档；
- ActionToken、receipt 和 safety guard 保证命令、执行与证据链一致。

这个方法适合的问题是：

> 已经有一批完整联合路线，现在选哪一批更好？

它的主要局限是：一套完整联合路线通常绑定当前团队规模。一架无人机消失后，不只是少了一条轨迹，整个联合候选的结构都可能失效。

---

## 8. 新旧方法详细对比

| 对比方面 | 旧方法：RB-SF-SAC + realised-QD | 新方法：BCRR |
| --- | --- | --- |
| 核心问题 | 固定团队下一步选择哪套联合路线 | 部分可观测成员删除后怎样重组三维搜索责任 |
| 主要决策对象 | 完整 joint route candidate | 可组合 responsibility code |
| 决策顺序 | 先有完整路线，再评价和选择 | 先分配/修复责任，再解码为路线 |
| 团队规模 | 更适合相对固定的 N | set/graph 表示支持变量 N 和 active set |
| 成员失效 | 主要通过上下文、成本或重新选候选间接处理 | 显式 health belief、删除投影、orphan 与 repair |
| 故障信息 | 旧设计没有把故障不可见性作为理论中心 | 禁止读取真实故障 ID，只能从 health evidence 推断 |
| 通信变化 | 作为上下文或成本的一部分 | 直接影响 relay 责任、连通性和 repair 可行性 |
| QD archive 内容 | 完整路线、策略或 realised behavior elite | responsibility code、prototype 或小型 adapter |
| 行为描述符 | 空间覆盖、垂直覆盖等真实执行结果 | 区域、高度、观察机会、relay、负载、重复和 orphan |
| RL 的角色 | 直接评价并选择联合路线候选 | 编码团队、评价责任 code、解码责任和辅助 repair |
| successor features | 旧方法的重要组成 | 条件保留；无独立收益就删除 |
| 理论重点 | planned-to-realised、归档稳定、候选置信等分散问题 | repertoire 删除稳定性与 reallocation regret 这一核心 |
| 主要恢复动作 | 重新生成或重新选择完整联合路线 | 局部投影、产生 orphan、约束修复和责任切换 |
| 跨 N 复用 | 完整联合路线难直接复用 | 共享 policy 把同一责任 code 解码到不同团队 |
| 主要指标 | reward、recall、archive coverage、安全成本 | recall AUC、韧性损失、恢复时延、orphan、switching cost 等 |
| QD 地位 | 方法身份中的默认外层 | 必须击败 no-QD、repair 和 ensemble 后才允许保留 |
| 当前成熟度 | 已有旧代码和部分单元测试 | 当前仍是待证明和待实现的研究假设 |

最简洁的区别是：

```text
旧方法：从“完整答案列表”里挑一个答案。

新方法：团队成员变化后，先重新决定“谁负责什么”，
        再把新的分工变成可执行路线。
```

---

## 9. 新方法是不是把旧方法全部推翻

不是。正确关系是“保留可靠设施，重写核心对象，删除不再成立的研究身份”。

### 9.1 明确保留或重新验证后复用

- ActionToken；
- command-before-step；
- variable-duration timing；
- source-observation binding；
- receipt；
- safety guard；
- failure ledger；
- 配置 hash 和证据链；
- 碰撞、能量与通信成本审计；
- 与场景无关的 QD/RL 单元测试。

### 9.2 需要重写或大幅改造

- RB-SF-SAC：从固定候选改成变量长度、pad + mask 批处理；
- neural selector：改为 set/graph 编码；
- realised-QD：改成 responsibility-code repertoire；
- replay 和 critics：向量化并使用 shared trunk；
- process boundary：改为持久 worker，同时保留 canary、hash 和隔离；
- Isaac 运行：静态城市持久化，动态 EpisodeSpec reset。

### 9.3 明确不能继承

- RiverMark/CityLite 固定坐标和路线库；
- 固定四机假设；
- 完整 joint route 作为主要 QD elite；
- CCUQD/SCRR 旧名称；
- 旧的三项薄理论主张；
- 没有重新运行和重新准入的旧结果；
- 读取目标坐标、目标数量、FaultSpec 或失效 agent ID 的捷径。

所以 BCRR 不是旧方法换名字，也不是把所有旧代码删除重写。它是：

```text
保留旧方法可靠的执行和证据基础
       +
把核心决策对象从完整路线改成可组合责任
       +
加入部分可观测删员下的投影、orphan 和修复问题
```

---

## 10. BCRR 的理论创新候选是什么

先说结论：

> BCRR 有理论创新的候选方向，但目前没有完成定理证明，因此现在不能说“已经有理论创新”。

候选理论核心只有一个：

> 在无法直接知道真实故障成员时，一个紧凑、可组合的责任 repertoire 在成员删除后能否稳定投影和修复，并控制相对理想方案的性能损失？

### 10.1 什么是理想对照

理想对照知道真实幸存无人机集合，并在同一任务与安全条件下重新寻找最佳分工。它叫 deletion-aware optimum。

BCRR 不知道真实集合，只持有 belief。因此我们要研究 BCRR 比理想对照损失多少。

### 10.2 候选公式的直白解释

研究合同中的目标形式是：

```text
BCRR 修复后的搜索效用 - 切换成本
至少应接近
知道真实幸存集合的理想效用
- repertoire 不完整造成的损失
- health belief 判断错误造成的损失
- 通信受损造成的损失
- 责任切换和重新规划造成的损失
```

形式上写成：

```text
U(Repair(P_A(z))) - lambda C_switch
>= alpha OPT(A*)
   - epsilon_repertoire
   - epsilon_belief
   - epsilon_connectivity
   - lambda switching_penalty
```

这只是“需要证明的目标”，不是已证明公式。

### 10.3 真正可能构成贡献的内容

至少需要证明或严格验证以下内容：

1. 一个紧凑 code repertoire 能覆盖许多不同的成员删除情况，不必为每个幸存集合保存整套团队网络；
2. health belief 的误检和漏检怎样具体增加 reallocation regret；
3. repertoire 选择加局部 repair 是否比故障后从头优化更快，并在 deadline 内产生净收益；
4. 通信图断裂、幸存容量不足或责任不可分时，哪些问题本来就无解。

### 10.4 什么不算 BCRR 的理论创新

- 使用 QD；
- 多智能体强化学习；
- descriptor-conditioned critic；
- fragment descriptor；
- successor features；
- safety critic；
- permutation equivariance；
- 普通置信区间；
- 标准次模 greedy 的近似界；
- 把多个已有模块放在同一个系统中。

如果最终只能证明“删掉一架无人机后再运行 greedy，仍有普通 greedy 保证”，那么 BCRR 的理论创新失败。

---

## 11. 怎么判断 BCRR 到底有没有价值

BCRR 必须回答三个层次的问题。

### 11.1 任务质量

- 搜索和确认目标是否更快；
- confirmed-recall AUC 是否更高；
- 故障后任务质量下降多少；
- 是否减少无人负责和重复负责的区域。

### 11.2 恢复效率

- 多久发现可能失效；
- 多久产生第一次合法重分配；
- 恢复通信需要多久；
- 在线选择和 repair 需要多少计算；
- 切换责任和重规划造成多少成本。

### 11.3 付出的全部代价

- repertoire 构建成本；
- archive 重评估成本；
- environment interactions；
- 原生 physics seconds；
- wall-clock；
- CPU/GPU-hours；
- RAM/VRAM；
- 模型和 archive 大小；
- timeout、崩溃、OOM 和失败 seed。

不能只看最终分数，也不能只看 evaluation count。必须扣除为了建立 repertoire 花掉的时间和算力。

---

## 12. BCRR 必须击败哪些简单方法

这些基线不是陪衬，而是可以直接判死 BCRR 的 kill baselines：

- single set/permutation-equivariant robust policy；
- graph MAPPO/RMAPPO + agent/message dropout；
- greedy responsibility repair；
- auction/market reassignment；
- no-QD composable responsibility library；
- random/policy ensemble；
- MIQD-compatible clean-room reproduction；
- COPA/open-ad-hoc style policy；
- fault-aware centralized oracle。

比较必须保证：

- 相同 observation privilege；
- 相同环境 interactions；
- 相同 physics 和 sensing 合同；
- 相同安全 guard；
- 可比较的参数量；
- 同时报 matched-interactions 和 matched-wall-clock。

如果简单方法已经达到同样效果，正确做法是删除不必要的 QD 或停止 BCRR，而不是继续增加模块。

---

## 13. 当前到底做到哪一步了

截至 2026-07-29，已经完成的是：

- 研究问题收缩；
- MIQD 和主要近邻的初步审查；
- 新旧方法边界；
- BCRR 的六个必要组成；
- 候选理论目标；
- kill baseline 与停止条件；
- 旧代码复用审计；
- 长进程加速和分阶段执行计划。

尚未完成的是：

- BCRR 正式算法实现；
- responsibility、projection 和 repair 的完整形式化；
- toy exact oracle 和反例；
- 定理证明；
- MIQD-compatible 原任务复现；
- matched-budget kill baseline 实验；
- 原生 Isaac 正式训练；
- AeroCityBench frozen evaluation；
- 任何正式性能或新颖性结论。

所以当前最准确的说法是：

> BCRR 是一条经过文献审查后值得进入 M0–M2 反证阶段的研究路线，不是一套已经成功的方法。

---

## 14. 接下来为什么不直接跑几天 Isaac

因为现在最大的风险不是“训练时间不够”，而是“研究假设可能本身不成立”。

当前顺序是：

### M0：先证明问题定义不是空话

- 定义 responsibility；
- 定义 deletion projection；
- 定义 orphan 和 repair；
- 做 exact oracle；
- 构造两个正例、两个反例和一个无解条件；
- 找出 greedy 什么时候够用、什么时候不够用。

### M1：先学会并复现最强近邻

- 做 MIQD-compatible clean-room 原任务复现；
- 验证 fragment、MI reward 和 neighborhood policy 的趋势；
- 建立变量长度向量化内核；
- 用 QDax/JAX 测试 code repertoire。

### M2：让简单方法来攻击 BCRR

- no-QD library；
- graph dropout；
- greedy；
- auction；
- ensemble；
- BCRR。

只有 BCRR 在至少一个预注册结构条件下，扣除 repertoire 成本后仍有净恢复收益，才允许继续进入昂贵阶段。

---

## 15. 常见误解

### 误解 1：BCRR 就是 MIQD 加一个故障输入

不是。MIQD 的主要对象是固定二智能体团队的完整 policy archive。BCRR 的主要对象是变量团队的责任 code，以及成员删除后的投影和修复。

如果实现只是给 MIQD 输入增加一个 `fault_id`，那不仅没有解决 BCRR，还违反了故障真值不可见的合同。

### 误解 2：BCRR 是故障诊断方法

不是。health-belief filter 可以替换，也可以使用已有诊断器。诊断结果是 BCRR 的信息条件。BCRR 的候选核心是 belief 有误时责任 repertoire 怎样投影、修复并承担 regret。

### 误解 3：无人机数量越多，QD 就越必要

不成立。团队大只说明组合更多，不说明 QD 比 greedy、auction 或 ensemble 更有价值。QD 的必要性必须通过净收益证明。

### 误解 4：算法模块越多，创新越强

不成立。BCRR 必须只有一个理论核心。ActionToken、receipt、安全 critic 和 SF 都可能有工程价值，但不能并列包装成多个创新。

### 误解 5：在 toy 或 geometry-only 环境获胜就算成功

不成立。低 fidelity 用于定义、筛选和加速。正式 elite、baseline 和论文结果最终必须在相同的原生 Isaac 合同下独立复评。

### 误解 6：旧代码没有用了

不成立。旧代码中的执行、证据、安全和测试设施很有价值，但旧问题设定、固定路线和旧理论身份不能直接继承。

### 误解 7：现在已经可以说 BCRR 有理论创新

不可以。当前只能说“存在明确的理论创新候选”。完成非平凡定理、近邻查重和 kill baseline 反证后，才能决定是否成立。

---

## 16. 最后用三个不同长度复述

### 十秒版本

BCRR 在无人机可能失效且真实故障身份不可见时，重新组合团队的三维搜索和通信责任。

### 三十秒版本

BCRR 不直接选择一整套固定团队路线。它根据心跳、消息、运动和能量证据估计哪些无人机仍然可用，从一个紧凑责任方案库中选择分工，删除疑似失效成员的责任，把无人负责的区域和 relay 工作重新分给幸存者，再解码成真实路线和 ActionToken。它的价值必须超过简单 graph policy、greedy/auction repair 和普通 ensemble。

### 一分钟版本

旧方法在已有完整联合路线中做选择，适合相对固定的团队。BCRR 把决策对象提升为区域、高度、观察机会和通信中继等可组合责任。当成员可能失效时，方法不能读取真实故障 ID，只能形成 health belief；然后将原责任投影到候选幸存团队，把被删除成员留下的任务标记为 orphan，在容量、通信、安全和截止时间约束下修复，并计算任务质量损失与切换成本。QD 只负责尝试建立多样的紧凑责任 repertoire；如果不用 QD 的简单方法一样好，就删除 QD。当前 BCRR 仍处于 M0–M2 的形式化和反证阶段。

---

## 17. 权威文档

本说明服从以下权威材料；若以后这些材料更新，本说明也必须同步：

- `README.md`
- `docs/NEW_CONVERSATION_HANDOFF.md`
- `docs/METHOD_RESEARCH_CONTRACT.md`
- `docs/NOVELTY_MATRIX.md`
- `docs/MIQD_AND_RESEARCH_VALUE_REVIEW_2026-07-29.md`
- `docs/METHOD_EXECUTION_AND_ACCELERATION_PLAN.md`
- `docs/LEGACY_REUSE_AUDIT.md`
- `reason/method-final-decision-20260729/verdict.md`

当前裁决是：

```text
REFORMULATE legacy method -> GO on BCRR M0-M2
```

它的含义是：允许继续形式化、复现和反证；不表示 BCRR 已经证明有效。
