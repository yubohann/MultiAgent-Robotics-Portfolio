# HM3D Realised-QD：300 字概括

HM3D Realised-QD 研究 HM3D-derived 三维环境中的无目标多 UAV 协作探索，正式观测为 `sparse_range_3d`，主指标是自由飞行探索体积随物理时间的 AUC。每架 CF2X 从局部 belief 和有限通信中产生候选，realised-QD 只在真实执行 receipt、覆盖体积、时间和失败原因生成后更新 archive；RL、baseline、safety、runtime 和 evaluation 分成独立模块。P0-P10 文档冻结任务、控制器、预算、数据和统计边界，IsaacLab wrapper 负责原生回放，HM3D 资产、私有评测、checkpoint 和 raw log 保留在本地。项目适合展示 QD、多智能体探索、稀疏传感器、通信和安全执行的结合；单元测试或回放通过不能自动升级为正式主表结果，必须绑定相同场景、控制器、观测 profile、时间预算和 manifest。

