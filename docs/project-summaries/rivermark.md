# Rivermark Benchmark：300 字概括

Rivermark 是面向 Isaac Sim 多 UAV 三维搜索研究的基准基础设施。项目围绕八架 CF2X，统一采集 RGB、深度、语义、LiDAR、IMU、动作、接触和外参，并用协议、运行时、源代码和 episode receipt 的 SHA-256 绑定保证可追溯。CPU 路径可以独立执行 schema、泄漏、重复性和数据准入检查，Isaac Sim 路径负责原生 PhysX 采集；评测器通过稳定 ABI 兼容经典规划、RL/MARL、QD、离线 RL 和 VLA。项目的核心不是一段视频，而是把数据 provenance、失败 ledger、崩溃恢复、split 隔离和多范式评测做成可审计工程。它适合展示机器人 benchmark、实验治理和可复现研究能力；CPU 测试、媒体和 formal episode 必须按各自证据等级解释，不能把仿真基础设施直接等同于真实飞行安全或论文 SOTA。

