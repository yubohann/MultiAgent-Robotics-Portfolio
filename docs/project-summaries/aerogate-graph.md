# AeroGate Graph：300 字概括

AeroGate Graph 是固定高度二维无人机竞速研究环境，提供单机、可变规模多机和动态门密度任务。核心由受限运动学、圆形门柱碰撞、A* 全局路由、图观测、虚拟结构编队、动作安全 shield 和统一 reward/termination 组成；Graph-SAC、Graph-MASAC、BC/DAgger 和 FlashSAC 是可选训练路径，Isaac Lab 只作为可选回放适配器。NumPy core 无需 GPU 即可运行 smoke 和 reproduce，测试覆盖布局、配置、公开 API、团队几何和确定性诊断。项目适合展示多智能体规划、图强化学习、编队控制和可复现实验设计；二维固定高度结果不能外推到真实飞行动力学、感知、通信或执行器安全，训练和渲染证据必须与 CPU 合同分开说明。

