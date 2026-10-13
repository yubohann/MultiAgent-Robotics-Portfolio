# FraudGraph ML Engineering：300 字概括

FraudGraph ML Engineering 将金融交易转换成图结构、关系序列和事件序列三种视图，用 SplitGNN、Transformer 和融合分类头开展欺诈检测。仓库提供 IEEE-CIS、Elliptic、AMLSim、信用卡、Ethereum 和 DeFi 等数据适配器，训练、评估、checkpoint、embedding 分析、TensorBoard 审计和 comparison report 均有独立入口。manifest 记录 seed、数据修订、配置、环境和命令；validation 负责选模型和阈值，held-out test 只在最后评估；数据和生成图不自动下载或提交。项目围绕图信号、序列增益和低标签鲁棒性构造可证伪协议，适合展示图学习、时序建模、第三方代码归属和 ML 实验治理。任何性能结果都必须绑定数据版本、划分、配置和 artifacts，不能仅凭 README 宣传语判断为统一 SOTA。

