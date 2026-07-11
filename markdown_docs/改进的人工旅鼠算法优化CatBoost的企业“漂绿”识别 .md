## 改进人工旅鼠算法优化 **CatBoost** 的企业“漂绿”识别与动因挖掘 

+ 郭一帆 , 刘升 , 李原 

上海工程技术大学 管理学院,上海  201620 + 通信作者 E-mail：ls6601@163.com 

摘要： 针对企业“漂绿”识别样本高度不平衡，分类模型超参数寻优困难的问题，提出一种融合 SMOTE-Tomek 采 样与改进人工旅鼠算法（improved artificial lemming algorithm，IALA）优化 CatBoost 的“漂绿”识别模型（SMOTETomek-IALA-CatBoost）。首先，针对人工旅鼠算法寻优精度不高、易陷入局部最优的问题，IALA 引入多子群环 形迁移、信息熵自适应种群规模与跨子群精英解回溯三种策略，并在 CEC2022 测试集上与先进智能算法对比，验 证其寻优性能。其次，由 IALA 对 CatBoost 的学习率、树深度关键超参数寻优。最后，针对类别不平衡，采用 SMOTE-Tomek 组合采样，并与 SMOTE、ROS、ADASYN、RUS、NearMiss、Tomek Links 等采样方法及 RF、 XGBoost、SVM 等模型组合对比，以召回率、精确率、F1 与 AUC 衡量。结果表明，SMOTE-Tomek-IALA-CatBoost 的整体表现最优，SHAP 分析所识别的高贡献特征符合“漂绿”行为的实际表现，验证了模型识别依据的合理性。 关键词： 人工旅鼠算法；“漂绿”识别；CatBoost 模型；不平衡数据；SHAP 文献标志码 **: A** 中图分类号 **:** TP181 

## **Corporate Greenwashing Identification Using an Improved Artificial Lemming Algorithm-Optimized CatBoost** 

Guo Yifan,Liu Sheng[+] ,Li Yuan 

School of Management, Shanghai University of Engineering Science, Shanghai 201620, China 

**Abstract** ：Greenwashing identification of enterprises faces severely imbalanced samples and difficult hyperparameter optimization of classification models. To address these issues, a greenwashing identification model that integrates SMOTETomek sampling with an improved artificial lemming algorithm (IALA) for optimizing CatBoost is proposed, denoted SMOTE-Tomek-IALA-CatBoost. To mitigate the tendency of the artificial lemming algorithm to fall into local optima, IALA introduces three strategies, namely multi-subpopulation ring migration, entropy-based adaptive population size, and elite-archive backtracking, and its optimization performance is verified against advanced intelligent algorithms on the CEC2022 test suite. IALA then optimizes the key CatBoost hyperparameters such as tree depth and learning rate. For the class imbalance problem, SMOTE-Tomek hybrid sampling is adopted and compared, together with the sampling methods SMOTE, ROS, ADASYN, RUS, NearMiss, and Tomek Links and the models RF, XGBoost, and SVM, in terms of recall, precision, F1, and AUC. The results show that SMOTE-Tomek-IALA-CatBoost achieves the best overall performance, and the high-contribution features identified by SHAP analysis are consistent with the actual manifestations of greenwashing, which verifies the plausibility of the identification basis of the model. 

**Key words** ：artificial lemming algorithm; greenwashing detection; CatBoost model; imbalanced data; SHAP 

> 作者简介 **:** 郭一帆(2000—)，男，硕士研究生，硕士研究生，CCF 会员，研究方向为智能算法；刘升(1966—)，男，博士，教授，研究方向为 进化算法；李原(2001—)，女，硕士研究生，硕士研究生，CCF 会员，研究方向为智能算法。 收稿日期： 2025-00-00 修回日期 ：2025-00-00 

--- end of page.page_number=1 ---

## **0** 引言 

近年来， ESG 投资理念在全球资本市场快速渗透，企 业通过夸大环境绩效、选择性披露负面信息等手段伪造可 持续发展形象，“漂绿”行为日益频发，严重损害了投资者 权益与绿色金融政策的实施效果[[1]] 。企业的策略性“漂绿” 行为会增加绿色转型风险，阻碍经济绿色发展，是 ESG 治 理研究的核心问题之一[[2]] 。准确识别企业 “ 漂绿 ” ，是实施 监管干预与市场治理的重要前提。 

已有研究多借助第三方 ESG 评级差异来识别企业“漂 绿”。 He 等基于中国上市公司发现，言行不一的企业往往 伴随更大的 ESG 评级分歧[[3]] 。 Zhang 等进一步表明，评级 分歧越大、企业实施“漂绿”的倾向越强[[4]] 。然而，评级 差异并不能直接等同于“漂绿”。 Berg 等指出，不同评级 机构在指标权重、数据来源与衡量方法上存在显著异质性， 对同一企业的评分常出现较大分歧[[5]] 。李九斤等基于上市 公司案例发现，企业可借助选择性披露在不同评级机构间 获取较高评分，使评级差异法在实际应用中存在局限[[6]] 。 

为突破上述局限，学者们逐步将机器学习引入“漂绿” 识别。杨七中和马蓓丽将环境信息披露积极、却因环境违 规受到处罚的企业界定为“漂绿”样本，据此构建随机森 林、 XGBoost 等模型并以 SMOTE 过采样缓解样本不平衡 [7] 。 Chen 和 Ma 基于 Word2Vec 与 TF-IDF 等文本挖掘方 法，从 A 股上市公司 ESG 报告中量化测度企业“漂绿” 程度[[8]] 。 Zhang 等构建基于 XGBoost 与 SHAP 的预测模型， 并与 RF 、 SVM 、 LightGBM 等比较[[9]] 。 

相较于评级差异法，机器学习方法有效提升了“漂绿” 识别的效率与准确性。然而，“漂绿”样本在全部样本中占 比极低，类别不平衡会使模型偏向多数类、影响识别效果。 合理的不平衡处理对提升识别性能具有重要作用[[10]] 。此外， 模型性能对超参数较为敏感，网格搜索、随机搜索等传统 方法在参数寻优中效率有限，难以获得最优配置[[11]] 。 

针对上述类别不平衡与超参数寻优问题，本文提出一 种融合 SMOTE-Tomek 采样与改进人工旅鼠算法（ IALA ） 优化 CatBoost 的企业“漂绿”识别模型。主要贡献如下： 

（ 1 ）针对正负样本悬殊问题，采用 SMOTE-Tomek 组 合采样： SMOTE 在少数类邻域内合成新的“漂绿”样本， Tomek Links 删除类别边界处相互混叠的样本，在补充少 数类的同时使类间边界更易区分。将其与 SMOTE 、 ROS 、 ADASYN 、 RUS 、 NearMiss 、 Tomek Links 对比，以召回 率、精确率、 F1 与 AUC 评价， SMOTE-Tomek 综合表现 较好，能有效提升模型对“漂绿”样本的识别能力。 

（ 2 ）针对人工旅鼠算法收敛精度不足、易陷入局部 最优的问题， IALA 将种群划分为多个子群并以环形方式 迁移最优个体，维持种群多样性。在此基础上，依据信息 

熵动态调整种群规模，平衡全局探索与局部开发。此外， 构建跨子群精英档案，使历史优质解重新参与迭代。在 CEC2022 测试集上， IALA 的均值与标准差综合排名第一， 在多数测试函数上的收敛精度与稳定性优于对比算法。 

（ 3 ）针对“漂绿”特征中行业属性、企业性质等类别 型特征较多的特点，选用可直接处理类别特征、并以有序 提升缓解预测偏移的 CatBoost 。利用 IALA 对其关键超参 数寻优，构建 SMOTE-Tomek-IALA-CatBoost “漂绿”识别 模型。在对比实验中，该模型的识别效果优于其他对比模 型。最后，借助 SHAP 分析各特征贡献，重要特征与 ESG 审查实践重点关注的内容基本一致，验证了模型的合理性。 

## **1** 基本人工旅鼠算法 

**==> picture [251 x 60] intentionally omitted <==**

其中， _rand_ 为 [0,1] 内的随机数； _UB j_ 和 _LB j_ 分别是第 _j_ 维 的上界和下界。 

## **1.1** 长途迁徙阶段 

当种群密度过高导致食物短缺时，旅鼠依据当前位置 与种群中随机个体的位置进行长距离迁徙，位置更新公式 见式 (2): 

**==> picture [251 x 60] intentionally omitted <==**

搜索个体。 _BM_ 为表征布朗运动的随机数向量，其步长由 方差为 1 、均值为 0 的标准正态分布的概率密度函数生成： 

**==> picture [198 x 30] intentionally omitted <==**

_F_ 为搜索方向标志，用于避免陷入局部最优。 _R_ 为控制最 优个体与随机个体相对移动的 1  _Dim_ 随机向量，其元素 均匀分布在 [ − 1,1] 内。二者分别由式 (4) 、 (5) 定义： 

**==> picture [187 x 47] intentionally omitted <==**

## **1.2** 挖洞阶段 

**==> picture [251 x 48] intentionally omitted <==**

--- end of page.page_number=2 ---

> 其中， _Zb_ ( _t_ ) 为种群中随机选择的搜索个体， _L_ 为与当前迭 代次数相关的随机数： 

**==> picture [172 x 26] intentionally omitted <==**

## **1.3** 觅食阶段 

旅鼠在觅食区域内的随意游荡行为通过螺旋缠绕机 制建模，位置更新公式见式 (8) ： _Zi_ ( _t_ + 1) = _Zbest_ ( _t_ ) + _F_  _spiral_  _rand_  _Zi_ ( _t_ ) (8) 其中， _spiral_ 为表征觅食轨迹螺旋形状的项， _radius_ 为当 前位置与最优解之间的欧几里得距离，表示觅食范围的半 径，二者分别由式 (9) 、 (10) 给出： 

**==> picture [216 x 43] intentionally omitted <==**

## **1.4** 躲避天敌阶段 

旅鼠借助欺骗性动作逃避捕食者的追击，数学表达式 如式 (11) 所示： 

_Zi_ ( _t_ + 1) = _Zbest_ ( _t_ ) + _F_  _G_  _Levy_ ( _Dim_ )  ( _Zbest_ ( _t_ ) − _Zi_ ( _t_ )) (11) 其中， _G_ 为旅鼠的逃逸系数，随迭代次数增加而减小；  _Levy_ ( ) 为莱维飞行函数，用于模拟旅鼠逃跑时的欺骗动作， 二者分别由式 (12) 、 (13) 定义： 

**==> picture [194 x 92] intentionally omitted <==**

其中 , _Tmax_ 为最大迭代次数， _u_ 、 _v_ 为 [0,1] 内的随机数， 为常数，取值 1.5 。 

## **1.5** 能量因子 

为平衡勘探与开发，引入随迭代过程减小的能量因子 _E_ ，其表达式为： 

**==> picture [187 x 29] intentionally omitted <==**

当 _E_  1 时，若 _rand_  0.3 执行迁徙，否则执行挖洞； 当 0  _E_  1 时，若 _rand_  0.5 则执行觅食，否则执行躲避天 敌。 

## **2** 改进的人工旅鼠算法 

针对人工旅鼠算法收敛精度不足、易陷入局部最优及 历史搜索信息利用率低的问题，提出一种改进的人工旅鼠 算法（ IALA ）。首先，将种群划分为 _k_ 个独立子群，各子 群独立寻优并通过子群间环形迁移交换最优个体，以维护 种群多样性、提高求解质量。其次，依据种群熵值自适应 

调整种群规模，平衡算法的探索与开发能力。最后，构建 跨子群精英档案，以递减的回忆概率将历史精英解注入随 机子群，充分利用历史搜索信息，提升全局收敛能力与求 解精度。 

## **2.1** 子群划分与环形迁移 

ALA 的四个寻优阶段均以全局最优解为基准更新个 体位置，在多峰函数上一旦陷入局部最优，全体个体的搜 索方向随之偏转，难以探索其他峰值区域。 

针对上述不足， IALA 借鉴岛屿模型的多种群并行思 想[[13,14]] ，将规模为 _N_ 的种群均匀划分为 _k_ 个独立子群，各 子群独立维护自身个体并分别完成位置更新与适应度评 估，再通过子群间环形迁移周期性交换最优个体，扩大解 空间的搜索覆盖范围，缓解单一种群陷入局部最优。各子 群规模见式 (15) ： 

**==> picture [251 x 59] intentionally omitted <==**

为在子群间共享信息， IALA 采用环形迁移策略，即 第 _i_ 个子群的最优个体替换第 _i_ + 1 个子群中的最差个体， 末尾子群的最优个体迁移至第 1 个子群，各子群首尾相连 形成单向环形结构。迁移周期  _t_ 随迭代进度动态调整，定 义见 (16) ： 

**==> picture [185 x 79] intentionally omitted <==**

当全局最优解连续 30 代适应度改善量低于 10⁻⁶ 时立 即触发迁移。 

## **2.2** 基于信息熵的种群规模动态调整 

种群规模直接影响算法的搜索性能， ALA 在演化全 程采用固定种群规模 _N_ 更新个体位置，种群规模不随演化 进程改变。针对上述不足， IALA 引入信息熵作为种群多 样性的度量指标[[15]] ，并依据熵值动态调整种群规模。演化 前期熵值较高，算法扩大种群规模以增强全局探索；演化 后期熵值降低、种群趋于聚集，算法缩减种群规模以加速 向局部最优收敛。 

算法首先将种群个体适应度归一化为概率分布，如下： − = _fi f_ min _pi f_ max − _f_ min + (17) 其中， _fi_ 为第 _i_ 个旅鼠的适应度值， _f_ min 和 _f_ max 分别为当 前最小、最大的适应度值，  为极小的数。 

--- end of page.page_number=3 ---

在此基础上，按 Shannon 信息熵定义计算种群熵值， 如式 (18) 所示： 

_N H_ = − _i_ = 1 _pi_  log( _pi_ + ) (18) 其中， _H_ 为种群熵值，反映种群个体分布的均匀程度， 值越大表明分布越均匀、种群多样性越充分。 

考虑到单代熵值易受随机扰动产生瞬时波动， IALA 对最近 _w_ 代熵值进行算术平均得到平滑熵 _H smooth_ ，本文取 _w_ = 5 ，由式 (19) 给出： 

**==> picture [166 x 22] intentionally omitted <==**

_H_ 最后，算法以熵比率 _smpoth_ 为依据动态调整种群规 log( _Ncurrent_ ) 模，更新公式见式 (20) ： 

**==> picture [231 x 94] intentionally omitted <==**

其中， _Ncurrent_ 、 _N_ new 分别为调整前后的种群规模； _N_ min 、 _N_ max 为种群规模的下界与上界，分别取 10 和 2 _N_ 。 

## **2.3** 跨子群精英解回溯 

标准 ALA 未对历史搜索过程中产生的优秀解加以保 存与利用，在迭代后期种群趋于聚集，个体间差异减小， 搜索步长趋近于零，加剧种群的过早收敛。 

记忆回溯策略受记忆机制启发，通过思考、回忆与记 忆三个阶段为算法引入群体记忆，复用历史优质解以提升 算法的收敛精度[[16]] 。借鉴其回忆机制， IALA 在多子群结 构下构建跨子群精英档案，将历代全局最优解保存至档案， 并以一定概率重新注入种群，使历史精英解持续参与种群 更新，维持后期的种群多样性。 

档案容量 _M_ 与精英档案 _M_ ( _t_ ) _M_  _D_ 分别如式 (21) 与式 

(22) 所示 ; 

_M_ = max(50,1.5  _N_ ( _t_ ) ) (21) _M_ ( _t_ ) _M_  _D_ =  _m_ 1, ( _t_ ) , _m_ 2 , ( _t_ ) ,  , _mM_ ( _t_ ) (22) 其中， _M_ 为档案容量，随种群规模 _N_ ( _t_ ) 动态调整，下 界设为 50；档案初始化为零矩阵，内部适应度值初始 化为正无穷； _D_ 为问题维度。 

> 回忆阶段，算法以回忆概率 _p_ 从档案中随机抽取一个 精英解，注入随机子群以替换其中的随机个体，更新规则 如式 (23) 所示： 

**==> picture [195 x 29] intentionally omitted <==**

_t_ 其中， _r_ 为 [0,1] 的随机数；回忆概率 _p_ = 0.1(1 − ) ，随迭 _T_ 代次数线性衰减。 

记忆阶段，算法在每次迭代结束时将当前全局最优解 存入档案。档案未满时直接存入，已满时若其优于档案中 最差解则替换该最差解，否则档案保持不变，数学表达式 见式 (24) ： 

**==> picture [249 x 24] intentionally omitted <==**

其中， 𝑀𝑤𝑜𝑟𝑠𝑡(𝑡) 为第 _t_ 轮迭代时档案中最差的解。 

## **2.4IALA** 算法流程 

IALA 实现的算法流程如图 1 所示。 

**==> picture [182 x 298] intentionally omitted <==**

图 1IALA 算法流程图 

Fig. 1Flowchart of the IALA algorithm 

## **2.5** 时间复杂度分析 

时间复杂度是算法运行效率的重要体现。假设种群规 模为 _N_ ，最大迭代次数为 _T_ ，问题维度为 _D_ 。 

在 ALA 的初始化阶段，其时间复杂度可表示为 _O_ (1) 。 随后，在每一次迭代中， _N_ 个旅鼠个体各自进行一次位置 更新操作，经过 _T_ 轮迭代后，该过程累积的时间复杂度为 _O TND_ ( ) 。期间对个体进行适应度评估的时间复杂度为 _O_ ( _TN_ ） 。 ALA 的时间复杂度为 _O_ (1) + _O T_ ( _ND_ ) + _O TN_ ( ） ，简 

--- end of page.page_number=4 ---

## 化为 _O TND_ ( ) 。 

## _O TND_ ( ) 。 

对于 IALA ，初始化阶段除生成 _N_ 个 _D_ 维个体外，还 需划分 _k_ 个独立子群并构建容量为 _C_ 的精英档案，与 ALA 初始化同阶。 IALA 虽改进了寻优机制，但每个旅鼠个体 **2.6IALA** 在每轮迭代中依然仅进行一次位置更新，未引入额外循环 选取新近提出的 结构，位置更新与适应度评估的时间复杂度仍为 _O TND_ ( ) 的 WOA[[[19]]] 、 GWO 与 _O_ ( _TN_ ) 。子群并行搜索仅重组种群的组织方式，各子群 数集上验证 IALA 规模之和仍为 _N_ ，不引入额外开销。环形迁移在 _k_ 个子群 CEC2022 中线性查找最优与最差个体；基于信息熵的种群规模调整 对 _N_ 个适应度做归一化并计算熵值；跨子群精英存档在容 F1 为单峰函数， 量 _C_ = max(501.5， _N_ ) 的档案中线性查找最差解并替换，三 F9~F12 项操作单次迭代的时间复杂度均为 _O_ ( _N_ ) ，经过 _T_ 轮迭代 累计为 _O_ ( _TN_ ) ，被位置更新的 _O TND_ ( ) 吸收。综上， IALA 在 dim=20 总的时间复杂度为 _O TND_ ( ) + _O TN_ ( ) + _O TN_ ( ) ，简化为 500 表 1 CEC2022 测试集算法对比结果 (dim=20) 

由上述分析可知， IALA 与 ALA 时间复杂度同阶，三 处改进策略未改变算法的时间复杂度阶数。 

## **2.6IALA** 性能测试 

选取新近提出的 ALA[[12]] 、 BKA[[17]] 、 HO[[18]] ，以及经典 的 WOA[[[19]]] 、 GWO[[20]] 作为对比算法，在 CEC2022 测试函 数集上验证 IALA 的寻优能力。 

CEC2022 测试集由 IEEE 进化计算大会提出，共含 12 个测试函数，按结构分为单峰、多峰、混合与组合四类。 F1 为单峰函数， F2~F5 为多峰函数， F6~F8 为混合函数， F9~F12 为组合函数，可分别考察算法的收敛速度、跳出 局部最优的能力以及求解高复杂度问题的性能。实验统一 在 dim=20 下进行，种群规模设为 30 ，最大迭代次数设为 500 ，各算法独立运行 30 次后取平均值。 

Table 1 CEC2022 test set comparison results (dim=20) 

|Function||ALA|WOA|GWO|BKA|HO|IALA|
|---|---|---|---|---|---|---|---|
|F1|std|**1.24E+03**|1.70E+04|5.00E+03|1.03E+04|8.48E+03|7.46E+03|
||avg|**2.96E+03 −**|4.09E+04 +|1.65E+04 +|7.92E+03 −|2.46E+04 +|1.16E+04|
|F2|std|1.91E+01|1.07E+02|3.75E+01|2.99E+02|5.29E+01|**1.75E+01**|
||avg|5.56E+01 =|2.48E+02 +|9.86E+01 +|2.17E+02 +|1.33E+02 +|**5.18E+01**|
|F3|std|2.38E+00|1.30E+01|3.31E+00|9.18E+00|9.19E+00|**2.66E+00**|
||avg|3.62E+00 +|7.30E+01 +|6.22E+00 +|5.50E+01 +|5.49E+01 +|**2.77E+00**|
|F4|std|2.21E+01|3.21E+01|2.44E+01|1.64E+01|1.05E+01|**2.09E+01**|
||avg|6.11E+01 =|1.44E+02 +|6.31E+01 =|7.60E+01 +|8.07E+01 +|**5.61E+01**|
|F5|std|1.88E+02|1.61E+03|3.03E+02|4.70E+02|3.34E+02|**2.23E+02**|
||avg|2.14E+02 =|3.56E+03 +|2.92E+02 =|1.29E+03 +|1.35E+03 +|**1.98E+02**|
|F6|std|8.22E+03|2.71E+07|8.73E+06|1.49E+07|9.39E+03|**7.88E+03**|
||avg|1.24E+04 +|1.10E+07 +|2.81E+06 +|3.92E+06 =|8.21E+03 =|**7.22E+03**|
|F7|std|3.41E+01|7.17E+01|3.62E+01|3.20E+01|2.69E+01|**4.06E+01**|
||avg|8.41E+01 =|2.21E+02 +|8.38E+01 =|1.23E+02 +|1.40E+02 +|**7.59E+01**|
|F8|std|**4.42E+00**|1.23E+02|6.82E+01|9.34E+01|1.29E+01|3.89E+01|
||avg|**3.23E+01 =**|1.40E+02 +|7.55E+01 =|9.61E+01 +|4.66E+01 =|4.78E+01|
|F9|std|9.56E-02|4.60E+01|3.02E+01|8.38E+01|4.51E+01|**3.63E-04**|
||avg|1.81E+02 +|3.04E+02 +|2.17E+02 +|2.33E+02 +|2.61E+02 +|**1.81E+02**|
|F10|std|8.39E+02|1.34E+03|**7.55E+02**|1.14E+03|1.13E+03|8.45E+02|
||avg|1.48E+03 +|2.44E+03 +|**1.06E+03 =**|2.01E+03 +|1.77E+03 +|1.08E+03|
|F11|std|1.01E+01|4.17E+02|5.32E+02|1.16E+03|1.65E+02|**2.42E-03**|
||avg|3.16E+02 +|1.38E+03 +|1.12E+03 +|1.61E+03 +|4.78E+02 +|**3.00E+02**|
|F12|std|4.56E+01|8.23E+01|2.41E+01|1.24E+02|1.12E+02|**2.57E+01**|
||avg|2.67E+02 =|3.89E+02 +|2.82E+02 +|3.75E+02 +|4.31E+02 +|**2.63E+02**|
|F_av||2.08|5.83|3.08|4.42|4.17|1.42|
|RANK||2|6|3|5|4|1|



--- end of page.page_number=5 ---

## **2.6.1** 均值和方差 

表 1 给出了六种算法在 12 个函数上的均值与标准差， 均值反映算法的收敛精度，标准差反映其运行稳定性， F_av 为各算法在 Friedman 检验下的平均排名值， RANK 为据此得到的最终排名。各算法的 F_av 依次为 2.08 、 5.83 、 3.08 、 4.42 、 4.17 和 1.42 ，对应排名分别为第 2 、 6 、 3 、 5 、 4 和 1 ， IALA 排名第一。 

单峰函数 F1 上， IALA 的均值与标准差均劣于 ALA 和 BKA 。 IALA 的改进策略以维持种群多样性、跳出局部 最优为出发点，在不存在多个局部最优的单峰场景下作用 有限。多峰函数 F2-F5 上， IALA 各项指标均取得最优， 得益于子群划分与环形迁移保留了多样化的搜索方向，使 其在多峰场景下能有效跳出局部最优。混合函数 F6-F8 上， IALA 在 F6 、 F7 上表现最优， F8 稍逊于 ALA 但差距不 大，基于信息熵的种群规模动态调整在其中平衡了全局探 索与局部开发。组合函数 F9-F12 上， IALA 在 F9 、 F11 、 F12 上效果最优， F10 略逊于 GWO 但优于其余算法，跨 子群精英存档通过历代精英解回注，维持了后期的种群多 样性，提升了求解精度 

## **2.6.2** 收敛图 

图 1 给出了 CEC2022 测试集 20 维条件下各算法的迭 代收敛曲线。所选 F2 、 F4 、 F6 、 F7 、 F10 和 F12 共 6 个代 表性测试函数涵盖了不同复杂度的优化问题，其中 F2 、 F4 为多峰函数， F6 、 F7 为混合函数， F10 、 F12 为组合函数， 

能够反映各算法求解不同类型问题时的收敛速度和寻优 能力。 

从图 1 可知，在 F2 上， WOA 、 BKA 和 HO 等算法在 迭代中期后陷入停滞， ALA 和 IALA 仍持续下降，且 IALA 收敛精度更高。在 F4 上， BKA 和 HO 前期收敛较快但中 后期趋于停滞。 IALA 初期收敛较慢，但中后期寻优能力 更强，最终取得更优的收敛结果。在 F6 上， IALA 前期即 快速逼近最优值，全程保持领先，最终收敛到最低的适应 度值。 

在 F7 上， WOA 、 BKA 、 HO 于迭代中期相继陷入停 滞， ALA 后期仍保持下降趋势， IALA 则始终保持领先， 最终收敛效果最佳。在 F10 上， IALA 前期即与其他算法 拉开明显差距，收敛曲线平滑下降。虽然 GWO 在迭代末 段略优于 IALA ，但二者最终收敛值相近。在 F12 上， IALA 、 ALA 、 GWO 在前 100 次迭代内即领先于 WOA 、 BKA 、 HO ，其中 HO 收敛效果相对较差， IALA 收敛到最低的适 应度值。 

综合来看， IALA 在不同类型函数上均保持了较为稳 定的收敛性能，尤其在进化中后期仍能保持较快的收敛速 度，未出现明显的过早收敛。相比之下， ALA 的收敛曲线 在所有函数上均位于 IALA 上方，其收敛速度与求解精度 均不及 IALA ，说明子群划分与环形迁移、基于信息熵的 种群规模动态调整以及跨子群精英存档三种策略的引入 改善了善法的性能。 

**==> picture [391 x 97] intentionally omitted <==**

**==> picture [383 x 148] intentionally omitted <==**

**----- Start of picture text -----**<br>
(a) F 2 (b) F 4 (c) F 6<br>(d) F 7 (e) F 10 (f) F 12<br>图  2  算法收敛曲线图<br>**----- End of picture text -----**<br>


Fig. 2Convergence Curves of Each Algorithm 

**==> picture [90 x 28] intentionally omitted <==**

**----- Start of picture text -----**<br>
3 漂绿数据集<br>3.1 “漂绿”指标<br>**----- End of picture text -----**<br>


参照杨七中等[[7]] 的研究思路，将“漂绿”界定为企业 年报中环境议题的口头披露强度（ Oral ）与其实际环境表 

--- end of page.page_number=6 ---

现（ Actual ）之间的不一致，并通过文本分析法对该指标 加以量化。具体测度流程如下。 

## **3.1.1** 构造种子词集 

沿用杨七中等整理的 20 个环境信息种子词作为反映 环境议题的初始词集。 

## **3.1.2** 生成拓展词集 

由于初始种子词数量较少，难以涵盖企业环境信息披 露中出现的多样化表述，需对其进行语义层面的扩充。本 文借助大语言模型，按语义相关性对种子词进行同义、近 义及上下位扩展，去重后得到包含 200 个词语的拓展词集， 涉及低碳发展、污染减排、清洁能源、循环经济等议题， 部分结果见表 2 。 

表 2 种子词集和拓展词集部分结构 

Table 2 Partial results of seed word set and expanded word set 

||||
|---|---|---|
|语料库|种子词集|拓展词集|
|MD&A<br>文本、<br>企业网<br>站文本|绿色、环境保护、低<br>碳、污染、减排、节<br>能、生态、循环、清<br>洁、蓝天、绿水、转<br>型、再生、降耗……|空气、颗粒物、细颗粒物、挥发性有<br>机物、偷排、废气、脱硫、脱硝、危<br>废处置、环保投资、地下水、风电、<br>可再生能源、新能源、碳中和、碳交<br>易、绿色金融、绿水、重金属……|



## **3.1.3** 计算企业环境信息关注度 

TF-IDF 能够兼顾词语在单篇文档中的频次与其在整 体语料中的区分能力，因此选用该算法计算权重。对于拓 展词集中第 _x_ 个词语，计算其在第 _y_ 家企业 MD&A 及官 网文本中的 TF-IDF 权重 _wxy_ ，再将各词权重加总得到该 企业的环境信息关注度，如式（ 25 ）所示： 

**==> picture [214 x 28] intentionally omitted <==**

> 其中， _EnvINF_ 为企业信息环境关注度； _wxy_ 为拓展词集中 

第 _x_ 个词语的权重； _tf xy_ 是第 _x_ 个词语在第 _y_ 家企业的词 频； _df x_ 是含有第 _x_ 个词语的文档数； _N_ 是全部文档总数。 **3.1.4** 测度“漂绿”指标 

针对各企业年度观测值： 

步骤 **1** 若某企业的环境信息关注度在同行业同年度 排名处于前 10% ，则 Oral 赋值为 1 ，否则为 0 ； 

步骤 **2** 若该企业当年受到生态环境主管部门的行政 处罚，则 Actual 赋值为 1 ，否则为 0 ； 

步骤 **3** 仅当 Oral 与 Actual 同时为 1 时， Greenwash 取 值为 1 ，该样本被识别为“漂绿”样本，其余情形下 Greenwash 取值为 0 。 

## **3.2** 数据来源 

究样本，并作如下处理：（ 1 ）剔除金融行业企业；（ 2 ）剔 除被特别处理或已退市的企业；（ 3 ）剔除资产负债率高于 100% 的企业；（ 4 ）剔除关键变量存在缺失的样本；（ 5 ）剔 除未披露年度报告或披露年份不足 5 年的企业样本。本文 所用数据来源于 CSMAR 国泰安数据库、 CCER 色诺芬数 据库及巨潮资讯网。 

## **3.3** 特征变量 

准确识别“漂绿”行为，需要从其形成机制出发系统 组织特征变量。“漂绿”的发生受企业财务状况、治理结 构、外部约束与绿色实质投入等多重因素驱动，不同因素 对应不同维度的可观测指标。本文从财务水平、内部治理、 外部治理与绿色经营 4 个维度共选取 27 个特征变量。 

第一，财务状况是驱动企业披露决策的重要因素。企 业在融资约束加剧或业绩承压时，倾向于通过策略性环境 信息披露塑造正向形象，以获取低成本绿色资金[[2]] 。本文 选取资产规模、资产负债率、现金流量、应收账款、资产 报酬率、资产周转率、收入增长率、融资约束与信息透明 度共 8 个变量。 

第二，内部治理是约束管理层“漂绿”机会主义行为 的事前与事中制衡机制，高管特征与内部监督对企业“漂 绿”行为具有直接抑制作用[[21]] 。本文选取高管薪酬、高管 年龄、上市年限、大股东持股比、董事会规模、独立董事 占比、企业性质、两职合一、大股东资金占用与内控水平 共 11 个变量。 

第三，外部治理通过资本市场与信息中介对“漂绿” 行为形成事后矫正，投资者关注水平的提升与机构投资者 绿色偏好的增强均能显著抑制企业的策略性环境信息披 露[[22]] 。本文选取分析师关注度、机构持股比例、股票收益 波动率与市净率共 4 个变量。 

第四，绿色经营反映企业的环境监管情境与绿色行动 的实质水平，高管绿色经历与绿色专利产出能够直接刻画 企业的绿色实质投入，与符号化的环境披露形成对照[[7,23]] 。 本文选取重污染行业、高科技行业、高管绿色经历与绿色 专利数共 4 个变量。 

最终选取的特征变量及其定义见表 3 。 

## **3.4** 数据预处理 

首先，依据 3.1 节方法对 2015—2024 年的样本进行 “漂绿”指标测度，共识别出 656 个“漂绿”样本，并按 1∶5 的比例随机抽取 3280 个正常样本作为对照，共 3936 个标注样本。其次，将上述标注样本按 8∶2 的比例划分为 训练集与测试集，用以训练模型并评估其分类性能。再次， 为削弱极端值的干扰，本文对所有连续变量在 1% 与 99% 分位数上进行缩尾处理，并通过归一化消除量纲差异。 

本文选取 2015—2024 年我国沪深 A 股上市企业为研 

--- end of page.page_number=7 ---

表 3 变量定义 

Table 3 Variable definitions 

|Table 3 Variable definitions|Table 3 Variable definitions|Table 3 Variable definitions|
|---|---|---|
|维度<br>变量名<br>定义|||
|财务状况|资产规模<br>资产负债率<br>现金流量<br>应收账款<br>资产报酬率<br>资产周转率<br>收入增长率<br>融资约束|ln(年末总资产)<br>年末总负债/总资产<br>经营活动现金流量净额/营业收入<br>应收账款/总资产<br>净利润/总资产<br>营业收入/总资产<br>主营业务收入增长率<br>SA 指数|
|内部治理|高管薪酬<br>高管年龄<br>上市年限<br>信息透明度<br>大股东持股比<br>董事会规模<br>独立董事占比<br>企业性质<br>两职合一<br>大股东资金占用<br>内控水平|ln(前三名高管薪酬总额)<br>董监高平均年龄<br>企业上市至今的年份时长<br>沪深证券交易所年度信息披露质量评级<br>第一大股东持股比例<br>ln(董事会人数)<br>独立董事人数/董事会人数<br>国有=1，其他=0<br>董事长兼总经理=1，否则=0<br>其他应收款/总资产<br>迪博内部控制指数|
|外部治理|分析师关注度<br>机构持股比例<br>股票收益波动率<br>市净率|ln(1+被分析师关注次数)<br>机构持股数/总股本<br>月个股收益率年内标准差<br>股价/每股净资产|
|绿色经营|重污染行业<br>高科技行业<br>高管绿色经历<br>绿色专利数|16类重污染行业=1，否则=0<br>7类高科技行业=1，否则=0<br>高管有绿色经历=1，否则=0<br>ln(1+集团绿色专利累积数)|



## **4 SMOTE-Tomek-IALA-CatBoost** “漂绿”识别 模型 

## **4.1 CatBoost** 模型 

CatBoost 是一种基于梯度提升决策树的集成学习模 型。与传统梯度提升方法相比， CatBoost 在基学习器构造、 迭代更新和类别特征处理方面进行了改进，能够较好刻画 特征变量之间的非线性关系与交互关系。对于上市公司 “漂绿”识别任务而言，财务状况、内部治理、外部治理 和绿色经营等变量之间可能存在复杂组合影响， CatBoost 可通过多轮迭代不断修正前一轮模型的识别偏差，从而完 成对样本类别的判别。 

CatBoost 的关键技术之一是有序目标统计量。该方法 在计算样本统计值时引入随机排列顺序，使当前样本仅依 赖其之前的历史样本信息，从而降低目标泄露对模型训练 的影响。样本目标统计值计算公式为： 

**==> picture [179 x 47] intentionally omitted <==**

> _i i_ 其中： _[x] j_[、] _[x] k_[、] _[x] j_[、] _[y] i_[均为训练样本相关变量；] _Dk_ 为第 _k_ 个样本之前的数据集； _a_ 为权重系数； _P_ 为先验值。 

## **4.2 SMOTE-Tomek-IALA-CatBoost** 模型流程 

针对上市公司“漂绿”样本数量较少、类别分布不均 衡而导致模型对少数类识别不足的问题，本章提出一种 SMOTE-Tomek-IALA-CatBoost “漂绿”识别模型。模型先 利用 SMOTE 在少数类样本邻域内生成新的“漂绿”样本， 缓解样本数量不足对模型训练的影响。随后采用 Tomek Links 删除类别边界附近的重叠样本，使训练数据的类别 边界更加清晰。最后，引入 IALA 对 CatBoost 关键参数进 行寻优，提高参数组合与样本分布的适配程度。模型流程 如图 3 所示，具体步骤如下。 

步骤 **1** 数据预处理。对数据进行缺失值处理、缩尾处 理、归一化处理并划分训练集与测试集。分别用 SMOTE 、 Tomek Links 、 SMOTE-Tomek 、 ROS 、 RUS 、 NearMiss 、 ADASYN 这 7 种不平衡数据处理方法对训练样本进行预 处理。 

步骤 **2** 设置参数。确定 CatBoost 调参范围，并设置 IALA 的种群规模、最大迭代次数等。 

步骤 **3** 超参数寻优。挑选对 CatBoost 模型性能影响 较大的迭代次数、树深度、学习率和 L2 正则化系数 4 个 参数作为寻优对象，利用 IALA 搜索最优参数组合。 

步骤 **4** 模型效果检验。以召回率、精确率、 F1 、准确 率、 AUC 作为评判模型效果的指标，分别对比步骤 1 中 7 种样本处理方法在 CatBoost 、 XGBoost 、 GBDT 、 SVM 、 RF 、 IALA-CatBoost 模型上的效果。 

## **4.3** 不平衡样本采样技术比较 

上市公司“漂绿”样本在全体样本中占比较小，类别 分布的不均衡性使模型训练易偏向多数类，削弱对少数类 “漂绿 ’ 样本的识别能力。为系统考察采样技术对识别效 果的影响，选取 SMOTE 、 ROS 、 ADASYN 、 RUS 、 NearMiss 、 Tomek Links 及 SMOTE-Tomek 共 7 种采样方法，分别与 RF 、 GBDT 、 XGBoost 、 SVM 及 CatBoost5 种模型组合， 以准确率、精确率、召回率、 F1 值和 AUC 为评价指标进 行综合比较，结果见表 4 。 

由表 4 可知，过采样与组合采样方法整体优于欠采样 方法。在 ROS 过采样技术下，各模型的识别效果最好， 准确率均在 91% 以上、 AUC 均在 97 以上。但 ROS 通过 随机重复复制少数类样本实现类别均衡，并不引入新的样 本信息 , 较高指标可能受样本重复影响，存在评价结果偏 乐观和过拟合的风险。与 ROS 不同， SMOTE-Tomek 先在 少数类样本邻域内合成新样本，再清除类别边界附近的重 叠样本，在样本扩充与边界修正之间更为稳妥，且综合表 现仅次于 ROS 。 

相比之下，欠采样技术下模型性能普遍较差， RUS 与 NearMiss 处理后各模型准确率多降至 75% 以下。 Tomek Links 处理后模型偏向多数类，召回率明显偏低，如 Tomek 

--- end of page.page_number=8 ---

Links-CatBoost 仅为 39.69% ，表明欠采样在压缩多数类样 本的同时易造成正常企业样本的关键信息损失。综合各方 

法的指标表现与采样机理，选取 SMOTE-Tomek 作为后续 模型训练的样本不平衡处理方法。 

**==> picture [478 x 195] intentionally omitted <==**

图 3 SMOTE-Tomek-IALA-CatBoost 漂绿识别模型流程图 

Fig. 3 Flowchart of the SMOTE-Tomek-IALA-CatBoost greenwashing identification model 

## **4.3** 不平衡样本采样技术比较 

上市公司“漂绿”样本在全体样本中占比较小，类别 分布的不均衡性使模型训练易偏向多数类，削弱对少数类 “漂绿 ’ 样本的识别能力。为系统考察采样技术对识别效 果的影响，选取 SMOTE 、 ROS 、 ADASYN 、 RUS 、 NearMiss 、 Tomek Links 及 SMOTE-Tomek 共 7 种采样方法，分别与 RF 、 GBDT 、 XGBoost 、 SVM 及 CatBoost5 种模型组合， 以准确率、精确率、召回率、 F1 值和 AUC 为评价指标进 行综合比较，结果见表 4 。 

由表 4 可知，过采样与组合采样方法整体优于欠采样 方法。在 ROS 过采样技术下，各模型的识别效果最好， 准确率均在 91% 以上、 AUC 均在 97 以上。但 ROS 通过 随机重复复制少数类样本实现类别均衡，并不引入新的样 本信息 , 较高指标可能受样本重复影响，存在评价结果偏 乐观和过拟合的风险。与 ROS 不同， SMOTE-Tomek 先在 少数类样本邻域内合成新样本，再清除类别边界附近的重 叠样本，在样本扩充与边界修正之间更为稳妥，且综合表 现仅次于 ROS 。 

相比之下，欠采样技术下模型性能普遍较差， RUS 与 NearMiss 处理后各模型准确率多降至 75% 以下。 Tomek Links 处理后模型偏向多数类，召回率明显偏低，如 Tomek Links-CatBoost 仅为 39.69% ，表明欠采样在压缩多数类样 本的同时易造成正常企业样本的关键信息损失。综合各方 法的指标表现与采样机理，选取 SMOTE-Tomek 作为后续 模型训练的样本不平衡处理方法。 

## **4.4** “漂绿”现象识别结果分析 

基于 SMOTE-Tomek 处理后的样本，对各模型的“漂 

绿”识别结果进行比较，结果见表 4 。 

由表 4 可知， RF 的测试样本识别准确率为 89.74% ， 精确率为 89.43% ，召回率为 90.12% ， F1 值为 89.78% ， AUC 值为 96.66% ，各项指标在所有模型中处于较低水平。 SVM 通过最大化分类间隔进行决策，对少数类样本的边 界较为敏感，易将更多样本判定为“漂绿”，导致精确率明 显低于召回率，其准确率虽达到 93.21% ，但精确率仅为 89.77% 。 

GBDT 、 XGBoost 与 CatBoost 三种模型的识别效果明 显优于 RF 与 SVM 。其中， GBDT 与 XGBoost 的各项指 标较为接近， GBDT 的准确率和精确率分别为 93.67% 和 94.08% ， XGBoost 的准确率和精确率分别为 92.59% 和 93.81% 。相比之下， CatBoost 的表现更优，其准确率、精 确率、召回率、 F1 值和 AUC 值分别为 94.07% 、 94.20% 、 93.91% 、 94.06% 和 98.63% 。上述三种模型均采用梯度提 升机制，通过多轮迭代逐步修正前一轮的识别偏差，对复 杂非线性关系的拟合能力相对更强。其中 CatBoos 采用有 序提升策略缓解预测偏移，并能直接处理数据中的类别型 特征，减少预处理带来的信息损失，在所有模型中表现最 好。 

由于 CatBoost 在上述模型中表现最优，进一步采用 IALA 算法对其超参数进行寻优。经优化后， IALACatBoost 的准确率、精确率、召回率、 F1 值和 AUC 值分 别为 94.52% 、 94.73% 、 94.29% 、 94.51% 和 98.72% ，各项 指标较 CatBoost 均有所提高，且精确率与召回率同步上 升，表明优化后的模型在减少误判与降低漏判两方面同时 改善，识别结果更为均衡，对“漂绿”样本的识别能力进 

--- end of page.page_number=9 ---

## 一步提升。 

## 表 4 不同采样方法下模型识别效果比较 

Table 4 Comparison of model identification performance under different sampling methods 

|采样方法|模型|Accuracy|Precision|Recall|F1|AUC|
|---|---|---|---|---|---|---|
||RF|90.02|90.08|89.94|90.01|96.59|
||GBDT|93.29|94.94|91.46|93.17|97.91|
|SMOTE|XGBoost|92.15|93|91.16|92.07|97.69|
||SVM|92.84|88.92|97.87|93.18|96.89|
||CatBoost|93.9|94.72|92.99|93.85|98.67|
||RF|91.08|87.48|95.88|91.49|97.19|
||GBDT|95.5|92.46|99.09|95.66|99.6|
|ROS|XGBoost|93.29|90.11|97.26|93.55|98.57|
||SVM|92.45|87.89|98.48|92.88|97.53|
||CatBoost|95.5|92.7|98.78|95.65|99.65|
||RF|90.14|89.03|91.14|90.07|96.67|
||GBDT|92.31|93.19|90.98|92.07|97.97|
|ADASYN|XGBoost|92.78|94.84|90.19|92.46|97.77|
||SVM|92.62|88.19|98.1|92.88|97.11|
||CatBoost|93.48|94.48|92.09|93.27|98.43|
||RF|73.38|73.28|73.28|73.28|78.02|
||GBDT|73|73.08|72.52|72.8|78.27|
|RUS|XGBoost|71.86|71.76|71.76|71.76|78.3|
||SVM|71.86|72.44|70.23|71.32|77.83|
||CatBoost|70.72|70.45|70.99|70.72|80.04|
||RF|73.76|75|70.99|72.94|79.43|
||GBDT|73.76|74.6|71.76|73.15|79.4|
|NearMiss|XGBoost|75.67|77.24|72.52|74.8|80.81|
||SVM|74.9|74.81|74.81|74.81|79.37|
||CatBoost|74.52|76.23|70.99|73.52|81.84|
||RF|83.92|67.65|17.56|27.88|80.23|
||GBDT|83.51|55.7|33.59|41.9|79.46|
|TomekLinks|XGBoost|85|64.71|33.59|44.22|81.51|
||SVM|81.76|48.44|47.33|47.88|78.49|
||CatBoost|85.95|67.53|39.69|50|81.19|
||RF|89.74|89.43|90.12|89.78|96.66|
||GBDT|93.67|94.08|93.21|93.64|97.8|
||XGBoost|92.59|93.81|91.2|92.49|97.94|
|SMOTE-Tomek|||||||
||SVM|93.21|89.77|97.53|93.49|96.97|
||CatBoost|94.07|94.2|93.91|94.06|98.63|
||IALA-CatBoost|94.52|94.73|94.29|94.51|98.72|



## **4.5 SHAP** 全局动因挖掘 

为挖掘影响企业“漂绿”的共性动因，采用 SHAP 方 法度量 SMOTE-Tomek-IALA-CatBoost 模型中各特征对识 别结果的贡献程度，并据此对特征的相对重要性进行排序。 

由图 4(a) 和表 5 可知，绿色专利数、资产规模的相对 重要性居于前列，分别为 10.48% 和 7.34% 是影响“漂绿” 识别的关键特征。结合图 4(b) 可知，绿色专利较多、资产 规模较大的企业，反而更容易被模型识别为“漂绿”。这一 现象与一般认知存在差异，实则反映出此类企业往往具有 

更强的绿色形象塑造动机和披露能力，其高调的绿色信号 反而更易掩盖实质性绿色行动的不足，成为“漂绿”的高 发对象。与之类似，资产负债率越高的企业越倾向于被判 定为“漂绿”，表明较大的财务压力可能驱使企业以绿色 叙事粉饰经营状况。 

从特征维度来看（见表 6 ），内部治理与绿色经营是企 业“漂绿”的主要动因，累计相对重要性分别为 36.59% 和 30.01% 。内部治理薄弱往往为企业选择性披露提供空间， 是 ESG 漂绿的重要诱因[[24]] 。绿色经营贡献靠前，则源于 

--- end of page.page_number=10 ---

部分企业以绿色宣传塑造形象、却缺乏相应的实质性投入 [25] 。财务状况也具有一定的解释力，累计相对重要性为 

**==> picture [260 x 202] intentionally omitted <==**

## 26.17% ，外部治理的解释力则相对较弱。 

**==> picture [198 x 202] intentionally omitted <==**

(a) 基于 SHAP 均值的特征重要性排序 (b) 各特征 SHAP 值正负分布 

图 4 基于 SHAP 的特征可解释性分析（ Top 20 ） 

Fig. 4 Feature interpretability analysis based on SHAP (Top 20) 

表 5 特征变量相对重要性排序结果 ( 单位 :%) 

Table 5 Ranking Results of Relative Importance of Feature 

|||Variables|(Unit:|%)||
|---|---|---|---|---|---|
||特征|相对重要性||特征|相对重要性|
|1|绿色专利数|10.48|11|高管薪酬|3.93|
|2|资产规模|7.34|12|融资约束|3.84|
|3|重污染行业|6.68|13|应收账款|3.27|
|4|高科技行业|6.67|14|上市年限|3.22|
|5<br>6<br>7|高管绿色经<br>信息透明度<br>资产负债率|6.18<br>5.65<br>4.90|15<br>16<br>17|董事会规模<br>机构持股比<br>资产报酬率|2.73<br>2.25<br>2.21|
|8|两职合一|4.56|18|资产周转率|2.18|
|9|独立董事占|4.43|19|高管年龄|2.02|
|10|企业性质|4.08|20|大股东持股|2.02|



表 6 不同特征维度下变量的累计相对重要性 ( 单位 :%) 

Table 6 Cumulative Relative Importance of Variables across 

Different Feature Dimensions(Unit: %) 

||内部治理|绿色经营|财务状况|外部治理|
|---|---|---|---|---|
|IALA-CatBoost|36.59|30.01|26.17|7.23|



## **5** 结束语 

针对企业“漂绿”识别中样本不平衡与超参数寻优两 个难点，提出了一种融合 SMOTE-Tomek 采样与改进人工 旅鼠算法优化 CatBoost 的识别模型（ SMOTE-TomekIALA-CatBoost ）。通过对比实验与分析，得出以下结论： (1)SMOTE-Tomek-IALA-CatBoost 综合表现最优，准 确率、精确率、召回率、 F1 与 AUC 分别达到 94.52% 、 

94.73% 、 94.29% 、 94.51% 与 98.72% ，且经 IALA 寻优后 各项指标较 CatBoost 进一步提升。 IALA 借助多子群环形 迁移、基于信息熵的种群规模自适应与历史精英解回溯， 增强全局搜索能力与收敛精度，为 CatBoost 提供了更优 的超参数组合，验证了改进人工旅鼠算法在超参数寻优中 的有效性。 

(2) 在“漂绿”识别的高度不平衡场景下，过采样与组 合采样的整体效果优于欠采样。欠采样因删减多数类而损 失关键信息， RUS 、 NearMiss 处理下多数模型准确率降至 75% 以下。过采样技术 ROS 各项指标虽高，但因随机复制 少数类样本而存在过拟合风险。相比之下，组合采样 SMOTE-Tomek 兼顾少数类样本合成与边界噪声清理，在 识别性能与稳健性上更具优势。 

(3) 模型与特征对比中，梯度提升类模型的识别效果优 于其他模型，其中 CatBoost 表现最优。 SHAP 分析显示， 绿色专利数与资产规模贡献突出，且绿色专利较多、资产 规模较大的企业反而更易被判定为“漂绿”。从维度看，内 部治理与绿色经营是“漂绿”的主要动因。这一结果切中 了“漂绿”言行不一、以符号化披露掩盖实质投入不足的 隐蔽特征，表明模型识别依据具有合理性。 

尽管上述模型取得了良好的识别效果，但所采用的 “漂绿”标签依据环境信息关注度排名与行政处罚构造， 可能遗漏未被处罚或隐性的“漂绿”行为。未来可融合年 报文本、社交媒体舆情等信息细化标签口径，提升对隐性 “漂绿 ” 的识别能力。此外，可探索 IALA 控制参数的自适 应策略以降低调参成本，并将该识别框架推广至财务造假、 

--- end of page.page_number=11 ---

## 信用违约等同类不平衡任务，结合业务规则与可解释结果， 为 ESG 监管提供更具针对性的决策支持。 参考文献 

[1] LYON T P, MAXWELL J W. Greenwash: corporate environmental disclosure under threat of audit[J]. Journal of Economics & Management Strategy, 2011, 20(1): 3-41. “ ” [2] 张云, 杨振宇. 机构投资者绿色关注与企业 漂绿 行 为：效应、诱因与治理[J]. 财经研究, 2024, 50(11): 95-110. ZHANG Y, YANG Z Y. Institutional investors’ green attention and corporate greenwashing behavior: effects, incentives, and governance[J]. Journal of Finance and Economics, 2024, 50(11): 95-110. 

[3] HE R, CHEN H, ZHU X. Corporate hypocrisy and ESG rating divergence[J]. Corporate Social Responsibility and Environmental Management, 2025, 32(1): 1122-1146. 

[4] ZHANG Z, ZHENG X, MENG X. Corporate ESGwashing strategies responding to external rating divergence: moderating effects of normative institutional pressures[J/OL]. Humanities and Social Sciences Communications, 2026[2026-06-02]. https://www.nature.com/articles/s41599026-07306-9. [5] BERG F, KÖLBEL J F, RIGOBON R. Aggregate confusion: the divergence of ESG ratings[J]. Review of Finance, 2022, 26(6): 1315-1344. 

” [6] 李九斤, 衡好婷, 赖峰雷, 等. 企业 ESG“漂绿 行为 动因与绩效优化——基于 HG 公司的案例研究[J]. 财务管 理研究, 2024(4): 10-22. 

LI J J, HENG H T, LAI F L, et al. Motivations and performance optimization of corporate ESG greenwashing: a case study of HG company[J]. Finance and Accounting Research, 2024(4): 10-22. 

“ ” [7] 杨七中, 马蓓丽. 基于机器学习的企业 漂绿 现象识 别与动因挖掘[J]. 统计与决策, 2025(21): 161-166. 

YANG Q Z, MA B L. Identification and driver mining of corporate greenwashing based on machine learning[J]. Statistics & Decision, 2025(21): 161-166. 

[8] CHEN Y, MA D. Detection of greenwashing in ESG reports of Chinese listed companies based on Word2vec and TF-IDF[C]//Proceedings of the 2024 International Conference on Innovation in Artificial Intelligence. New York: Association for Computing Machinery, 2024: 159-164. 

[9] ZHANG J F, QI T T. Interpretable predictive model for listed companies ESG greenwashing based on XGBoost and SHAP[J]. Scientific Reports, 2026, 16(1): 12899. 

[10] 周玉, 孙红玉, 房倩, 等. 不平衡数据集分类方法研 究综述[J]. 计算机应用研究, 2022, 39(6): 1615-1621. 

ZHOU Y, SUN H Y, FANG Q, et al. Review of imbalanced data classification methods[J]. Application Research of Computers, 2022, 39(6): 1615-1621. 

[11] BISCHL B, BINDER M, LANG M, et al. Hyperparameter optimization: foundations, algorithms, best practices, and open challenges[J]. WIREs Data Mining and Knowledge Discovery, 2023, 13(2): e1484. 

[12] XIAO Y, CUI H, KHURMA R A, et al. Artificial lemming algorithm: a novel bionic meta-heuristic technique for solving real-world engineering optimization problems[J]. Artificial Intelligence Review, 2025, 58(3): 84. 

[13] 钱峥远, 曾国荪. 精英化岛屿种群引导的差分进化算 法[J]. 计算机工程与应用, 2021, 57(20): 73-81. 

QIAN Z Y, ZENG G S. Differential evolution algorithm guided by elite island population[J]. Computer Engineering and Applications, 2021, 57(20): 73-81. 

[14] WANG X, WANG F, HE Q, et al. A multi-swarm optimizer with a reinforcement learning mechanism for largescale optimization[J]. Swarm and Evolutionary Computation, 2024, 86: 101486. 

[15] TUĞAL İ. Energy efficiency in building: entropy-based grey wolf optimization for improved MLP performance[J]. Energy Reports, 2025, 13: 4247-4260. 

[16] JIA H, LU C, XING Z. Memory backtracking strategy: an evolutionary updating mechanism for meta-heuristic algorithms[J]. Swarm and Evolutionary Computation, 2024, 84: 101456. 

[17] WANG J, WANG W, HU X, et al. Black-winged kite algorithm: a nature-inspired meta-heuristic for solving benchmark functions and engineering problems[J]. Artificial Intelligence Review, 2024, 57(4): 98. 

[18] AMIRI M H, MEHRABI HASHJIN N, MONTAZERI M, et al. Hippopotamus optimization algorithm: a novel natureinspired optimization algorithm[J]. Scientific Reports, 2024, 14(1): 5032. 

[19] MIRJALILI S, LEWIS A. The whale optimization algorithm[J]. Advances in Engineering Software, 2016, 95: 51-67. 

[20] MIRJALILI S, MIRJALILI S M, LEWIS A. Grey wolf optimizer[J]. Advances in Engineering Software, 2014, 69: 46-61. 

“ ” [21] 孙自愿, 汪玮, 孙孟欣, 等. 媒体报道对企业 漂绿 

--- end of page.page_number=12 ---

的影响：高管特征与内部监督的中介作用[J]. 北京理工大 学学报(社会科学版), 2023, 25(1): 67-79. 

SUN Z Y, WANG W, SUN M X, et al. The impact of media coverage on corporate greenwashing: the mediating role of executive characteristics and internal supervision[J]. Journal of Beijing Institute of Technology (Social Sciences Edition), 2023, 25(1): 67-79. 

[22] 李惠蓉, 赵小克. 投资者关注与企业 ESG 信息披露 “漂绿”[J]. 财会通讯, 2023(23): 51-56. 

LI H R, ZHAO X K. Investor attention and greenwashing in corporate ESG disclosure[J]. Communication of Finance and Accounting, 2023(23): 51-56. 

[23] 杨永聪, 李学轩. 从象征性披露到实质性投入：中央 环保督察何以抑制企业“漂绿”行为[J]. 环境经济研究, 

## 2025, 10(3): 126-151. 

YANG Y C, LI X X. From symbolic disclosure to substantive investment: how central environmental inspection curbs corporate greenwashing[J]. Journal of Environmental Economics, 2025, 10(3): 126-151.. 

[24] ZENG F, WANG J, ZENG C. An optimized machine learning framework for predicting and interpreting corporate ESG greenwashing behavior[J]. PLOS ONE, 2025, 20(3): e0316287. 

[25] LAI H, QUAN L, WU F, et al. Corporate environmental publicity and green innovation: are words consistent with actions?[J]. Humanities and Social Sciences Communications, 2025, 12(1): 514. 

--- end of page.page_number=13 ---

