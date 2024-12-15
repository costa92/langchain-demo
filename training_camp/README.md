

## 大语言模型训练阶段

1. **预训练 (Pre-Training)**
   - 在大规模无标注语料上进行自监督学习
   - 学习语言的基本特征和语义知识
   - 通常使用掩码语言模型(MLM)和下一句预测(NSP)等任务
   - 需要大量计算资源和训练时间
   - 代表模型:BERT、GPT、RoBERTa等

2. **微调 (Instruction-Tuning)**
   - 在预训练模型基础上针对特定任务进行调整
   - 使用带标注的任务数据进行有监督学习
   - 可以使用全量微调或参数高效微调(PEFT)
   - 相比预训练需要更少的计算资源
   - 常见技术:LoRA、P-tuning、Prompt-tuning等

3. **对齐 (Alignment)**
   - 使模型输出符合人类价值观和偏好
   - 通常包括RLHF(基于人类反馈的强化学习)
   - 提高模型输出的安全性和可控性
   - 代表技术:PPO、DPO、RLAIF等

总结:大语言模型的训练是一个多阶段过程,从通用预训练到特定任务微调,再到人类对齐,每个阶段都有其特定的目标和技术方法。这种渐进式的训练方式能够让模型既掌握基础语言知识,又能很好地完成下游任务,同时保持输出的可靠性。

## ChatGPT 三阶段训练法

1. **有监督微调 (Supervised Fine-tuning, SFT)**
   - 使用高质量的人工标注数据进行微调
   - 让模型学习遵循指令的基本能力
   - 通过人工编写的问答对训练模型
   - 提高模型的任务完成能力

2. **奖励建模 (Reward Modeling, RM)**
   - 训练奖励模型来评估回答质量
   - 使用人类偏好数据训练奖励模型
   - 为每个模型回答打分和排序
   - 指导后续强化学习阶段

3. **强化学习优化 (Reinforcement Learning, RL)**
   - 使用 PPO 算法基于奖励模型进行训练
   - 最大化期望奖励来优化模型策略
   - 让模型输出更符合人类偏好
   - 实现与人类价值观的对齐
  

## 大语言模型基础概念

1. **In-Context Learning (上下文学习)**
   - 让模型通过上下文示例来理解和执行任务
   - 无需额外训练,直接在提示中提供示例
   - 适用于简单任务和快速原型验证

2. **Few-shot Learning (少样本学习)**
   - 使用少量标注数据进行学习和泛化
   - 通过提供少量示例来指导模型完成任务
   - 降低了数据标注成本,提高了模型应用效率

3. **Prompt Engineering (提示工程)**
   - 设计和优化提示以提升模型性能
   - 包括提示模板设计、示例选择、提示策略等
   - 是提高大语言模型效果的关键技术

## Prompt思维 (Prompt Thinking)

1. **思维链 (Chain of Thought, CoT 链式思维)**
   - 让模型展示推理过程的提示技术 (Prompting technique for showing reasoning process)
   - 通过分步骤思考提高推理准确性 (Step-by-step reasoning for better accuracy)
   - 适用于复杂问题的解决 (Suitable for complex problem solving)
   - 常见形式:"让我们一步一步思考" (Common format: "Let's think step by step")
   
   工作原理：
   - 通过提示词引导模型将复杂问题分解为多个简单步骤
   - 每个步骤都显式展示推理过程和中间结果
   - 最后基于完整的推理链得出最终答案
   - 提高了推理透明度和可解释性

2. **零样本思维链 (Zero-shot CoT 零样本链式思维)**
   - 无需示例的思维链推理 (Chain-of-thought reasoning without examples)
   - 直接要求模型分步骤思考 (Direct step-by-step reasoning)
   - 简单但效果显著的提示方法 (Simple but effective prompting method)

   工作原理：
   - 在提示中直接加入"让我们一步一步思考"等触发词
   - 模型基于预训练知识自动生成推理步骤
   - 不需要人工提供示例就能进行链式推理
   - 利用模型的内在推理能力

3. **少样本思维链 (Few-shot CoT 少样本链式思维)** 
   - 提供少量带推理过程的示例 (Providing few examples with reasoning process)
   - 引导模型学习类似的推理模式 (Guide model to learn similar reasoning patterns)
   - 提高复杂任务的准确率 (Improve accuracy on complex tasks)

   工作原理：
   - 提供2-3个包含完整推理过程的示例
   - 模型通过模仿示例的推理模式来解决新问题
   - 示例中的推理步骤为模型提供了学习范式
   - 结合了示例学习和链式推理的优势

4. **自洽性 (Self-Consistency 自我一致性)**
   - 生成多个推理路径 (Generate multiple reasoning paths)
   - 通过多样化思考提高可靠性 (Improve reliability through diverse thinking)
   - 选择最一致的答案作为输出 (Select most consistent answer as output)

   工作原理：
   - 对同一问题生成多条不同的推理路径
   - 比较不同路径得出的结论是否一致
   - 采用多数表决或置信度加权的方式选择最终答案
   - 通过多样化推理提高结果的可靠性

5. **思考-行动框架 (ReAct Reasoning-Acting Framework)**
   - 结合思考(Reasoning)和行动(Action) (Combine reasoning and action)
   - 交替进行推理和工具使用 (Alternate between reasoning and tool use)
   - 增强模型解决实际问题的能力 (Enhance model's practical problem-solving ability)

   工作原理：
   - 模型在推理过程中识别需要使用工具的场景
   - 调用相应的工具获取必要信息
   - 将工具返回的结果整合到推理过程中
   - 通过思考和行动的交互完成复杂任务

6. **思维树 (Tree of Thought)** 
   - 构建多分支的推理树结构 (Build multi-branch reasoning tree structure)
   - 探索多个可能的思维路径 (Explore multiple possible thinking paths)
   - 通过搜索和评估选择最优路径 (Select optimal path through search and evaluation)
   - 适用于需要深度推理的复杂问题 (Suitable for complex problems requiring deep reasoning)
   - 支持回溯和分支决策 (Support backtracking and branching decisions)
   
   工作原理：
   - 将问题分解为多个决策节点
   - 在每个节点生成多个可能的推理分支
   - 使用启发式搜索算法探索决策树
     - 广度优先搜索 ()
     - 深度优先搜索 ()
   - 评估每条路径的可行性和结果质量
   - 选择最优路径作为最终解决方案
    

## 参考文档

- [保姆级讲解BERT](https://mp.weixin.qq.com/s/Pa69sOWy4fCsyEntwg_F-g)
- [openai-quickstart](https://github.com/DjangoPeng/openai-quickstart)
- [agicto api](https://agicto.com/)