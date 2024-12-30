# multi-agent


multi-agent框架的核心交互流程可以概括如下：

controller更新当前环境的状态，选择下一时刻行动的agentA。
agentA t与环境交互，更新自身的memory信息。
agentA调用LLM，基于指令执行动作，获取输出message。
将输出message更新到公共环境中。
