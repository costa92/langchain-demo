import time
from langchain.globals import set_llm_cache
from langchain_ollama import OllamaLLM
from langchain_community.cache import InMemoryCache

# 初始化LLM模型和缓存
llm = OllamaLLM(model="llama3", temperature=0)
set_llm_cache(InMemoryCache())

# 测试缓存效果
prompts = ["Tell me a joke"] * 2
start_times = [time.time() for _ in range(len(prompts))]

for i, prompt in enumerate(prompts):
    print(llm.invoke(prompt))
    print(f"elapsed time for prompt {i+1}: {time.time() - start_times[i]}")