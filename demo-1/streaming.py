from langchain_ollama import OllamaLLM

llm = OllamaLLM(model="llama3", temperature=0, max_tokens=512)
for chunk in llm.stream("怎么评价人工智能，中文回答。Think step-by-step"):
    print(chunk, end="", flush=True)