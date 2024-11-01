from langchain_ollama import OllamaLLM


llm = OllamaLLM(model="llama3:8b", temperature=0)
print(llm.invoke("tell me a joke about bear."))