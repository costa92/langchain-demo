from turtle import mode
from langchain_ollama import OllamaLLM

model_name = "llama3"
temperature = 0
llm = OllamaLLM(model=model_name, temperature=temperature)
print(llm.invoke("tell me a joke about bear."))
  