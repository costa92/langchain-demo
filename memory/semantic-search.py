from langchain_ollama import OllamaEmbeddings
from langchain_openai import ChatOpenAI
from langgraph.store.memory import InMemoryStore
from langgraph.store.base import BaseStore
from langgraph.graph import START, MessagesState, StateGraph
import os

os.environ["TOKENIZERS_PARALLELISM"] = "true"

# embedd_model = "mxbai-embed-large"
# embeddings = OllamaEmbeddings(model=embedd_model)


# ### =============== HuggingFaceEmbeddings start ==============================
# # 设置模型参数，指定设备为 CPU
from langchain_huggingface import HuggingFaceEmbeddings  # Updated import

model_kwargs = {'device': 'cpu'}
# 设置编码参数，禁用嵌入归一化
encode_kwargs = {
    'normalize_embeddings': False,
    'clean_up_tokenization_spaces': False  # Explicitly set to avoid FutureWarning
}
model_name = "sentence-transformers/all-mpnet-base-v2"

# 创建 HuggingFaceEmbeddings 实例，使用指定的模型和参数
embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
)
# ======================= HuggingFaceEmbeddings end ==========================

store = InMemoryStore(
    index={
        "embed": embeddings,
        "dims": 1536,
    }
)

# Adding memories
store.put(("user_123", "memories"), "1", {"text": "I love pizza"})
store.put(("user_123", "memories"), "2", {"text": "I prefer Italian food"})
store.put(("user_123", "memories"), "3", {"text": "I don't like spicy food"})
store.put(("user_123", "memories"), "3", {"text": "I am studying econometrics"})
store.put(("user_123", "memories"), "3", {"text": "I am a plumber"})
# Find memories about food preferences
memories = store.search(("user_123", "memories"), query="I like food?", limit=5)

for memory in memories:
    print(f'Memory: {memory.value["text"]} (similarity: {memory.score})')


model_name="Qwen/Qwen2.5-7B-Instruct"
# model_name="Qwen/Qwen2-VL-72B-Instruct"
import os
api_key =  os.getenv("OPENAI_API_KEY")
base_url="https://api.siliconflow.cn/v1/"

# We will use this model for both the conversation and the summarization
# :zh: 我们将使用这个模型来进行对话和总结
model = ChatOpenAI(
    model=model_name,
    api_key=api_key,
    base_url=base_url,
)

def chat(state,*,store: BaseStore):
    # Search based on user's last message
    items = store.search(
        ("user_123", "memories"), query=state["messages"][-1].content, limit=2
    )
    memories = "\n".join(item.value["text"] for item in items)
    memories = f"## Memories of user\n{memories}" if memories else ""
    response = model.invoke(
        [
            {"role": "system", "content": f"You are a helpful assistant.\n{memories}"},
            *state["messages"],
        ]
    )
    return {"messages": [response]}


builder = StateGraph(MessagesState)
builder.add_node(chat)
builder.add_edge(START, "chat")
graph = builder.compile(store=store)

# INSERT_YOUR_REWRITE_HERE
from IPython.display import display
from PIL import Image as PILImage
import io
image_data = graph.get_graph().draw_mermaid_png()
image = PILImage.open(io.BytesIO(image_data))
image.save("semantic-search.png")

for message, metadata in graph.stream(
    input={"messages": [{"role": "user", "content": "I'm hungry"}]},
    stream_mode="messages",
):
    print(message.content, end="")