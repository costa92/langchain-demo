from langchain_community.embeddings import HuggingFaceBgeEmbeddings

model_name = "BAAI/bge-large-en-v1.5"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": True} # set normalize_embeddings to True

model = HuggingFaceBgeEmbeddings(
  model_name=model_name,
  model_kwargs=model_kwargs,
  encode_kwargs=encode_kwargs,
  query_instruction="为这个句子生成表示以用于检索相关文章："
)

print(model.embed_query("Hello, world!").sort())
