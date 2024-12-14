import torch
from transformers import pipeline, AutoTokenizer
import warnings

# 忽略特定的 FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning, message=".*clean_up_tokenization_spaces.*")

# device = "cuda" if torch.cuda.is_available() else "cpu"
device = 0
model_name = "Helsinki-NLP/opus-mt-en-zh"
# 创建tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    clean_up_tokenization_spaces=True  # 明确设置为False以避免弃用警告
)

# pipe = pipeline("translation_en_to_fr", model="Helsinki-NLP/opus-mt-en-fr")

pipe = pipeline(
  "translation_en_to_zh", 
  model=model_name,
  device=device,
  pad_token_id=tokenizer.eos_token_id,
)

result = pipe("I love programming.")

print(result[0]['translation_text'])