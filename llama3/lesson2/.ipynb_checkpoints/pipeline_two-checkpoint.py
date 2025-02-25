

from transformers import pipeline



def data():
  for i in range(10):
    yield f"My example {i}"


pipe = pipeline(model="openai-community/gpt2", device=0,eos_token_id=2, pad_token_id=2)

generated_characters = 0
for out in pipe(data()):
    generated_characters += len(out[0]["generated_text"])

print(generated_characters)
