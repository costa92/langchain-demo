from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
import multiprocessing
import os

# Set TOKENIZERS_PARALLELISM to false to avoid deadlocks after forking
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def init_model():
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model

def generate_text(tokenizer, model, messages):
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512,
        streamer=streamer,
    )
    return generated_ids

if __name__ == "__main__":
    tokenizer, model = init_model()
    
    messages = [
        # {"role": "system", "content": "你是一个诗人，请写一首赞美春天的诗，要求不包含春字."},
        {"role": "user", "content": "请写一首赞美春天的诗，要求不能包含春字."},
    ]
    
    generated_ids = generate_text(tokenizer, model, messages)
    # First print via streamer in generate_text()
    # Second print here
    # print(generated_ids.char())
