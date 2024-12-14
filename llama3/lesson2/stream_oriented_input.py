from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
import multiprocessing
import os
from threading import Thread

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
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    generation_kwargs = dict(
        model_inputs, 
        streamer=streamer,
        max_new_tokens=512,  # 增加生成长度
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    return streamer

if __name__ == "__main__":
    tokenizer, model = init_model()
    
    messages = [
        {"role": "system", "content": "你是一个诗人，请写一首赞美春天的诗，要求不包含春字."},
        {"role": "user", "content": "请写一首赞美春天的诗，要求不能包含春字."},
    ]
    
    streamer = generate_text(tokenizer, model, messages)
    
    output = ""
    for text in streamer:
        output += text
  
    if output:  # 打印剩余内容
        print(output)