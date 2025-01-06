from llama_cpp import Llama

from huggingface_hub import hf_hub_download
from typing import Optional, List, Mapping, Any, ClassVar
from langchain.llms.base import LLM

# 下载  huggingface-cli download TheBloke/Llama-2-7B-GGUF llama-2-7b.Q4_K_M.gguf --local-dir . --local-dir-use-symlinks False
# 使用本地模型
# llm = Llama(model_path="./llama-2-7b.Q4_K_M.gguf", rms_norm_eps=1e-5)

MODEL_NAME = 'ggml-llama-3.2-1B-Instruct-Q4_K_M.gguf'
# MODEL_NAME = 'llama-2-7b-chat.ggmlv3.q4_K_S.bin'
MODEL_PATH = '/Users/costalong/code/python/models/Llama-3.2-1B-Instruct/'


# 自定义的LLM类，继承自基础LLM类
class CustomLLM(LLM):
    model_name: ClassVar[str] = MODEL_NAME

    # 该方法使用Llama库调用模型生成回复
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        prompt_length = len(prompt) + 5
        # 初始化Llama模型，指定模型路径和线程数
        llm = Llama(model_path=MODEL_PATH+MODEL_NAME, n_threads=4)
        # llm = Llama(model_path="./llama-2-7b.Q4_K_M.gguf", rms_norm_eps=1e-5)

        # 使用Llama模型生成回复
        response = llm(f"Q: {prompt} A: ", max_tokens=256)

        print(response)
        
        # 从返回的回复中提取文本部分
        output = response['choices'][0]['text'].replace('A: ', '').strip()

        # 返回生成的回复，同时剔除了问题部分和额外字符
        return output[prompt_length:]

    # 返回模型的标识参数，这里只是返回模型的名称
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"name_of_model": self.model_name}

    # 返回模型的类型，这里是"custom"
    @property
    def _llm_type(self) -> str:
        return "custom"
    

# 初始化自定义LLM类
llm = CustomLLM()

# 使用自定义LLM生成一个回复
result = llm.invoke("昨天有一个客户抱怨他买了花给女朋友之后，两天花就枯了，作为客服我应该怎么解释？")

# 打印生成的回复
print(result)