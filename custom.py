from typing import Any, List, Optional
 
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
 
 
#  一些可选的实现接口方法：

# _acall：与_call类似，异步版，供ainvoke调用。
# _stream：逐个token地流式输出。
# _astream：stream的异步版。
# _identifying_params：@property修饰的属性，用于帮助识别模型并打印LLM，应返回一个字典。


class CustomLLM(LLM):
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        if stop is not None:
            raise NotImplementedError("stop kwargs are not permitted.")
        return "Yeh! I know everything, but I don't want to tell you!"
 
    @property
    def _llm_type(self) -> str:
        return "my-custom-llm"
 
 
llm = CustomLLM()
print(llm.invoke("Tell me a joke about bear"))