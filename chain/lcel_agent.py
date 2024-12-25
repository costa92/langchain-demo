from langchain_core.tools import tool

"""  
定义Agent需要使用的Tools  
"""  
@tool
def search(query:str)->str:
    """  
    选择动作  
    """  
    if "冰淇淋" in query:
        return "冰淇淋是一种非常美味的食物"
    return "我不知道"

