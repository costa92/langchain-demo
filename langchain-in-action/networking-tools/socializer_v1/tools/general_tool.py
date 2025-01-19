import re 

def contains_chinese(text: str) -> bool:
    return bool(re.search(r'[\u4e00-\u9fff]', text))


def remove_non_chinese_fields(d):
    if isinstance(d, dict):
        to_remove = [key for key, value in d.items() if isinstance(value, (str, int, float, bool)) and (not contains_chinese(str(value)))]
        for key in to_remove:
            del d[key]
        
        for key, value in d.items():
            if isinstance(value, (dict, list)):
                remove_non_chinese_fields(value)
    elif isinstance(d, list):
        to_remove_indices = []
        for i, item in enumerate(d):
            if isinstance(item, (str, int, float, bool)) and (not contains_chinese(str(item))):
                to_remove_indices.append(i)
            else:
                remove_non_chinese_fields(item)
        
        for index in reversed(to_remove_indices):
            d.pop(index)