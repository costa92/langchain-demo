# punkt 错误

## 1.错误信息

```sh
  Resource punkt not found.
  Please use the NLTK Downloader to obtain the resource:

  >>> import nltk
  >>> nltk.download('punkt')
  
  For more information see: https://www.nltk.org/data.html

  Attempted to load tokenizers/punkt/PY3/english.pickle

  Searched in:
    - '/home/costalong/nltk_data'
    - '/home/costalong/anaconda3/nltk_data'
    - '/home/costalong/anaconda3/share/nltk_data'
    - '/home/costalong/anaconda3/lib/nltk_data'
    - '/usr/share/nltk_data'
    - '/usr/local/share/nltk_data'
    - '/usr/lib/nltk_data'
    - '/usr/local/lib/nltk_data'
    - ''
**********************************************************************
```

## 解决方式

1. 下载 punkt

```sh
python -m nltk.downloader punkt
```

输出:

```sh
/home/costalong/anaconda3/lib/python3.10/runpy.py:126: RuntimeWarning: 'nltk.downloader' found in sys.modules after import of package 'nltk', but prior to execution of 'nltk.downloader'; this may result in unpredictable behaviour
  warn(RuntimeWarning(msg))
[nltk_data] Downloading package punkt to /home/costalong/nltk_data...
[nltk_data]   Unzipping tokenizers/punkt.zip.

```

## 2. nltk 错误

```sh
INFO: NumExpr defaulting to 12 threads.
Traceback (most recent call last):
  File "/home/costalong/code/python/langchain/demo-1/loading_local.py", line 9, in <module>
    document = loader.load()
  File "/home/costalong/anaconda3/lib/python3.10/site-packages/langchain_core/document_loaders/base.py", line 31, in load
    return list(self.lazy_load())
  File "/home/costalong/anaconda3/lib/python3.10/site-packages/langchain_unstructured/document_loaders.py", line 178, in lazy_load
    yield from load_file(f=self.file, f_path=self.file_path)
  File "/home/costalong/anaconda3/lib/python3.10/site-packages/langchain_unstructured/document_loaders.py", line 212, in lazy_load
    else self._elements_json
  File "/home/costalong/anaconda3/lib/python3.10/site-packages/langchain_unstructured/document_loaders.py", line 231, in _elements_json
    return self._convert_elements_to_dicts(self._elements_via_local)
  File "/home/costalong/anaconda3/lib/python3.10/site-packages/langchain_unstructured/document_loaders.py", line 249, in _elements_via_local
    return partition(
  File "/home/costalong/anaconda3/lib/python3.10/site-packages/unstructured/partition/auto.py", line 411, in partition
    elements = partition_text(
  File "/home/costalong/anaconda3/lib/python3.10/site-packages/unstructured/partition/common/metadata.py", line 162, in wrapper
    elements = func(*args, **kwargs)
  File "/home/costalong/anaconda3/lib/python3.10/site-packages/unstructured/chunking/dispatch.py", line 74, in wrapper
    elements = func(*args, **kwargs)
  File "/home/costalong/anaconda3/lib/python3.10/site-packages/unstructured/partition/text.py", line 104, in partition_text
    element = element_from_text(ctext)
  File "/home/costalong/anaconda3/lib/python3.10/site-packages/unstructured/partition/text.py", line 149, in element_from_text
    elif is_possible_narrative_text(text):
  File "/home/costalong/anaconda3/lib/python3.10/site-packages/unstructured/partition/text_type.py", line 74, in is_possible_narrative_text
    if exceeds_cap_ratio(text, threshold=cap_threshold):
  File "/home/costalong/anaconda3/lib/python3.10/site-packages/unstructured/partition/text_type.py", line 270, in exceeds_cap_ratio
    if sentence_count(text, 3) > 1:
  File "/home/costalong/anaconda3/lib/python3.10/site-packages/unstructured/partition/text_type.py", line 219, in sentence_count
    sentences = sent_tokenize(text)
  File "/home/costalong/anaconda3/lib/python3.10/site-packages/unstructured/nlp/tokenize.py", line 134, in sent_tokenize
    _download_nltk_packages_if_not_present()
  File "/home/costalong/anaconda3/lib/python3.10/site-packages/unstructured/nlp/tokenize.py", line 123, in _download_nltk_packages_if_not_present
    tokenizer_available = check_for_nltk_package(
  File "/home/c o s ta long/anaconda3/lib/python3.10/site-packages/unstructured/nlp/tokenize.py", line 110, in check_for_nltk_package
    nltk.find(f"{package_category}/{package_name}", paths=paths)
  File "/home/costalong/anaconda3/lib/python3.10/site-packages/nltk/data.py", line 537, in find
    return FileSystemPathPointer(p)
  File "/home/costalong/anaconda3/lib/python3.10/site-packages/nltk/compat.py", line 41, in _decorator
    return init_func(*args, **kwargs)
  File "/home/costalong/anaconda3/lib/python3.10/site-packages/nltk/data.py", line 312, in __init__
    raise OSError("No such file or directory: %r" % _path)
OSError: No such file or directory: '/home/costalong/nltk_data/tokenizers/punkt/PY3_tab'
```

解决方式:

```sh
pip install --user -U nltk  
```

输出:

```sh
Looking in indexes: https://pypi.tuna.tsinghua.edu.cn/simple/
Requirement already satisfied: nltk in /home/costalong/anaconda3/lib/python3.10/site-packages (3.7)
Collecting nltk
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/4d/66/7d9e26593edda06e8cb531874633f7c2372279c3b0f46235539fe546df8b/nltk-3.9.1-py3-none-any.whl (1.5 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.5/1.5 MB 9.2 MB/s eta 0:00:00
Requirement already satisfied: click in /home/costalong/anaconda3/lib/python3.10/site-packages (from nltk) (8.1.7)
Requirement already satisfied: joblib in /home/costalong/anaconda3/lib/python3.10/site-packages (from nltk) (1.4.2)
Requirement already satisfied: regex>=2021.8.3 in /home/costalong/anaconda3/lib/python3.10/site-packages (from nltk) (2024.9.11)
Requirement already satisfied: tqdm in /home/costalong/anaconda3/lib/python3.10/site-packages (from nltk) (4.66.5)
Installing collected packages: nltk
Successfully installed nltk-3.9.1
```

## 参考文档

[nltk](https://www.nltk.org/install.html)
[Python安装NLTK遇到的Punkt问题解决方案](https://gitcode.com/Resource-Bundle-Collection/1fe89/overview?utm_source=pan_gitcode&index=bottom&type=card&webUrl&isLogin=1)