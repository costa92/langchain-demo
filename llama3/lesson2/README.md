
## 模型下载的位置

默认在：~/.cache/huggingface/hub 文件下 

预先使用 PreTrainedModel.from_pretrained() 下载文件：

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "Qwen/Qwen2.5-7B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True, device_map="auto")
```

使用 PreTrainedModel.save_pretrained() 将文件保存至指定目录：

```python 
tokenizer.save_pretrained("./your/path/bigscience_t0")
model.save_pretrained("./your/path/bigscience_t0")
```

你可以在离线时从指定目录使用 PreTrainedModel.from_pretrained() 重新加载你的文件：

```python
tokenizer = AutoTokenizer.from_pretrained("./your/path/bigscience_t0")
model = AutoModelForSeq2SeqLM.from_pretrained("./your/path/bigscience_t0")
```


## 使用 huggingface

### 方法一：使用 huggingface-cli  

```bash
huggingface-cli download Qwen/Qwen2.5-7B-Instruct
```

### 方法二：使用 huggingface

1.安装依赖：

```bash
pip install -U huggingface_hub
```

2. 设置环境变量：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

3. 下载模型或数据集：

```bash
# 下载模型
huggingface-cli download --resume-download [模型名称] --local-dir [本地目录]

# 可以添加 --local-dir-use-symlinks False 参数禁用文件软链接，这样下载路径下所见即所得。
huggingface-cli download --resume-download [模型名称] --local-dir [本地目录] --local-dir-use-symlinks False

# 下载数据集
huggingface-cli download --repo-type dataset --resume-download [数据名称] --local-dir [本地目录]
```



## 文档

- [推理pipeline](https://huggingface.co/docs/transformers/zh/pipeline_tutorial)
