# 制作并量化GGUF模型

## 安装 llama.cpp 

1.下载

```sh
git clone https://github.com/ggerganov/llama.cpp.git
```

2. 安装

```sh
cd llama.cpp

brew install cmake

pip install -r requirements.txt

cmake -B build

cmake --build build --config Release
```

3.验证

```sh
./llama-quantize --help
```

## 下载模型

1. 需要获取 meta-llama/Llama-3.2-1B-Instruct 权限
2. 下载 Llama-3.2-1B-Instruct 模型，--local-dir 指定保存到当前目录

```sh
huggingface-cli download meta-llama/Llama-3.2-1B-Instruct --local-dir Llama-3.2-1B-Instruct
```

## 模型转换为 GGUF 格式与量化模型


1. 将模型转换为 ggml FP16 格式

```sh
python3 convert_hf_to_gguf.py models/mymodel/
```

进入下载的model目录

```sh
python3 ../llama.cpp/convert_hf_to_gguf.py Llama-3.2-1B-Instruct
```

2. 将模型量化为4位（使用Q4_K_M方法）

```sh
./llama-quantize ./models/mymodel/ggml-model-f16.gguf ./models/mymodel/ggml-model-Q4_K_M.gguf Q4_K_M
```

```sh
../llama.cpp/llama-quantize Llama-3.2-1B-Instruct/Llama-3.2-1B-Instruct-F16.gguf ./Llama-3.2-1B-Instruct/ggml-llama-3.2-1B-Instruct-Q4_K_M.gguf Q4_K_M
```

3.如果旧版本现在不再受支持，请将 gguf 文件类型更新为当前版本。

```sh
./llama-quantize ./models/mymodel/ggml-model-Q4_K_M.gguf ./models/mymodel/ggml-model-Q4_K_M-v2.gguf COPY

```

可以将模型转换为 FP16 精度的 GGUF 模型，并分别用 Q8_0、Q6_K、Q5_K_M、Q5_0、Q4_K_M、Q4_0、Q3_K、Q2_K