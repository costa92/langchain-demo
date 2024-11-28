# langchain

## 环境安装

[conda 安装](/docs/conda%20安装.md)


## 使用 LangServe 提供服务
```sh
pip install "langserve[all]"
```

## 本地运行的加载程序

请按照以下步骤获取 unstructured 和 其依赖项在本地运行。

```sh
pip install "unstructured[local-inference]" 
```

需要安装的扩展：

  libmagic-dev（文件类型检测）
  poppler-utils（图像和 PDF）
  tesseract-ocr（图像和 PDF）
  libreoffice（MS Office 文档）
  pandoc（EPUB）

## 参考文档

[爱鼓捣-blog](https://techdiylife.github.io/blog/topic.html?category2=t07&blogid=0043)
[langchain 中文](http://python.langchain.com.cn/)
[LangChain实战 | 3分钟学会SequentialChain传递多个参数](https://blog.csdn.net/sinat_29950703/article/details/139263894)
[langchain学习之chain机制](https://blog.csdn.net/zc1226/article/details/140011057?spm=1001.2101.3001.6650.15&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-15-140011057-blog-139263894.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-15-140011057-blog-139263894.235%5Ev43%5Epc_blog_bottom_relevance_base9&utm_relevant_index=18)


## 许可证

本仓库的代码和文档遵循 [MIT License](LICENSE) 许可证。
