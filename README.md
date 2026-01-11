# GeoCrawler
本项目是一个专为肺癌单细胞研究（或扩展至其他生物医学领域）设计的元数据抓取与深度分析工具。它通过结合 NCBI Entrez API、静态网页爬虫以及大语言模型（LLM），能够自动化从 GEO 数据库中提取、处理并分析复杂的数据集信息。

## 🚀 核心优化功能

相比于基础版本，Beta 1.0 引入了以下重大改进：

1.  **文献深度爬取 (Source Literature Extraction)**：
    *   **精准分析**：只要 GSE 数据集提供了 Citation 链接（PubMed ID），脚本会自动调取 PubMed API 获取摘要，并尝试从 PubMed Central (PMC) 爬取全文（聚焦 Methods, Results 等核心章节）。
    *   **信息补全**：通过分析源文献，大幅提高了对实验设计、样本组织、干预措施等研究内容的分析准确性。

2.  **增强的鲁棒性 (Enhanced Robustness)**：
    *   **抗网络波动**：引入了 `tenacity` 重试机制和指数退避策略，有效应对网络超时或 API 频控。
    *   **语料截断**：针对 LLM 的 Token 限制，加入了智能上下文处理与长度截断功能，确保长语料下分析流程不中断。
    *   **断点续传**：支持保存临时进度文件（CSV），在发生意外中断后可减少重复工作。

3.  **多模型云端驱动 (Hybrid API Support)**：
    *   **ModelScope 社区 API（默认）**：深度集成阿里云 ModelScope 接口，用户可自行注册获取访问令牌（Token），享受每日免费对话额度。
    *   **OpenAI 兼容接口**：提供标准的 OpenAI 兼容调用方式，支持更灵活的模型切换。

4.  **细粒度分析维度**：
    *   自动化分析：**是否提供原始测序数据 (SRA)**、**样本组织类型**、**数据集研究设计**、**测序技术类型**、**样本数量**、**发表杂志名称及预估影响因子 (IF)**、**是否包含子数据集 (SubSeries)**、**是否包含配对数据,如癌与旁、血、治疗前后**等深度字段。

---

## ⚠️ 重要说明

1.  **Prompt 定制化**：脚本内预设的细粒度分析逻辑是基于**深度定制的肺癌研究 Prompt** 实现的。如果您从事的是其他癌种或研究领域，请务必根据自己的研究需求调整脚本中 `ai_extract_info` 函数内的 System Prompt。
2.  **局限性与幻觉提示**：本项目仍处于 Beta 阶段，存在分析速度受限、算力利用率不高以及结果准确度波动等不足。由于 LLM（大语言模型）本质上可能产生“幻觉”，**请勿完全相信 AI 生成的分析结果**。建议用户对导出结果进行手动核查，最终结论应以 NCBI 数据库的手动检索和原始文献内容为准。
   
---

## 🛠 环境准备

### 1. 创建 Conda 环境
建议使用 Python 3.9+。在终端中运行以下命令：

```bash
# 创建环境
conda create -n geo_crawler python=3.9 -y

# 激活环境
conda activate geo_crawler
```

### 2. 下载相关包
安装脚本运行所需的依赖库：

```bash
pip install pandas biopython tqdm requests beautifulsoup4 python-dotenv openai tenacity openpyxl
```

---

## 🔑 接入指南

在使用脚本前，需要进行简单的配置：

### 1. 接入 NCBI 接口
在脚本 `Crawler_beta_1.0.py` 的“配置区域”中修改邮箱。NCBI 要求提供邮箱以识别频繁请求者。

```python
# NCBI 登陆邮箱
Entrez.email = "your_email@example.com"
```

### 2. 接入 ModelScope API (LLM 支持)
本项目默认使用 ModelScope 社区提供的 Qwen 系列模型。
1.  访问 [ModelScope 魔搭社区](https://modelscope.cn/) 注册账号。
2.  进入“个人中心” -> “访问令牌 (Access Token)” 获取你的 Token。
3.  **配置方式**（二选一）：
    *   **创建环境变量文件**：在脚本同级目录下新建 `.env` 文件，内容为 `MODELSCOPE_ACCESS_TOKEN=你的TOKEN`。
    *   **直接修改脚本**：在 `MODELSCOPE_ACCESS_TOKEN` 后填入你的字符串。

---

## 📖 脚本功能预览

脚本执行流程如下：
1.  **关键词搜索**：根据预设的 MeSH 词或关键词在 NCBI GDS 数据库中筛选数据集。
2.  **白名单过滤**：自动跳过 `whitelist.csv` 中已处理过的 GSE ID。
3.  **多源抓取**：综合抓取 GEO 网页静态信息 + PubMed 文献元数据 + PMC 全文。
4.  **AI 解析**：将非结构化文本传递给 LLM，解析为结构化 JSON，提取精确的样本量、亚型、治疗背景等信息。
5.  **结果导出**：生成带有时间戳的 `.xlsx` 表格文件，方便后续筛选。

## 📝 使用方法
```bash
python Crawler_beta_1.0.py
```
*提示：脚本运行期间会实时显示进度条（tqdm），每处理 4 个样本会保存一次临时进度。*

---

## 📊 部分输出字段的说明
*   `Accession`: GSE 编号及链接
*   `SRA`: 是否包含原始数据
*   `Journal_Name`: 发表期刊
*   `Estimated_IF`: 预估影响因子
*   `Sequencing_Strategy`: 研究中采取的测序技术
*   `Paired_Normal`: 是否包含配对癌旁
*   `Study_Architecture`: 包含样本分布、设计类别、治疗背景等深度结构化分析结果。
*   `SubSeries_IDs`: 包含的子数据集
*   `Treatment_Context`: 采取的干预措施
*   `Design_Class`: 数据集设计
*   `Dataset_Composition`: 数据集的结构
*   `Original_`: 保留的数据集介绍原始字段

---
*Disclaimer: This tool is for research purposes only. Please adhere to the usage policies of NCBI and relevant API providers.*
