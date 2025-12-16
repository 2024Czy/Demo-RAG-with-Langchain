# RAG 问答系统说明文档

本项目是一个基于 Python 实现的 **检索增强生成**（Retrieval-Augmented Generation, RAG）问答系统，旨在结合本地知识库检索与实时网络搜索能力，为用户提供有据可查、高质量的回答。

系统利用 LangChain 框架进行文档加载、文本切分和向量检索，并结合 HuggingFace 模型进行嵌入（Embedding），同时通过调用大型语言模型（LLM）实现智能问答、联网判定和信息矛盾检测。

---

## 📖 主要功能

1. **多格式文档加载**  
   支持 `.docx`, `.doc`, `.txt`, `.pdf`, `.md`, `.html` 等多种格式作为本地知识库。

2. **数据清洗与预处理**  
   - 去除 HTML 标签、不可打印字符
   - 合并多余空格
   - 删除空文本块、简单去重，提升数据质量

3. **文本切分**  
   使用 `RecursiveCharacterTextSplitter` 递归切分策略，确保语义完整性。

4. **向量存储与检索**  
   - 向量数据库：**Chroma**（支持持久化与追加存储）
   - 嵌入模型：默认使用 `Qwen/Qwen3-Embedding-0.6B`（通过 HuggingFace）

5. **智能联网决策**  
   利用 LLM 分析本地检索结果，自动判断是否需进行实时网络搜索。

6. **网络搜索与信息验证**  
   - 集成 **SerpApi** 调用 Google 搜索获取最新信息
   - 使用 LLM 进行 **矛盾检测** 与 **可信源筛选**

7. **严格引用生成**  
   所有事实性陈述均附带引用：
   - 本地知识库：包含文件名、起始索引、页码等元数据
   - 网络信息：附带原始 URL
   - 支持审计与核查

8. **LLM 交互支持**  
   - 本地部署：通过 **Ollama**，**HuggingFace**
   - 远程调用：通过 **SiliconFlow API**

---

## ⚙️ 环境搭建与依赖

### 1. 基础要求
- Python 3.12

### 2. 安装依赖
```bash
pip install -r requirements.txt
```
暂无 `requirements.txt`，可手动安装核心包：
```bash
pip install langchain-community langchain-text-splitters langchain-huggingface
```

### 3. 外部服务配置

| 服务名称      | 用途                                 | 环境变量名             | 获取方式                     |
|---------------|--------------------------------------|------------------------|------------------------------|
| SiliconFlow   | 调用 LLM（问答、联网判定、矛盾检测） | `SILICONFLOW_API_KEY`  | [注册获取](https://siliconflow.cn) |
| SerpApi       | 实时 Google 搜索                     | `SERPAPI_KEY`          | [注册获取](https://serpapi.com)   |

在项目根目录创建 `.env` 文件：
```env
SILICONFLOW_API_KEY="your_siliconflow_api_key_here"
SERPAPI_KEY="your_serpapi_api_key_here"
```

### 4. 数据目录准备
- `data/`：首次构建知识库的原始文档（PDF、DOCX 等）
- `append_data/`：后续追加文档（如启用追加功能）

---

## 🚀 使用方法

### 1. 配置参数（编辑 `rag_langchain.py`）

在 `if __name__ == '__main__':` 块中设置：

```python
data_directory = os.path.join(os.path.dirname(__file__), "data")
append_files_path = os.path.join(os.path.dirname(__file__), "append_data")
embedding_model_name = "Qwen/Qwen3-Embedding-0.6B"
collection_name = "test-clean"
persist_directory = os.path.join(os.path.dirname(__file__), "chroma3-clean")
top_relevant_k = 10
local_model_name = "google/gemma-2-2b-it"  # 仅 local_model=True 时生效
query = "详细评价一下中国A股"
```

### 2. 首次运行
系统将自动：
- 加载 `data/` 中的文档
- 清洗、切分、向量化
- 存入 Chroma 数据库

```bash
python rag_langchain.py
```

### 3. 执行流程与输出
1. 加载或构建 Chroma 数据库  
2. 构建检索器  
3. 检索本地相关文本块（Top-K）  
4. LLM 判定是否需要联网  
5. （如需）执行网络搜索 + 矛盾检测  
6. 构造 RAG 提示词（含本地+网络上下文）  
7. 调用 LLM 生成带引用的回答  

**输出示例**：
```
数据库已加载成功...
模型判断是否需要联网：FALSE
不需要联网搜索...
问题：详细评价一下中国A股
回答：根据本地知识库资料显示，中国A股市场当前面临...[1]。

【来源说明】
[1] {"source": "report_2024.pdf", "start_index": 520, "page": 3}
```

---

## 💡 待改进与优化方向

1. **记忆与多轮对话**  
   支持对话历史管理，实现上下文连贯的多轮问答。

2. **回答回滚机制**  
   若 LLM 评估最终回答不可靠，可回滚联网判断或搜索结果。

3. **去噪优化**  
   针对 PPT/PDF/HTML 等格式设计专用清洗规则。

4. **常见问题知识库**（FAQ）  
   将用户采纳的优质问答存入独立库，加速高频问题响应。
