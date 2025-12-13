from langchain_community.document_loaders import Docx2txtLoader, TextLoader, PyPDFLoader,UnstructuredWordDocumentLoader,UnstructuredMarkdownLoader,UnstructuredHTMLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from chromadb.config import Settings
from langchain_chroma import Chroma
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import inspect
import os
import requests
import json
import docx
from langchain_core.documents import BaseDocumentTransformer, Document
from typing import Iterable

'''
待改进的点：
1. 文档加载器的选择，目前只支持docx格式，可以扩展到其他格式
2、多文件上传和追加存储
3、噪声处理机制
4、网络搜索机制
5、矛盾检测机制
6、自定义分块逻辑
7、本地模型回答混乱并且加载缓慢
8、模型回答溯源
。。。。

'''




# 加载单个文件
def load_single_file(file_path):  # 逐个读取，后期文件多可以设置mode="multi"实现同类型多文件读取
    '''
        输入：文件路径
        输出： List[Document]
    '''
    if file_path.endswith(".docx") or file_path.endswith(".doc"):
        loader = UnstructuredWordDocumentLoader(file_path)
    elif file_path.endswith(".txt"):
        loader = TextLoader(file_path, encoding="utf-8")
    elif file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".md"):
        loader = UnstructuredMarkdownLoader(file_path)
    elif file_path.endswith(".html"):
        loader = UnstructuredHTMLLoader(file_path)
    else:
        raise ValueError("不支持的文件类型")
    return loader.load()


# 加载一个目录下的所有文件
def load_directory(directory):
    '''
        输入：目录路径
        输出：List[Document]
    '''
    files = os.listdir(directory)
    print(files)
    docs = []
    for file in files:
        file_path = os.path.join(directory, file)
        if os.path.isfile(file_path):
            new_docs = load_single_file(file_path)
            docs = docs + new_docs
            print(f"加载{file_path}成功")
            print(f"新增长度：{len(new_docs)}, 当前长度：{len(docs)}")
    return docs


# 文本切分
def split_text(documents:Iterable[Document], chunk_size=200, chunk_overlap=30):
    '''
        输入：list[Document]，切分参数
        输出：list[Document]
    '''
    splliter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap) # 循环递归切分器（段落-句子-字符），最大字符数为500，切分重叠度为50
    chunks = splliter.split_documents(documents) 
    print(f"分块数：{len(chunks)}")
    return chunks

# 加载编码器
def load_encoder(embedding_model_name):
    '''
        输入：embedding_model_name
        输出：embedding 类实例
    '''
    embedding = HuggingFaceEmbeddings(model_name = embedding_model_name)
    print(f"加载{embedding_model_name}成功")
    return embedding

# Chroma存储
def store_chroma(documents:Iterable[Document], embedding_model_name, collection_name, persist_directory):
    '''
        输入：list[Document]，embedding_function，persist_directory
        输出：Chroma
    '''
    if not os.path.exists(os.path.join(persist_directory, "chroma.sqlite3")):
        os.makedirs(persist_directory, exist_ok=True)
        embedding = load_encoder(embedding_model_name)
        db = Chroma.from_documents(documents=documents,embedding=embedding,collection_name=collection_name, persist_directory=persist_directory) 
        print('数据库已存储成功')
        return db
    else: # 追加存储
        print(1)
        pass

# 加载Chroma
def load_chroma(embedding_model_name, persist_directory, collection_name):
    '''
        输入：embedding_function，persist_directory，collection_name
        输出：Chroma
    '''
    if not os.path.exists(os.path.join(persist_directory, "chroma.sqlite3")):
        raise FileNotFoundError("chroma.sqlite3不存在")
        
    else:
        print("chroma.sqlite3已存在，加载...")
        embedding = load_encoder(embedding_model_name)
        db = Chroma(
            collection_name=collection_name,
            embedding_function=embedding,
            persist_directory=persist_directory,
        )
        print('数据库已加载成功')
        return db

# 构建检索器
def build_retriever(db, search_type="similarity", search_kwargs={"k": 3}):
    '''
        输入：db, search_type，search_kwargs
        输出：检索器
    '''
    retriever = db.as_retriever(search_type=search_type, search_kwargs=search_kwargs)
    return retriever



# 选择回答的语言模型
def model_answer(prompt,local_model_name,SILICONFLOW_API_KEY, max_new_tokens=350, temperature=0.7,top_p=0.7, top_k=50, local_model=False):
    '''
        输入：prompt，local_model
        输出：回答（待统一类型）
    '''
    if local_model:
        # 本地HuggingFace LLM加载慢
        tokenizer = AutoTokenizer.from_pretrained(local_model_name)
        model = AutoModelForCausalLM.from_pretrained(local_model_name)

        # pipline封装了输入处理、模型推理、输出处理，每个样本返回一个字典，放入列表中返回
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer,max_new_tokens=max_new_tokens, temperature=temperature) # 指定 pipeline 类型为 text-generation，模型生成的最大 token 数256

        return  pipe(prompt) 
        
    else:
        
        url = "https://api.siliconflow.cn/v1/messages"

        payload = {
            "model": "deepseek-ai/DeepSeek-V3.1",
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": max_new_tokens,
            "system": "你是一个股票经纪人",
            "stop_sequences": [],
            "stream": False,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
        }
        headers = {
            "Authorization": f"Bearer {SILICONFLOW_API_KEY}",
            "Content-Type": "application/json"
        }

        response = requests.post(url, json=payload, headers=headers)
        response_dict = json.loads(response.text)
        return response_dict['content'][0]['text']



# 问答
def rag_answer(query):
    print("query:", query)
    print(inspect.getsource(retriever._get_relevant_documents))
    docs = retriever._get_relevant_documents(query,run_manager=None)
    context = " \n".join([f"{i}:{d.page_content}" for i,d in zip(range(1,top_relevant_k+1),docs)])
    print("context：\n", context)
    
    prompt = f"请根据以下知识库内容回答问题，并指明来源:\n知识库内容：\n{context}\n\n问题: {query} "
    
    print("输入：\n", prompt)
    print("\n输出：\n")
    return model_answer(prompt,local_model_name,SILICONFLOW_API_KEY, max_new_tokens=350, temperature=0.7,top_p=0.7, top_k=50, local_model=False)

 
data_directory = os.path.join(os.path.dirname(__file__), "data")
embedding_model_name = "Qwen/Qwen3-Embedding-0.6B"
collection_name = "test"
persist_directory = os.path.join(os.path.dirname(__file__), "chroma")
top_relevant_k=3
SILICONFLOW_API_KEY = os.environ.get("SILICONFLOW_API_KEY")
local_model_name = "google/gemma-2-2b-it"


docs = load_directory(data_directory)
chunks = split_text(docs, chunk_size=200, chunk_overlap=30)
db = store_chroma(chunks, embedding_model_name,collection_name, persist_directory)
if db is None:
    db = load_chroma(embedding_model_name, persist_directory, collection_name)
retriever = build_retriever(db, search_type="similarity", search_kwargs={"k": top_relevant_k})

# 测试
print(rag_answer("国家监督管理总局有什么新闻？"))