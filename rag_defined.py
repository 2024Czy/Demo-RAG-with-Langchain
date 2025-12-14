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
from serpapi import GoogleSearch





'''
待改进的点：
2、多文件上传和追加存储
3、噪声处理机制
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
        print("chroma.sqlite3不存在")
        return None
        
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

# 检索相关分块内容
def retrieve_relevant_chunks(query, retriever, top_relevant_k):
    '''
        输入：query，retriever，top_relevant_k
        输出：str
    '''
    docs = retriever._get_relevant_documents(query,run_manager=None) # list[Document]
    local_context = " \n".join([f"{i}:{d.page_content}" for i,d in zip(range(1,top_relevant_k+1),docs)])
    return local_context



# 网络搜索
def network_search(query, SERPAPI_KEY, engine="google_light",googl_domain="google.com",num=10):
    '''
        输入：query，top_k
        输出：
    '''
    params = {
    "engine": engine,
    "q": query,
    "google_domain": googl_domain,
    "hl": "zh-cn",
    "gl": "cn",
    "num":num,
    "api_key": SERPAPI_KEY
    }

    search = GoogleSearch(params)
    results = search.get_dict()
    organic_results = results["organic_results"]
    net_results = [ {"title": info["title"], "url": info["link"], "snippet": info["snippet"],'date':info['date'] if 'date' in info else None} for info in organic_results]
    net_context = "\n".join([f"{i}:{info['title']}\n{info['snippet']}\n{info['url']}\n{info['date']}\n" for i, info in enumerate(net_results,1)])
    return net_results, net_context

# 判断是否需要联网搜索
def need_net_search(query, local_context,local_model_name, SILICONFLOW_API_KEY):
    '''
        输入：query, local_context, local_model_name, Siliconflow_API_KEY
        输出：bool    
    '''
    prompt =build_net_search_prompt(query,local_context)
    response = model_answer(prompt,local_model_name,SILICONFLOW_API_KEY, max_new_tokens=350, temperature=0.7,top_p=0.7, top_k=50, local_model=False)
    print(f"模型判断是否需要联网：{response}")
    if  "TRUE" in str(response).upper():
        return True
    elif  "FALSE"  in str(response).upper():
        return False
    else:
        return "模型判断是否需要联网失败"

# 构建模型判断是否需要联网的提示词
def build_net_search_prompt(query,local_context):
    '''
        输入：query
        输出：str
    '''
    prompt = f"你只需要回答True或False。\n已知知识库：\n{local_context}\n\n对于问题: {query}，如果知识库中没有找到答案，回答:True; 如果知识库中找到答案，回答:False"
    return prompt


# 网络信息矛盾检测与可信源筛选, 本地知识库优先级最高默认完全正确，主要对网络结果做矛盾检测。
def detect_conflict_and_filter_net_results(query, SERPAPI_KEY, local_model_name, SILICONFLOW_API_KEY):
    '''
        输入：query，retriever，top_relevant_k,SERPAPI_KEY
        输出：检测结论,矛盾信息，可信信息
    '''
    net_results, _ = network_search(query, SERPAPI_KEY, engine="google_light", googl_domain="google.com", num=10)
    prompt = f"""
            你是一个网络信息一致性与证据筛选专家。
            给定以下网络搜索结果（net_results），它们描述的是同一主题或相关事实：

            net_results：
            {net_results}

            任务：
            1. 判断这些网络结果之间是否存在语义矛盾；
            2. 在此基础上，从网络结果中筛选出3条相对可信、可作为证据使用的信息；如果可信信息条数小于3条，可以返回低于3条的信息。
                如果没有可信信息，可以返回空数组。

            判定规则：

            【矛盾判定】
            - 如果不同结果对同一关键事实给出了互相冲突的描述（如时间、数值、身份、结论相反），视为存在矛盾；
            - 信息不完整、表述角度不同、时间精度不同，不视为矛盾；
            - 若无法判断，返回“不确定”。

            【可信度评估（仅基于给定信息）】
            优先选择具备以下特征的结果：
            - 网络域名是政府、企业、媒体、新闻网站等权威机构；
            - 描述具体、信息完整；
            - 与多数其他结果一致；
            - 时间标注明确且相对较新（如果提供了 date）；
            - 表述客观，不含明显猜测或模糊措辞。

            请严格按照以下步骤进行（不要在输出中展示思考过程）：
            1. 提取每条网络结果的核心事实；
            2. 比较这些事实，识别是否存在矛盾；
            3. 在无明显矛盾或冲突较小的结果中，筛选可信证据。

            输出格式（必须是合法 JSON，不要输出任何额外文本）：
            {{
            "has_contradiction": "true",
            "conflicting_pairs": [
                {{
                "snippet_a": "文本 A",
                "snippet_b": "文本 B",
                "conflict_reason": "冲突原因"
                }}
            ],
            "trusted_evidences": [
                {{
                "title": "标题",
                "url": "链接",
                "snippet": "摘要",
                "date": "日期或空字符串",
                "trust_reason": "可信原因"
                }}
            ],
            "summary": "整体判断说明"
            }}
            
            注意：
            - has_contradiction 只能是字符串："true"、"false" 或 "uncertain"
            - 如果没有冲突，conflicting_pairs 必须是 []
            - 如果没有可信证据，trusted_evidences 必须是 []
            - 所有字段都必须存在
        """
    response = model_answer(prompt, local_model_name, SILICONFLOW_API_KEY, max_new_tokens=5000, temperature=0.3, top_p=0.7, top_k=50, local_model=False)
    print(f"矛盾检测模型回复：\ntype:{type(response)}\n, {response}")
    
    try:
        response_dict = json.loads(response)
        print(f"矛盾检测模型回复json化：\n{response_dict}")
        has_contradiction = response_dict['has_contradiction']
        conflicting_pairs = response_dict['conflicting_pairs']
        trusted_evidences = response_dict['trusted_evidences']
        extracted_trusted_results = [
            {"title": info["title"], "url": info["url"], "snippet": info["snippet"], "date": info.get("date", "")}
            for info in trusted_evidences
        ]
    except json.JSONDecodeError as e:
        print(f"JSON解析错误: {e}")
        response_dict = {
            "has_contradiction": "uncertain",
            "conflicting_pairs": [],
            "trusted_evidences": [],
            "summary": "无法解析模型回复，返回不确定"
        }
        has_contradiction = "uncertain"
        conflicting_pairs = []
        extracted_trusted_results = []

    return has_contradiction, conflicting_pairs, extracted_trusted_results




# 构造模型回答提示词
def contruct_answer_prompt(query, local_context,local_model_name, SILICONFLOW_API_KEY):
    '''
        输入：query，local_context,local_model_name, SILICONFLOW_API_KEY
        输出：str
    '''
    need_net_search_flag  = need_net_search(query, local_context,local_model_name, SILICONFLOW_API_KEY)

    if need_net_search_flag :
        _, net_context = network_search(query,SERPAPI_KEY, engine="google_light",googl_domain="google.com",num=10)
        _, _, extracted_trusted_results = detect_conflict_and_filter_net_results(query,SERPAPI_KEY,local_model_name,SILICONFLOW_API_KEY)
        net_context = "\n".join([
            f"{i+1}. 标题: {info['title']}\n   URL: {info['url']}\n   摘要: {info['snippet']}\n   日期: {info['date']}\n"
            for i, info in enumerate(extracted_trusted_results)
        ])
        prompt = f"请根据以下网络搜索结果及知识库内容回答问题，并指明来源:\n网络搜索结果：\n{net_context}\n\n知识库内容：\n{local_context}\n\n问题: {query} "
        print(f"需要联网搜索，提示词：\n{prompt}")
        return prompt
    else:
        prompt = f"请根据以下知识库内容回答问题，并指明来源:\n知识库内容：\n{local_context}\n\n问题: {query} "
        print(f"不需要联网搜索，提示词：\n{prompt}")
        return prompt


# 模型回答

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
            "system": "",
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

        response = requests.post(url, json=payload, headers=headers, timeout=60)
        response_dict = json.loads(response.text)
        return response_dict['content'][0]['text']





if __name__ == '__main__':
    SILICONFLOW_API_KEY = os.environ.get("SILICONFLOW_API_KEY")
    SERPAPI_KEY = os.environ.get("SERPAPI_KEY")
    data_directory = os.path.join(os.path.dirname(__file__), "data")
    embedding_model_name = "Qwen/Qwen3-Embedding-0.6B"
    collection_name = "test"
    persist_directory = os.path.join(os.path.dirname(__file__), "chroma")
    top_relevant_k=3
    local_model_name = "google/gemma-2-2b-it"
    query = "美股能赚到钱吗"

    
    db = load_chroma(embedding_model_name, persist_directory, collection_name)
    if db is None:
        docs = load_directory(data_directory)
        chunks = split_text(docs, chunk_size=200, chunk_overlap=30)
        db = store_chroma(chunks, embedding_model_name,collection_name, persist_directory)
        db = load_chroma(embedding_model_name, persist_directory, collection_name)

    retriever = build_retriever(db, search_type="similarity", search_kwargs={"k": top_relevant_k})
    local_context = retrieve_relevant_chunks(query=query, retriever=retriever, top_relevant_k=top_relevant_k)

    prompt = contruct_answer_prompt(query=query, local_context=local_context,local_model_name=local_model_name, SILICONFLOW_API_KEY=SILICONFLOW_API_KEY)
    answer = model_answer(prompt,local_model_name,SILICONFLOW_API_KEY, max_new_tokens=350, temperature=0.7,top_p=0.7, top_k=50, local_model=False)
    print(f'\n问题：{query}\n回答：{answer}')
