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
from ollama import chat
from ollama import ChatResponse





'''
待改进的点：
3、噪声处理机制
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
            # print("查看详情：",new_docs)
            print(f"加载{file_path}成功")
            print(f"新增长度：{len(new_docs)}, 当前长度：{len(docs)}")
    return docs


# 文本切分
def split_text(documents:Iterable[Document], chunk_size=200, chunk_overlap=30):
    '''
        输入：list[Document]，切分参数
        输出：list[Document]
    '''
    splliter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap, add_start_index=True) # 循环递归切分器（段落-句子-字符），最大字符数为500，切分重叠度为50
    chunks = splliter.split_documents(documents)
    print("第一个分块：",chunks[0])
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
def store_chroma( embedding_model_name, collection_name, persist_directory, documents:Iterable[Document]=None, db=None, append_files_path = None):
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
        print("开始追加存储...")
        docs = load_directory(append_files_path)
        chunks = split_text(docs)
        embedding = load_encoder(embedding_model_name)
        db = db.from_documents(documents = chunks, embedding=embedding,collection_name=collection_name, persist_directory=persist_directory)
        print('数据库已追加存储成功')
        return db

# # 追加存储
# def append_store_chroma(db, append_files_path):
#     '''
#         输入：追加文件路径，embedding_function，persist_directory
#         输出：Chroma
#     '''
#     print("开始追加存储...")
#     docs = load_directory(append_files_path)
#     chunks = split_text(docs)
#     # db = load_chroma(embedding_model_name, persist_directory, collection_name)
#     db = db.from_documents()
#     print('数据库已追加存储成功')
#     return db


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
        输出：list[Document]，str，str
    '''
    docs = retriever._get_relevant_documents(query,run_manager=None) # list[Document]
    local_context = " \n".join([f"{i}:{d.page_content}" for i,d in zip(range(1,top_relevant_k+1),docs)])
    local_metadata = " \n".join([f"{i}:{d.metadata}" for i,d in zip(range(1,top_relevant_k+1),docs)])
    return docs, local_context, local_metadata



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
    response = model_answer(prompt,SILICONFLOW_API_KEY,local_model_name=local_model_name, max_new_tokens=350, temperature=0.7,top_p=0.7, top_k=50, local_model=False)
    print(f"模型判断是否需要联网：{response}")
    if  "TRUE" in str(response).upper():
        return True
    elif  "FALSE"  in str(response).upper():
        return False
    else:
        return "模型判断是否需要联网失败"
    
 # 构建网络搜索提示
def build_net_search_prompt(query, local_context): # 这里可以用检索块与问题的相关性得分来改进
    prompt = f"""
    你是一个【是否需要联网】的判定模块。

    你的任务是判断：**仅基于本地知识库内容，是否能够完成对用户问题的一次有观点的回答。**

    你只需要回答 True 或 False，不要输出任何解释。

    ====================
    本地知识库检索内容（local_context）：
    --------------------
    {local_context}
    --------------------

    ====================
    判定标准（对称、无倾向）：

    【回答 False（不需要联网）】
    当且仅当满足以下条件：
    - 本地知识库中包含与问题相关的核心对象、事实或概念；

    【回答 True（需要联网）】
    当且仅当满足以下条件：
    - 本地知识库中缺少问题所需的关键事实或核心信息；
    - 在不引入外部信息的前提下，无法形成一个逻辑自洽的回答；
    - 回答只能是“无法判断”“没有相关信息”。

    问题：
    {query}
    """


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
    - 输出的内容必须严格遵循上述 JSON 格式，不要包含任何额外的文本或字符
    - 生成的 JSON 请使用双引号，并确保所有字符串值都用双引号括起来
    - 确保生成的 JSON 内容中没有多余的逗号或其他语法错误
    """
    response = model_answer(prompt, SILICONFLOW_API_KEY,local_model_name=local_model_name, max_new_tokens=5000, temperature=0.3, top_p=0.7, top_k=50, local_model=False)
    print(f"矛盾检测模型回复：\ntype:{type(response)}\n, {response}")
    
    try:
        response_dict = json.loads(response)
        print("---------------------网络输出解析成功----------------------")
        # print(f"矛盾检测模型回复json化：\n{response_dict}")
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




# 构造模型回答提示词，提示词这里是一个需要不断根据出现的问题去迭代优化的地方
def contruct_answer_prompt(query, local_context, local_metadata, local_model_name, SILICONFLOW_API_KEY):
    '''
        输入：query，local_context, local_metadata, local_model_name, SILICONFLOW_API_KEY
        输出：str
    '''
    need_net_search_flag  = need_net_search(query, local_context,local_model_name, SILICONFLOW_API_KEY)

    if need_net_search_flag :
        # _, net_context = network_search(query,SERPAPI_KEY, engine="google_light",googl_domain="google.com",num=10)
        _, _, extracted_trusted_results = detect_conflict_and_filter_net_results(query,SERPAPI_KEY,local_model_name,SILICONFLOW_API_KEY)
        valid_net_context = "\n".join([
            f"{i+1}. 标题: {info['title']}\n   URL: {info['url']}\n   摘要: {info['snippet']}\n   日期: {info['date']}\n"
            for i, info in enumerate(extracted_trusted_results)
        ])
        prompt = f"""
        你是一个【严格遵守引用规范】的基于证据回答问题的专家。

        你的回答必须满足**可审计、可复现、可人工核查**的要求。

        ====================
        用户问题：
        {query}

        ====================
        网络搜索结果（net_context）：
        每条包含 标题、URL、摘要、日期（如有）
        {valid_net_context}

        ====================
        本地知识库内容（local_context）：
        以下为多个知识库分块（chunk）的原始内容
        {local_context}

        ====================
        本地知识库分块对应的元信息（local_metadata）：
        以下为与 local_context 一一对应的元信息对象，
        字段仅包含：source，start_index（必有），page（如有）

        {local_metadata}

        ====================
        回答与引用规则（必须严格遵守，否则视为失败）：

        【内容约束】
        1. 回答只能基于以上信息，不允许使用任何外部或常识性补充；
        2. 任何无法明确找到来源支撑的事实，禁止出现在回答中；

        【引用编号规则】
        3. 回答正文中的引用编号必须：
        - 从【1】开始；
        - 严格连续递增（1,2,3,...）；
        - 不允许跳号、不允许重复、不允许缺失；
        4. 每一个引用编号必须且只能对应【来源说明】中的一条来源；
        5. 如果你发现编号无法做到严格连续，请**重新生成整个回答**；

        【网络信息引用规则】
        6. 网络来源在【来源说明】中只允许输出对应的 URL，不允许任何改写或补充；

        【本地知识库引用规则（非常重要）】
        7. 引用本地知识库时必须从 local_metadata 中选择元信息
        - 示例：{{'source': 'd:\\MyGithubProjects\\test\\rag_test\\data\\06.循环神经网络和自然语言处理.pptx.pdf','start_index': 200, 'page': 5}}
        8. 如果某条本地知识库元信息不适合直接作为来源，请不要使用该 chunk；

        【一致性校验要求】
        9. 【回答】中出现的每一个引用编号，必须在【来源说明】中出现；
        10. 【来源说明】中不允许出现正文未使用的引用编号
        11、【来源说明】中的数量要和正文中引用的数量相等；【非常重要】
        
        如果上述1-11条任一规则无法满足，请重新生成答案，而不是输出一个近似结果。

        ====================
        输出格式（必须严格遵守，不要输出任何多余内容）：

        【回答格式】如下：

        根据公开资料显示，公司 X 的 CEO 为 Alice Smith，自 2023 年起开始任职【1】。
        任职期间，Alice Smith 主要负责公司 X 的业务发展，主要包括销售、市场营销、人力资源、财务等方面。
        本地知识库中同样提到 Alice Smith 于 2023 年被任命为公司 X 的 CEO【2】。

        【来源说明】
        [1] https://www.companyx.com/news/ceo-announcement
        [2] {{"source":"company_profile.pdf","start_index":2480,"page":12}}

        """

        # prompt = f"请根据以下网络搜索结果及知识库内容回答问题，并根据网络搜索结果链接和知识库元数据指明来源:\n网络搜索结果：\n{valid_net_context}\n\n知识库内容：\n{local_context}\n知识库对应的元数据：\n{local_metadata}\n\n问题: {query} "
        print(f"需要联网搜索，提示词：\n{prompt}")
        return prompt
    else:
        prompt = f"""
        你是一个【严格遵守引用规范】的基于证据回答问题的专家。

        你的回答必须满足**可审计、可复现、可人工核查**的要求。

        ====================
        用户问题：
        {query}

        ====================
        本地知识库内容（local_context）：
        以下为多个知识库分块（chunk）的原始内容
        {local_context}

        ====================
        本地知识库分块对应的元信息（local_metadata）：
        以下为与 local_context 一一对应的元信息对象，
        字段仅包含：source，start_index（必有），page（如有）

        {local_metadata}

        ====================
        回答与引用规则（必须严格遵守，否则视为失败）：

        【内容约束】
        1. 回答只能基于以上信息，不允许使用任何外部或常识性补充；
        2. 任何无法明确找到来源支撑的事实，禁止出现在回答中；

        【引用编号规则】
        3. 回答正文中的引用编号必须：
        - 从【1】开始；
        - 严格连续递增（1,2,3,...）；
        - 不允许跳号、不允许重复、不允许缺失；
        4. 每一个引用编号必须且只能对应【来源说明】中的一条来源；
        5. 如果你发现编号无法做到严格连续，请**重新生成整个回答**；


        【本地知识库引用规则（非常重要）】
        6. 引用本地知识库时必须从 local_metadata 中选择元信息
        - 示例：{{'source': 'd:\\MyGithubProjects\\test\\rag_test\\data\\06.循环神经网络和自然语言处理.pptx.pdf','start_index': 200, 'page': 5}}

        7. 如果某条本地知识库元信息不适合直接作为来源，请不要使用该 chunk；

        【一致性校验要求】
        8. 【回答】中出现的每一个引用编号，必须在【来源说明】中出现；
        9. 【来源说明】中不允许出现正文未使用的引用编号；
        10.【来源说明】中的数量要和正文中引用的数量相等；【非常重要】
        
        如果上述1-10条任一规则无法满足，请重新生成答案，而不是输出一个近似结果。

        ====================
        输出格式（必须严格遵守，不要输出任何多余内容）：

        【回答格式】如下：
        
        根据本地知识库资料显示，公司 X 当前的首席执行官为 Alice Smith【1】。
        Alice Smith 于 2023 年正式被任命为公司 X 的 CEO，并接替前任管理层。
        在其上任后，公司 X 的管理层结构进行了调整，高层管理团队的职责划分更加清晰【2】。

        【来源说明】
        [1] {{"source":"company_profile.pdf","start_index":820,"page":5}}
        [2] {{"source":"shi.docx","start_index":1320}}

        """

        # prompt = f"请根据以下知识库内容回答问题，并根据知识库元数据指明来源:\n知识库内容：\n{local_context}\n\n问题: {query} "
        print(f"不需要联网搜索，提示词：\n{prompt}")
        return prompt


# 模型回答，根据最终模型对问题的回答的有效性来纠正是否联网的判断是一个优化点，相当于对回答进行一次检验。
def model_answer(prompt,SILICONFLOW_API_KEY,local_model_name ="google/gemma-2-2b-it", max_new_tokens=350, temperature=0.7,top_p=0.7, top_k=50, local_model=False):
    '''
        输入：prompt，local_model
        输出：回答（待统一类型）
    '''
    if local_model: # 加载慢
        # # 方案一：调用HuggingFace LLM
        # tokenizer = AutoTokenizer.from_pretrained(local_model_name)
        # model = AutoModelForCausalLM.from_pretrained(local_model_name)

        # # pipline封装了输入处理、模型推理、输出处理，每个样本返回一个字典，放入列表中返回
        # pipe = pipeline("text-generation", model=model, tokenizer=tokenizer,max_new_tokens=max_new_tokens, temperature=temperature) # 指定 pipeline 类型为 text-generation，模型生成的最大 token 数256

        # return  pipe(prompt) 
        
        # 方案二：调用ollama部署在本地的模型,这里用gemma3:270m做个举例，实际不要参数这么少的
        response: ChatResponse = chat(model='gemma3:270m', messages=[
            {
                'role': 'user',
                'content': prompt,
            },
            ])
        return response['message']['content']
        
        
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
    append_files_path = os.path.join(os.path.dirname(__file__), "append_data")
    embedding_model_name = "Qwen/Qwen3-Embedding-0.6B"
    collection_name = "test"
    persist_directory = os.path.join(os.path.dirname(__file__), "chroma3")
    top_relevant_k= 10
    local_model_name = "google/gemma-2-2b-it"
    query = "清华大学在哪里"    # 这里可以做一个长文测试和错别字测试，会有优化点

    
    db = load_chroma(embedding_model_name, persist_directory, collection_name)
    if db is None:
        docs = load_directory(data_directory)
        chunks = split_text(docs, chunk_size=200, chunk_overlap=30)
        db = store_chroma( embedding_model_name,collection_name, persist_directory,documents=chunks)
        db = load_chroma(embedding_model_name, persist_directory, collection_name)
    # 追加
    # db = store_chroma(embedding_model_name,collection_name, persist_directory, db=db, append_files_path= append_files_path)
    retriever = build_retriever(db, search_type="similarity", search_kwargs={"k": top_relevant_k})
    docs, local_context, local_metadata = retrieve_relevant_chunks(query=query, retriever=retriever, top_relevant_k=top_relevant_k)

    prompt = contruct_answer_prompt(query=query, local_context=local_context,local_metadata=local_metadata,local_model_name=local_model_name, SILICONFLOW_API_KEY=SILICONFLOW_API_KEY)
    answer = model_answer(prompt,SILICONFLOW_API_KEY,local_model_name=local_model_name, max_new_tokens=350, temperature=0.7,top_p=0.7, top_k=50, local_model=True)
    print(f'\n问题：{query}\n回答：{answer}')
