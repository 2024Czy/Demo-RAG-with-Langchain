from langchain_community.document_loaders import Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from chromadb.config import Settings
from langchain_chroma import Chroma
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import inspect
import os
import requests
import json

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



# 自定义存储位置
persist_directory = os.path.join(os.path.dirname(__file__), "chroma2")
# 向量化
embedding = HuggingFaceEmbeddings(
    model_name="Qwen/Qwen3-Embedding-0.6B",
)

if not os.path.exists(os.path.join(persist_directory, "chroma.sqlite3")):
    print("chroma.sqlite3不存在，创建...")
    os.makedirs(persist_directory, exist_ok=True)
    # 加载文件
    loader = Docx2txtLoader("11-13晚.docx") # loader是类实例， Docx2txtLoader是一个类，用于加载docx文件
    docs = loader.load() # docs -> List[Document((page_content, metadata))]，长度为文件数
    print('文档内容：\n',docs[0].page_content)

    # 切分文本
    splliter = RecursiveCharacterTextSplitter(chunk_size=100,chunk_overlap=10) # 循环递归切分器（段落-句子-字符），最大字符数为500，切分重叠度为50
    chunks = splliter.split_documents(docs) # List[Document((page_content, metadata))]，长度为分块数
    for i, chunk in enumerate(chunks):
        print(f"第{i+1}个分块：\n{chunk.page_content}\t {len(chunk.page_content)}")

    
    db = Chroma.from_documents(documents=chunks,embedding=embedding, persist_directory=persist_directory) 
   
    print('数据库已存储',db)
else:
    print("chroma.sqlite3已存在，加载...")
    # 这里要注意的是如果设置collection_name要和之前的一致，默认是"langchain"，否则会重新初始化一个数据库;同时要显式指定嵌入模型
    db = Chroma(
        embedding_function=embedding,
        persist_directory=persist_directory
    )
   
    
    
# 构建检索器
top_k=3
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": top_k})
print(inspect.getsourcefile(db.as_retriever))


# 选择语言模型
def model_answer(prompt,local_model=False):
    if local_model:
        # 本地HuggingFace LLM加载慢
        model_name = "google/gemma-2-2b-it"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)

        # pipline封装了输入处理、模型推理、输出处理，每个样本返回一个字典，放入列表中返回
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer,max_new_tokens=350, temperature=0.5) # 指定 pipeline 类型为 text-generation，模型生成的最大 token 数256

        return  pipe(prompt)
        
    else:
        SILICONFLOW_API_KEY = os.environ.get("SILICONFLOW_API_KEY")
        
        url = "https://api.siliconflow.cn/v1/messages"

        payload = {
            "model": "deepseek-ai/DeepSeek-V3.1",
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": 8192,
            "system": "你是一个股票经纪人",
            "stop_sequences": [],
            "stream": False,
            "temperature": 0.7,
            "top_p": 0.7,
            "top_k": 50,
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
    context = " \n".join([f"{i}:{d.page_content}" for i,d in zip(range(1,top_k+1),docs)])
    print("context：\n", context)
    
    prompt = f"请根据以下知识库内容回答问题，并指明来源:\n知识库内容：\n{context}\n\n问题: {query} "
    
    print("输入：\n", prompt)
    print("\n输出：\n")
    return model_answer(prompt)
    


# 测试
print(rag_answer("国家监督管理总局有什么新闻？"))