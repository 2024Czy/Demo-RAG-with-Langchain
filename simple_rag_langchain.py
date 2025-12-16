from langchain_community.document_loaders import Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_chroma import Chroma

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import inspect

# 加载文件
loader = Docx2txtLoader("11-13晚.docx") # loader是类实例， Docx2txtLoader是一个类，用于加载docx文件
docs = loader.load() # docs -> List[Document((page_content, metadata))]，长度为文件数
print('文档内容：\n',docs[0].page_content)

# 切分文本
splliter = RecursiveCharacterTextSplitter(chunk_size=100,chunk_overlap=10) # 循环递归切分器（段落-句子-字符），最大字符数为500，切分重叠度为50
chunks = splliter.split_documents(docs) # List[Document((page_content, metadata))]，长度为分块数
for i, chunk in enumerate(chunks):
    print(f"第{i+1}个分块：\n{chunk.page_content}\t {len(chunk.page_content)}")


# 向量化
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2") # 类实例
db = Chroma.from_documents(chunks, embedding) 
print('db:\n',db)

# 构建检索器, langchain_core/vectorstores/base.py源码内部还没实现similarity_search_with_score()这一方法
retriever = db.as_retriever(search_kwargs={"k": 3})
print(inspect.getsourcefile(db.as_retriever))

# 本地HuggingFace LLM
model_name = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# pipline封装了输入处理、模型推理、输出处理，每个样本返回一个字典，放入列表中返回
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer,max_new_tokens=256) # 指定 pipeline 类型为 text-generation，模型生成的最大 token 数256

# HuggingFacePipeline封装了对pipline输出的处理，返回字符串
llm = HuggingFacePipeline(pipeline=pipe)

# 问答
def rag_answer(query):
    print("query:", query)
    print(inspect.getsource(retriever._get_relevant_documents))
    docs = retriever._get_relevant_documents(query,run_manager=None)
    context = " \n\n".join([d.page_content for d in docs])
    print("context\n:", context)
    
    prompt = f"问题: {query} \n知识库内容：\n{context}\n\n请基于知识库内容回答:"
    # return pipe(prompt)
    # return llm._generate(prompt).generations[0][0].text
    return llm._generate(prompt)


# 测试
print(rag_answer("11月12日沪指涨了多少？"))