from dotenv import load_dotenv
import os
from langchain_community.vectorstores import Pinecone
from langchain_upstage import UpstageDocumentParseLoader, UpstageEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import LLMChain
from langchain.prompts.chat import ChatPromptTemplate
from langchain.output_parsers import RetryOutputParser
from pinecone import Pinecone, ServerlessSpec

# 1. 환경 변수 로드
load_dotenv()

# Pinecone 클라이언트 설정
pc = Pinecone(
    api_key=os.getenv("PINECONE_API_KEY"),
)

# Pinecone 인덱스 생성
index_name = "document-index"
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1024,
        metric="euclidean",
        spec=ServerlessSpec(
            cloud="aws",
            region=os.getenv("PINECONE_ENVIRONMENT"),
        ),
    )

# 2. 데이터 수집 및 문서 로드
directory_path = r"C:\\Users\\SSAFY\\Desktop\\AI\\data"
file_paths = [os.path.join(directory_path, file) for file in os.listdir(directory_path) if file.endswith('.pdf')]

print(f"Files to process: {file_paths}")

# 문서 로드
loaded_documents = []
for file_path in file_paths:
    try:
        loader = UpstageDocumentParseLoader(file_path, output_format='html', coordinates=False)
        document = loader.load()
        loaded_documents.append(document)
        print(f"Loaded document: {file_path}")
    except Exception as e:
        print(f"Error loading {file_path}: {e}")

# 3. 청크 분할
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = []
for doc in loaded_documents:
    texts.extend(text_splitter.split_documents(doc))

print(f"Total text chunks: {len(texts)}")

# 4. 임베딩 생성 및 벡터 DB 구축
embeddings = UpstageEmbeddings(model="solar-embedding-1-large")
vectorstore = Pinecone.from_texts(
    texts=[text.page_content for text in texts],
    embedding=embeddings,
    index_name=index_name
)

# 5. Retriever 설정
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 4, "threshold": 0.7}
)

# 6. LLM 프롬프트 및 체인 정의
prompt = ChatPromptTemplate.from_messages([
    ("system", """
        너는 인공지능 챗봇으로, 주어진 문서를 정확하게 이해해서 답변을 해야 해.
        문서에 있는 내용으로만 답변하고 내용이 없다면, 잘 모르겠다고 답변해.
        ---
        CONTEXT:
        {context}
    """),
    ("human", "{input}")
])

from langchain_upstage import ChatUpstage
llm = ChatUpstage(model="solar-pro")
chain = LLMChain(prompt=prompt, llm=llm, output_parser=RetryOutputParser())

# 7. 질문 및 답변 생성
query = "국민은행의 이번 주요 사업은 무엇인가요?"
retrieved_docs = retriever.invoke(query)

response = chain.invoke({"context": retrieved_docs, "input": query})
print(response)
