import os
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from langchain_upstage import ChatUpstage
from langchain_upstage import UpstageEmbeddings
from pinecone import Pinecone, ServerlessSpec
from pydantic import BaseModel


# 0. 환경 변수 로드
load_dotenv()

# Pinecone 클라이언트 설정
pc = Pinecone(
    api_key=os.getenv("PINECONE_API_KEY"),
)

# 1. Pinecone 검색 : 이미 생성한 Pinecone 인덱스를 불러오기
vectorstore = LangchainPinecone.from_existing_index(
    index_name='ssafy-index',  # 기존에 사용한 인덱스 이름
    embedding=UpstageEmbeddings(model="solar-embedding-1-large")  # 동일한 임베딩 모델
)

# 2. Retriever 설정 : MMR(Minimal Marginal Relevance) 알고리즘을 사용하여 관련성 높은 문서 검색
retriever = vectorstore.as_retriever(
    search_type="mmr",  # mmr 알고리즘으로 검색
    search_kwargs={"k": 3, "threshold": 0.7}  # 검색 결과 상위 3개 반환, 유사도 임계값 0.7
)

# 3. 데이터 검색 및 응답 생성 : 데이터 검색 결과를 기반으로 upstage llm 모델을 호출하여 응답 생성성
# retrieal QA 구성
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatMessage(BaseModel):
    role: str
    content: str


class AssistantRequest(BaseModel):
    message: str
    thread_id: Optional[str] = None


class ChatRequest(BaseModel):
    messages: List[ChatMessage]  # Entire conversation for naive mode


class MessageRequest(BaseModel):
    message: str


@app.post("/chat")
async def chat_endpoint(req: MessageRequest):
    qa = RetrievalQA.from_chain_type(llm=ChatUpstage(model="solar-pro"),
                                     chain_type="stuff",
                                     retriever=retriever,
                                     return_source_documents=True)

    result = qa(req.message)
    return {"reply": result['result']}


@app.get("/health")
@app.get("/")
async def health_check():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)