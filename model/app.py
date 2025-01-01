import os
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from langchain.prompts import ChatPromptTemplate
from langchain_upstage import ChatUpstage
from langchain_upstage import UpstageEmbeddings
from pinecone import Pinecone
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

# 6. 프롬프트 정의
prompt = ChatPromptTemplate.from_messages([
    ("system", """
        너는 기업 신년사 PDF 문서로 학습된 기업 분석 AI 챗봇입니다. 사용자 질문에 대해 문서에 있는 내용으로 답변하세요.
        문서에 없는 내용이라면 '알 수 없습니다'라고 답변하세요.
        아래는 출력 형식 예제입니다. 항상 동일한 형식으로 출력해야 합니다.
        주요 키워드는 3개 이하, 지원 관련 경험 추천도 3개 이하입니다.
        
        ### 출력 형식 예제
        | **구분**            | **내용**                                                                                 |
        |---------------------|-----------------------------------------------------------------------------------------|
        | **정보 요약**       | LG의 신년사에서는 고객 중심 경영 강화, 신사업 영역 확대(AI 및 친환경 기술), 내부 디지털 전환 가속화가 주요 전략으로 제시되었습니다. |
        | **주요 키워드**     | 고객 중심, 신사업(AI, 친환경), 디지털 전환, 글로벌 시장 확대                      |
        | **지원 관련 경험 추천** | - 고객 중심 경영: 과거 고객 데이터를 분석하여 서비스 개선 경험<br>- 신사업 영역: AI 기반 프로젝트에 참여하여 새로운 서비스 설계 경험<br>- 디지털 전환: 기업 내 시스템 디지털화 프로젝트 수행 경험 |

        문서가 없는 경우의 출력 예제:
        | **구분**            | **내용**                     |
        |---------------------|-----------------------------|
        | **정보 요약**       | 문서에 해당 질문에 대한 정보가 없습니다. |
        | **주요 키워드**     | 없음                        |
        | **지원 관련 경험 추천** | 없음                        |

        항상 위와 같은 형식을 유지하세요.
        ---
        CONTEXT:
        {context}
     """
    ),
    ("human", "{input}")
])

# 3. 데이터 검색 및 응답 생성 : 데이터 검색 결과를 기반으로 upstage llm 모델을 호출하여 응답 생성
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
    qa = RetrievalQA.from_chain_type(
        llm=ChatUpstage(model="solar-pro"),
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}  # 프롬프트 추가
    )

    result = qa(req.message)
    return {"reply": result['result']}


@app.get("/health")
@app.get("/")
async def health_check():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
