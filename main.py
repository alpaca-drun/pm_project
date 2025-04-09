from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import os
import json
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# FastAPI 애플리케이션 생성 및 static, templates 설정
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# OpenAI API 키 설정 (환경 변수에서 가져옴)
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY가 .env 파일에 설정되어 있지 않습니다.")
os.environ["OPENAI_API_KEY"] = openai_api_key

# Vector DB 경로 지정
os.makedirs("vectorstore", exist_ok=True)


# 마크다운 파일 불러오기 및 처리
loader = TextLoader("C:/Users/k_373/Desktop/PM9/test2.md", encoding="utf-8")
documents = loader.load()

# 마크다운 직업 기준 분할
headers_to_split_on = [
    ("#", "직업")
]
markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on,
    strip_headers=False
)
# 텍스트 추출 및 분할
docs = []
for doc in documents:
    splits = markdown_splitter.split_text(doc.page_content)
    for split in splits:
        docs.append(split)

# 임베딩 모델 생성 및 벡터 DB 생성
embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")

vectorstore = FAISS.from_documents(docs, embedding_model)

# 임베딩 저장
vectorstore.save_local("vectorstore/faiss_index")

# 불러올 때
if os.path.exists("vectorstore/faiss_index"):
    vectorstore = FAISS.load_local("vectorstore/faiss_index", embedding_model, allow_dangerous_deserialization=True)
else:
    # 임베딩 생성 코드
    vectorstore = FAISS.load_local("vectorstore/faiss_index", embedding_model, allow_dangerous_deserialization=True)
    vectorstore.save_local("vectorstore/faiss_index")


retriever = vectorstore.as_retriever()

# LLM 모델 설정
llm = ChatOpenAI(temperature=0, streaming=True, model_name="gpt-4o")

# RIASEC 검사 결과 해석 LLM
async def interpret_survey_results(survey_results):
    try:
        # JSON 문자열 파싱
        survey_data = json.loads(survey_results)
        scores = survey_data.get("scores", {})
        
        # RIASEC 점수 추출
        r_score = scores.get("R", 0)
        i_score = scores.get("I", 0)
        a_score = scores.get("A", 0)
        s_score = scores.get("S", 0)
        e_score = scores.get("E", 0)
        c_score = scores.get("C", 0)
        
        # 가장 높은 점수 유형 찾기
        max_score = max(r_score, i_score, a_score, s_score, e_score, c_score)
        dominant_types = []
        
        if r_score == max_score:
            dominant_types.append("현실형(R)")
        if i_score == max_score:
            dominant_types.append("탐구형(I)")
        if a_score == max_score:
            dominant_types.append("예술형(A)")
        if s_score == max_score:
            dominant_types.append("사회형(S)")
        if e_score == max_score:
            dominant_types.append("기업형(E)")
        if c_score == max_score:
            dominant_types.append("관습형(C)")
        
        dominant_types_str = ", ".join(dominant_types)
        
        survey_interpreter = ChatOpenAI(temperature=0.2, model_name="gpt-4o")
        
        messages = [
            SystemMessage(content="""
            너는 RIASEC 직업 흥미 유형 검사 결과를 해석하는 전문가야.
            학생의 RIASEC 점수를 바탕으로 그들의 성격, 관심사, 강점 및 진로 방향에 대한 통찰력 있는 해석을 제공해줘.
            각 유형별 특징과 그에 맞는 직업군, 학과 추천도 포함해줘.
            
            RIASEC 유형 설명:
            - 현실형(R): 실제적이고 신체적인 활동을 선호하며, 기계나 도구 다루는 것을 좋아함
            - 탐구형(I): 논리적, 분석적 사고를 좋아하며 지적 호기심이 많음
            - 예술형(A): 창의성과 자기표현을 중시하며 관습에 얽매이지 않음
            - 사회형(S): 사람들을 돕고 가르치는 활동을 선호함
            - 기업형(E): 리더십을 발휘하고 목표 달성을 위해 타인을 설득하는 것을 즐김
            - 관습형(C): 체계적이고 조직적인 활동을 선호하며 정확성을 중시함
            """),
            HumanMessage(content=f"""
            다음은 학생의 RIASEC 검사 결과야:
            
            현실형(R): {r_score}
            탐구형(I): {i_score}
            예술형(A): {a_score}
            사회형(S): {s_score}
            기업형(E): {e_score}
            관습형(C): {c_score}
            
            가장 높은 점수 유형: {dominant_types_str}
            
            이 결과를 해석해주세요.
            """)
        ]
        
        response = await survey_interpreter.ainvoke(messages)
        return response.content
    except Exception as e:
        return f"설문 결과 해석에 실패하였습니다 {str(e)}"

# WebSocket 연결 관리자
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []
        self.memories = {} 
        self.survey_interpretations = {}

    async def connect(self, websocket: WebSocket, client_id: int):
        await websocket.accept()
        self.active_connections.append(websocket)
        
        # 클라이언트별 메모리 초기화
        if client_id not in self.memories:
            memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )
            # 메모리 초기화
            self.memories[client_id] = memory
            # 설문 해석 결과 초기화
            self.survey_interpretations[client_id] = ""

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

@app.get("/test")
async def test():
    return {"message": "hello FastAPI!"}

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: int):
    await manager.connect(websocket, client_id)
    try:
        # 첫 메시지를 설문 결과로 간주
        survey_results = await websocket.receive_text()
        
        # 설문 결과 해석
        interpretation = await interpret_survey_results(survey_results)
        
        # 해석 결과 저장
        manager.survey_interpretations[client_id] = interpretation
        
        # 설문 해석 결과 응답 전송
        await manager.send_personal_message(f"### RIASEC 검사 결과 분석 ###\n\n{interpretation}\n\n이제 진로에 대해 질문해주세요.", websocket)
        
        while True:
            # 사용자 메시지 수신
            query = await websocket.receive_text()
            
            # 해당 클라이언트의 대화 기록 가져오기
            memory = manager.memories[client_id]
            chat_history = memory.chat_memory.messages
            
            # 대화 기록 포맷팅
            formatted_history = ""
            for message in chat_history:
                if isinstance(message, HumanMessage):
                    formatted_history += f"사용자: {message.content}\n"
                elif isinstance(message, AIMessage):
                    formatted_history += f"어시스턴트: {message.content}\n"
            
            # 관련 문서 검색
            relevant_docs = retriever.invoke(query)
            context = "\n\n".join([doc.page_content for doc in relevant_docs])
            
            # 설문 해석 결과 가져오기
            survey_interpretation = manager.survey_interpretations[client_id]
            
            # 진로 상담 챗봇
            messages = [
                SystemMessage(content=f"""
                너는 직업과 학교, 학과를 알려주는 진로 상담 전문가야.
                
                반드시 이전 대화 내용을 기억하고 맥락을 유지해서 답변해.
                
                학생의 RIASEC 검사 결과 해석:
                {survey_interpretation}
                
                이전 대화 내용:
                {formatted_history}
                
                직업, 학교, 학과 정보:
                {context}
                
                답변은 반드시 한국어로 제공해야 한다.
                """),
                HumanMessage(content=query)
            ]
            
            # LLM 질문 및 응답
            response = llm.invoke(messages)
            answer = response.content
            
            # 응답 전송
            await manager.send_personal_message(answer, websocket)
            
            # 대화 내용 메모리 저장
            memory.chat_memory.add_user_message(query)
            memory.chat_memory.add_ai_message(answer)
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)