from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import os
from PyPDF2 import PdfReader
from langchain.chains.question_answering import load_qa_chain
from langchain_community.chat_models import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import numpy as np
import faiss
from dotenv import load_dotenv
from openai import AsyncOpenAI

import openai  # 직접 openai 라이브러리 사용

# FastAPI 애플리케이션 생성 및 static, templates 설정
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# WebSocket 연결 관리자
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

# API 키 설정 (.env 파일에서 로드)
load_dotenv()
api_key = os.getenv('API_KEY')
openai.api_key = api_key  # openai 라이브러리에서 직접 설정

# PDF 파일 처리 및 임베딩 생성 (글로벌)
folder_dir = './pdf'
pdfs = os.listdir(folder_dir)
raw_text = ""
for pdf_file in pdfs:
    reader = PdfReader(os.path.join(folder_dir, pdf_file))
    for page in reader.pages:
        text = page.extract_text()
        if text:
            raw_text += text

# 임베딩 및 FAISS 벡터 데이터베이스 생성
embeddings_model = OpenAIEmbeddings()
texts = raw_text.split("\n")
embeddings = embeddings_model.embed_documents(texts)
dimension = len(embeddings[0])
faiss_index = faiss.IndexFlatL2(dimension)
faiss_index.add(np.array(embeddings))
print(faiss_index)

# 벡터 데이터베이스를 이용한 검색 함수
def search_documents(query, k=5):
    query_embedding = embeddings_model.embed_query(query)
    distances, indices = faiss_index.search(np.array([query_embedding]), k)
    return [texts[i] for i in indices[0]]

# GPT 응답을 스트리밍으로 생성하여 WebSocket으로 전송하는 비동기 함수
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

async def get_response(prompt: str, websocket: WebSocket):
    # 관련 문서를 검색하여 프롬프트에 추가
    relevant_docs = search_documents(prompt)
    combined_docs = "\n".join(relevant_docs)
    final_prompt = f"{combined_docs}\n\nQuestion: {prompt}"
    messages = [{"role": "user", "content": final_prompt}]

    async def event_stream():
        stream = await client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            stream=True,
            temperature=0.2,
            max_tokens=1000,
            presence_penalty=0.3,
            frequency_penalty=0.5,
        )
        try:
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    token = chunk.choices[0].delta.content
                    if token:
                        await websocket.send_text(token)
                    
        except Exception as e:
            error_message = f"\n[Error: {str(e)}]"
            print(error_message)
            await websocket.send_text(error_message)

    await event_stream()

@app.get("/test")
async def test():
    return {"message": "hello FastAPI!"}

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: int):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # 사용자가 보낸 메시지는 그대로 표시
            await manager.send_personal_message(f"You wrote: {data}", websocket)
            # GPT 응답을 스트리밍 방식으로 전송
            await get_response(data, websocket)
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        await manager.broadcast(f"Client #{client_id} left the chat")