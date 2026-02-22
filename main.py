import os
import io
import uuid
from datetime import datetime
from typing import List, Dict, Optional

import uvicorn
import PIL.Image
import google.generativeai as genai
import aiofiles

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel

# Импорты твоих локальных модулей
from db.database import engine, Base, get_db
from db.repository import RoutingRepository
from db.models import RoutingHistory, RoutedTicketModel # Убедись, что RoutedTicketModel импортирован для AI Assistant
from services.parser import CSVParser
from services.pipeline import TicketPipeline
from ai_service.nlp_module import generate_chart_data

# Твои Pydantic схемы (убедись, что они есть в schemas/models.py)
from schemas.models import TicketCreate, RoutedTicket, AIAnalysisResult

# --- Инициализация ИИ (Gemini) ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "твой_ключ_здесь_если_нет_в_env")
genai.configure(api_key=GEMINI_API_KEY)
MODEL_NAME = 'gemini-3-flash-preview' 
model = genai.GenerativeModel(MODEL_NAME)

# --- Инициализация БД ---
# Магия хакатона: автоматическое создание всех таблиц в PostgreSQL при запуске!
Base.metadata.create_all(bind=engine)

app = FastAPI(title="F.I.R.E. API", version="1.0.0")

# Создаем папку для скриншотов
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


# --- СХЕМЫ ДАННЫХ ДЛЯ ЭНДПОИНТОВ ---
class ModelInfo(BaseModel):
    name: str
    version: str
    description: Optional[str]
    supported_methods: List[str]

class SystemModelsResponse(BaseModel):
    remote_models_gemini: List[ModelInfo]
    local_models_ollama: List[Dict]
    status: str

class BugAnalysisResponse(BaseModel):
    report_id: str
    status: str
    timestamp: datetime
    file_info: dict  # Путь к файлу, имя, размер
    analysis: Optional[dict] = None  # Распарсенный ответ от Gemini
    raw_text: Optional[str] = None   # Полный текст ответа на случай ошибки парсинга
    error_details: Optional[str] = None

class ChartRequest(BaseModel):
    query: str


# ==========================================
# ЭНДПОИНТЫ: БАЗОВЫЕ ПРОВЕРКИ
# ==========================================
@app.get("/health")
def health_check():
    return {"status": "ok", "service": "fire-backend"}


# ==========================================
# ЭНДПОИНТЫ: МАССОВАЯ МАРШРУТИЗАЦИЯ (CSV)
# ==========================================
@app.post("/api/v1/route-tickets/execute")
async def execute_mass_routing(db: Session = Depends(get_db)):
    try:
        # Проверяем наличие всех 3 файлов на диске
        paths = {
            "units": os.path.join(UPLOAD_DIR, "latest_units.csv"),
            "managers": os.path.join(UPLOAD_DIR, "latest_managers.csv"),
            "tickets": os.path.join(UPLOAD_DIR, "latest_tickets.csv")
        }
        
        for name, path in paths.items():
            if not os.path.exists(path):
                raise HTTPException(status_code=400, detail=f"Не найден файл {name}. Загрузите его сначала.")

        # Парсим файлы (в идеале CSVParser тоже сделать асинхронным или запускать через run_in_threadpool)
        with open(paths["units"], 'rb') as f_units, \
             open(paths["managers"], 'rb') as f_managers, \
             open(paths["tickets"], 'rb') as f_tickets:
            
            offices = CSVParser.parse_business_units(f_units)
            managers = CSVParser.parse_managers(f_managers)
            tickets = CSVParser.parse_tickets(f_tickets)
        
        # Балансируем 
        results = await TicketPipeline.process_all(tickets, managers, offices)
        
        # Сохраняем в PostgreSQL
        RoutingRepository.save_routing_results(db, results, tickets)
        
        return {"status": "success", "routed_tickets": len(results)}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка маршрутизации: {str(e)}")

# ==========================================
# ЭНДПОИНТЫ: STAR TASK (AI-ASSISTANT)
# ==========================================
@app.post("/api/v1/ai-assistant/chart")
async def ai_chart_endpoint(
    request: ChartRequest, 
    db: Session = Depends(get_db)
):
    try:
        # Достаем исторические данные для построения графиков
        db_records = db.query(RoutingHistory).all()
        
        if not db_records:
            return {"error": "База данных пуста. Сначала распределите тикеты.", "chart_type": "none"}

        appeals_data = []
        for record in db_records:
            appeals_data.append({
                "appeal_type": getattr(record, "ai_ticket_type", "Неизвестно"),
                "sentiment": getattr(record, "ai_sentiment", "Нейтральный"),
                "priority": getattr(record, "ai_complexity_score", 5),
                "language": "RU", # В RoutingHistory нет языка, ставим по умолчанию
                "geo": {"nearest_office": {"name": getattr(record, "assigned_office", "Не определён")}}
            })
            
        chart_json = await generate_chart_data(request.query, appeals_data)
        return chart_json
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка генерации графика: {str(e)}")


# ==========================================
# ЭНДПОИНТЫ: АНАЛИЗ СКРИНШОТОВ (GEMINI)
# ==========================================
@app.post("/api/v1/bugs/analyze", response_model=BugAnalysisResponse)
async def analyze_bug_screenshot(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Файл должен быть изображением")

    report_id = str(uuid.uuid4())
    timestamp = datetime.now()
    
    # --- 1. Сохранение файла (всегда) ---
    file_ext = os.path.splitext(file.filename)[1]
    filename = f"{report_id}{file_ext}"
    file_path = os.path.join(UPLOAD_DIR, filename)
    
    contents = await file.read()
    with open(file_path, "wb") as f:
        f.write(contents)
    
    file_info = {
        "filename": filename,
        "path": file_path,
        "size": len(contents)
    }

    # --- 2. Попытка анализа через Gemini ---
    try:
        image = PIL.Image.open(io.BytesIO(contents))
        
        prompt = """
        Проанализируй скриншот ошибки. Выдай ответ строго в формате:
        Локация: [где ошибка]
        Суть: [что случилось]
        Критичность: [Low/Medium/High]
        """
        
        response = model.generate_content([prompt, image])
        analysis_text = response.text

        return BugAnalysisResponse(
            report_id=report_id,
            status="success",
            timestamp=timestamp,
            file_info=file_info,
            raw_text=analysis_text,
            analysis={
                "summary": analysis_text.split('\n')[0] if '\n' in analysis_text else analysis_text,
                "full_report": analysis_text
            }
        )

    except Exception as e:
        # Если Gemini упал, файл всё равно остается сохраненным
        return JSONResponse(
            status_code=500,
            content=BugAnalysisResponse(
                report_id=report_id,
                status="partial_error",
                timestamp=timestamp,
                file_info=file_info,
                error_details=f"Файл сохранен, но Gemini выдал ошибку: {str(e)}"
            ).dict()
        )


# ==========================================
# ЭНДПОИНТЫ: ИСТОРИЯ И ТЕЛЕГРАМ-БОТ
# ==========================================
# Фрагмент для обновления в main.py (в обоих GET-эндпоинтах):

@app.get("/api/v1/routing-history/{ticket_guid}", response_model=RoutedTicket)
async def get_routing_result(ticket_guid: str, db: Session = Depends(get_db)):
    history_record = db.query(RoutingHistory).filter(RoutingHistory.ticket_guid == ticket_guid).first()
    if not history_record:
        raise HTTPException(status_code=404, detail="Не найдено")

    ai_data = AIAnalysisResult(
        ticket_type=history_record.ai_ticket_type,
        sentiment=history_record.ai_sentiment,
        complexity_score=history_record.ai_complexity_score,
        is_critical=True if history_record.ai_complexity_score >= 50 else False
    )

    return RoutedTicket(
        ticket_guid=history_record.ticket_guid,
        manager_fio=history_record.manager_fio,
        assigned_office=history_record.assigned_office,
        ai_analysis=ai_data,
        sla_deadline=history_record.sla_deadline, # <--- ДОБАВИТЬ ЭТУ СТРОКУ
        routing_reason=history_record.routing_reason
    )

# ==========================================
# ЭНДПОИНТЫ: ЗАГРУЗКА CSV ФАЙЛОВ
# ==========================================
@app.post("/api/v1/upload/{doc_type}")
async def upload_csv(doc_type: str, file: UploadFile = File(...)):
    if doc_type not in ["managers", "tickets", "units"]:
        raise HTTPException(status_code=400, detail="Неверный тип документа")
    
    # Сохраняем загруженный файл асинхронно, чтобы не блокировать сервер
    file_path = os.path.join(UPLOAD_DIR, f"latest_{doc_type}.csv")
    
    async with aiofiles.open(file_path, 'wb') as out_file:
        content = await file.read()
        await out_file.write(content)
        
    return {"status": "success", "processed_count": len(content.splitlines()) - 1}

@app.get("/api/v1/routing-history", response_model=List[RoutedTicket])
async def list_routing_history(skip: int = 0, limit: int = 10, db: Session = Depends(get_db)):
    records = db.query(RoutingHistory).order_by(RoutingHistory.created_at.desc()).offset(skip).limit(limit).all()
    
    result_list = []
    for record in records:
        ai_data = AIAnalysisResult(
            ticket_type=record.ai_ticket_type,
            sentiment=record.ai_sentiment,
            complexity_score=record.ai_complexity_score,
            is_critical=True if record.ai_complexity_score >= 50 else False
        )
        
        result_list.append(RoutedTicket(
            ticket_guid=record.ticket_guid,
            manager_fio=record.manager_fio,
            assigned_office=record.assigned_office,
            ai_analysis=ai_data,
            sla_deadline=record.sla_deadline, # <--- ДОБАВИТЬ ЭТУ СТРОКУ
            routing_reason=record.routing_reason
        ))
        
    return result_list


@app.post("/api/v1/tickets", response_model=RoutedTicket)
async def create_and_route_ticket(ticket: TicketCreate, db: Session = Depends(get_db)):
    """
    Принимает новый тикет (от Telegram бота или сайта), 
    анализирует, назначает менеджера и сохраняет в БД.
    """
    is_complaint = "жалоба" in ticket.description.lower()
    
    ai_data = AIAnalysisResult(
        ticket_type="Жалоба" if is_complaint else "Консультация",
        sentiment="Негативная" if is_complaint else "Нейтральная",
        complexity_score=50 if is_complaint else 10,
        is_critical=True if (ticket.segment == "VIP" and is_complaint) else False
    )

    assigned_manager = "Иванов Иван (VIP-отдел)" if ai_data.is_critical else "Петров Петр"
    assigned_office = "Офис Астана" if ticket.city.lower() == "астана" else "Офис Алматы"
    routing_reason = f"Назначен по сегменту {ticket.segment} и сложности {ai_data.complexity_score}"

    new_record = RoutingHistory(
        ticket_guid=ticket.guid,
        city=ticket.city,
        segment=ticket.segment,
        ai_ticket_type=ai_data.ticket_type,
        ai_sentiment=ai_data.sentiment,
        ai_complexity_score=ai_data.complexity_score,
        manager_fio=assigned_manager,
        assigned_office=assigned_office,
        routing_reason=routing_reason
    )
    
    db.add(new_record)
    db.commit()
    db.refresh(new_record) 

    return RoutedTicket(
        ticket_guid=new_record.ticket_guid,
        manager_fio=new_record.manager_fio,
        assigned_office=new_record.assigned_office,
        ai_analysis=ai_data,
        routing_reason=new_record.routing_reason
    )

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)