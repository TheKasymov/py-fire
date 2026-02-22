from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from sqlalchemy.orm import Session
from typing import List
import uvicorn

from services.parser import CSVParser
from services.pipeline import TicketPipeline
from schemas.models import RoutedTicket
from db.database import engine, Base, get_db
from db.repository import RoutingRepository
import os
import io
import google.generativeai as genai
from fastapi import FastAPI, UploadFile, File, HTTPException
import PIL.Image
from pydantic import BaseModel
from typing import Optional
from datetime import datetime
from typing import List, Dict
from db.models import RoutingHistory
from schemas.models import TicketCreate, RoutedTicket, AIAnalysisResult


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

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

# Магия хакатона: автоматическое создание всех таблиц в PostgreSQL при запуске!
Base.metadata.create_all(bind=engine)

MODEL_NAME = 'gemini-3-flash-preview' 
model = genai.GenerativeModel(MODEL_NAME)
app = FastAPI(title="F.I.R.E. API")

@app.post("/api/v1/route-tickets", response_model=List[RoutedTicket])
async def route_tickets_endpoint(
    tickets_file: UploadFile = File(...),
    managers_file: UploadFile = File(...),
    units_file: UploadFile = File(...),
    db: Session = Depends(get_db)  # Получаем сессию БД
):
    try:
        # 1. Парсим
        offices = CSVParser.parse_business_units(units_file.file)
        managers = CSVParser.parse_managers(managers_file.file)
        tickets = CSVParser.parse_tickets(tickets_file.file)
        
        # 2. Балансируем
        results = await TicketPipeline.process_all(tickets, managers, offices)
        
        # 3. Сохраняем в PostgreSQL
        RoutingRepository.save_routing_results(db, results, tickets)
        
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/api/v1/bugs/analyze")
async def analyze_bug_screenshot(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Файл должен быть изображением")

    # --- ШАГ 1: СОХРАНЯЕМ ФАЙЛ В ЛЮБОМ СЛУЧАЕ ---
    file_extension = os.path.splitext(file.filename)[1]
    unique_filename = f"{uuid.uuid4()}{file_extension}"
    file_path = os.path.join(UPLOAD_DIR, unique_filename)

    try:
        contents = await file.read()
        with open(file_path, "wb") as f:
            f.write(contents)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Не удалось сохранить файл: {str(e)}")

    # --- ШАГ 2: ОБРАБОТКА GEMINI ---
    try:
        # Открываем изображение для нейросети
        image = PIL.Image.open(io.BytesIO(contents))
        
        prompt = """
        Ты опытный QA-инженер. Проанализируй скриншот ошибки.
        Сформируй отчет для менеджера:
        **Где обнаружено:** ...
        **Суть проблемы:** ...
        **Влияние на пользователя:** ...
        """

        response = model.generate_content([prompt, image])
        
        return {
            "status": "success",
            "saved_file": file_path,
            "pm_description": response.text
        }

    except Exception as e:
        # Если Gemini не сработал, мы всё равно возвращаем 200 или 500, 
        # но сообщаем, что файл-то мы сохранили!
        return JSONResponse(
            status_code=500,
            content={
                "status": "partial_error",
                "message": f"Файл сохранен как {unique_filename}, но Gemini выдал ошибку.",
                "details": str(e),
                "saved_file_path": file_path
            }
        )

@app.post("/api/v1/bugs/analyze", response_model=BugAnalysisResponse)
async def analyze_bug_screenshot(file: UploadFile = File(...)):
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
        
        # Просим Gemini вернуть ответ в формате JSON-подобной структуры
        prompt = """
        Проанализируй скриншот. Выдай ответ строго в формате:
        Локация: [где ошибка]
        Суть: [что случилось]
        Критичность: [Low/Medium/High]
        """
        
        response = model.generate_content([prompt, image])
        analysis_text = response.text

        # Пример простой логики парсинга (можно усложнить)
        return BugAnalysisResponse(
            report_id=report_id,
            status="success",
            timestamp=timestamp,
            file_info=file_info,
            raw_text=analysis_text,
            analysis={
                "summary": analysis_text.split('\n')[0], # Пример грубого парсинга
                "full_report": analysis_text
            }
        )

    except Exception as e:
        # Если Gemini упал, файл всё равно остается сохраненным
        return BugAnalysisResponse(
            report_id=report_id,
            status="partial_error",
            timestamp=timestamp,
            file_info=file_info,
            error_details=str(e)
        )

@app.get("/api/v1/routing-history/{ticket_guid}", response_model=RoutedTicket)
async def get_routing_result(ticket_guid: str, db: Session = Depends(get_db)):
    """
    Получить итоговый результат маршрутизации конкретного тикета
    """
    # 1. Ищем запись в вашей БД (RoutingHistory)
    history_record = db.query(RoutingHistory).filter(RoutingHistory.ticket_guid == ticket_guid).first()
    
    if not history_record:
        raise HTTPException(status_code=404, detail="История маршрутизации для данного тикета не найдена")

    # 2. Собираем вложенный объект ИИ аналитики из плоских колонок БД
    ai_data = AIAnalysisResult(
        ticket_type=history_record.ai_ticket_type,
        sentiment=history_record.ai_sentiment,
        complexity_score=history_record.ai_complexity_score,
        # Примечание: is_critical нет в вашей БД, поэтому вычисляем на лету
        # или ставим False по умолчанию
        is_critical=True if history_record.ai_complexity_score >= 50 else False
    )

    # 3. Возвращаем итоговую схему RoutedTicket
    return RoutedTicket(
        ticket_guid=history_record.ticket_guid,
        manager_fio=history_record.manager_fio,
        assigned_office=history_record.assigned_office,
        ai_analysis=ai_data,
        routing_reason=history_record.routing_reason
    )

@app.get("/api/v1/routing-history", response_model=List[RoutedTicket])
async def list_routing_history(skip: int = 0, limit: int = 10, db: Session = Depends(get_db)):
    """
    Получить список последних маршрутизаций
    """
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
            routing_reason=record.routing_reason
        ))
        
    return result_list

@app.post("/api/v1/tickets", response_model=RoutedTicket)
async def create_and_route_ticket(ticket: TicketCreate, db: Session = Depends(get_db)):
    """
    Принимает новый тикет (от Telegram бота или сайта), 
    анализирует через ИИ, назначает менеджера и сохраняет в БД.
    """
    
    # ---------------------------------------------------------
    # ШАГ 1: АНАЛИЗ ИИ (AI Module)
    # ---------------------------------------------------------
    # Здесь вызовите вашу реальную нейросеть (Gemini, Llama или вашу appeals_nb_model)
    # Для примера я напишу простую логику-заглушку:
    
    is_complaint = "жалоба" in ticket.description.lower()
    
    ai_data = AIAnalysisResult(
        ticket_type="Жалоба" if is_complaint else "Консультация",
        sentiment="Негативная" if is_complaint else "Нейтральная",
        complexity_score=50 if is_complaint else 10,
        is_critical=True if (ticket.segment == "VIP" and is_complaint) else False
    )

    # ---------------------------------------------------------
    # ШАГ 2: МАРШРУТИЗАЦИЯ (Geo Balancer)
    # ---------------------------------------------------------
    # Здесь логика подбора менеджера (проверка скиллов, города, загрузки)
    
    assigned_manager = "Иванов Иван (VIP-отдел)" if ai_data.is_critical else "Петров Петр"
    assigned_office = "Офис Астана" if ticket.city.lower() == "астана" else "Офис Алматы"
    routing_reason = f"Назначен по сегменту {ticket.segment} и сложности {ai_data.complexity_score}"

    # ---------------------------------------------------------
    # ШАГ 3: СОХРАНЕНИЕ В БАЗУ ДАННЫХ (PostgreSQL)
    # ---------------------------------------------------------
    new_record = RoutingHistory(
        ticket_guid=ticket.guid,
        city=ticket.city,
        segment=ticket.segment,
        
        # Данные от ИИ
        ai_ticket_type=ai_data.ticket_type,
        ai_sentiment=ai_data.sentiment,
        ai_complexity_score=ai_data.complexity_score,
        
        # Данные маршрутизации
        manager_fio=assigned_manager,
        assigned_office=assigned_office,
        routing_reason=routing_reason
    )
    
    db.add(new_record)
    db.commit()
    db.refresh(new_record) # Получаем обновленные данные (с ID и датой)

    # ---------------------------------------------------------
    # ШАГ 4: ОТВЕТ ДЛЯ TELEGRAM БОТА
    # ---------------------------------------------------------
    # Возвращаем данные строго по схеме RoutedTicket
    return RoutedTicket(
        ticket_guid=new_record.ticket_guid,
        manager_fio=new_record.manager_fio,
        assigned_office=new_record.assigned_office,
        ai_analysis=ai_data,
        routing_reason=new_record.routing_reason
    )