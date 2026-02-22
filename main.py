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