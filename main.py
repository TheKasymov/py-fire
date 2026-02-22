from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from sqlalchemy.orm import Session
from typing import List
import uvicorn

from services.parser import CSVParser
from services.pipeline import TicketPipeline
from schemas.models import RoutedTicket
from db.database import engine, Base, get_db
from db.repository import RoutingRepository

# Магия хакатона: автоматическое создание всех таблиц в PostgreSQL при запуске!
Base.metadata.create_all(bind=engine)

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