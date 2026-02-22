from sqlalchemy import Column, String, Integer, DateTime, Boolean, Text
from datetime import datetime
from db.database import Base

class RoutingHistory(Base):
    __tablename__ = "routing_history"

    id = Column(Integer, primary_key=True, index=True)
    ticket_guid = Column(String, index=True)
    manager_fio = Column(String)
    assigned_office = Column(String)
    
    # Данные AI
    ai_ticket_type = Column(String)
    ai_sentiment = Column(String)
    ai_complexity_score = Column(Integer)
    
    # Новые поля по ТЗ
    sla_deadline = Column(String)  # <--- НОВОЕ ПОЛЕ ДЛЯ SLA
    routing_reason = Column(Text)
    
    created_at = Column(DateTime, default=datetime.utcnow)