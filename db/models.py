from sqlalchemy import Column, Integer, String, Boolean, DateTime
from datetime import datetime
from db.database import Base

class RoutingHistory(Base):
    __tablename__ = "routing_history"

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # --- Данные Лида (Клиента) ---
    ticket_guid = Column(String, index=True)
    city = Column(String)
    segment = Column(String)
    
    # --- Данные Аналитики ИИ ---
    ai_ticket_type = Column(String)
    ai_sentiment = Column(String)
    ai_complexity_score = Column(Integer)
    
    # --- Данные Назначенного Менеджера ---
    manager_fio = Column(String, index=True)
    assigned_office = Column(String)
    routing_reason = Column(String)