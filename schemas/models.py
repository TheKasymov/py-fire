from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import date

# ==========================================
# Схемы для сущностей из CSV (Входящие данные)
# ==========================================

class BusinessUnit(BaseModel):
    name: str = Field(..., description="Название офиса (например, 'Офис Астана')")
    address: str = Field(..., description="Физический адрес")
    city: str = Field(..., description="Город (Алматы или Астана)")

class Manager(BaseModel):
    fio: str = Field(..., description="ФИО сотрудника")
    position: str = Field(..., description="Должность: Спец, Ведущий спец, Глав спец")
    skills: List[str] = Field(default_factory=list, description="Список навыков: ['VIP', 'ENG', 'KZ']")
    office_name: str = Field(..., description="Привязка к офису")
    
    # Для логики "Антивыгорания"
    current_ticket_count: int = Field(default=0, description="Количество обращений в работе (из CSV)")
    current_score: int = Field(default=0, description="Сумма баллов сложности (накапливается в рантайме)")

class TicketCreate(BaseModel):
    guid: str = Field(..., description="Уникальный ID клиента")
    gender: Optional[str] = None
    birth_date: Optional[str] = None
    segment: str = Field(..., description="Сегмент: Mass, VIP, Priority")
    description: str = Field(..., description="Текст обращения")
    city: str = Field(..., description="Населенный пункт (для маршрутизации)")
    
    # Опциональные поля адреса (могут быть пустыми)
    country: Optional[str] = None
    region: Optional[str] = None
    street: Optional[str] = None
    house: Optional[str] = None

# ==========================================
# Схемы для результатов работы ИИ (AI Module)
# ==========================================

class AIAnalysisResult(BaseModel):
    ticket_type: str = Field(..., description="Категория: Жалоба, Смена данных, Тех запрос и т.д.")
    sentiment: str = Field(..., description="Тональность: негативная, нейтральная и т.д.")
    complexity_score: int = Field(..., description="Баллы сложности: 5 (легко), 10 (жалоба), 50 (мошенничество)")
    is_critical: bool = Field(default=False, description="True, если VIP+ и кризис (отправка старшему)")

# ==========================================
# Схемы для итогового результата (Выходящие данные)
# ==========================================

class RoutedTicket(BaseModel):
    ticket_guid: str
    manager_fio: str
    assigned_office: str
    ai_analysis: AIAnalysisResult
    routing_reason: str = Field(..., description="Почему назначен именно этот менеджер (например, 'Round Robin, Score: 15')")