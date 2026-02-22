from pydantic import BaseModel, Field
from typing import List, Optional

class BusinessUnit(BaseModel):
    name: str
    address: str
    city: str

class Manager(BaseModel):
    fio: str
    position: str
    skills: List[str] = Field(default_factory=list)
    office_name: str
    current_ticket_count: int = 0
    current_score: int = Field(default=0, description="Сумма баллов сложности")

class TicketCreate(BaseModel):
    guid: str
    gender: Optional[str] = None
    birth_date: Optional[str] = None
    segment: str
    description: str
    city: str
    country: Optional[str] = None
    region: Optional[str] = None
    street: Optional[str] = None
    house: Optional[str] = None

class AIAnalysisResult(BaseModel):
    ticket_type: str
    sentiment: str
    complexity_score: int
    is_critical: bool

class RoutedTicket(BaseModel):
    ticket_guid: str
    manager_fio: str
    assigned_office: str
    ai_analysis: AIAnalysisResult
    sla_deadline: str = Field(..., description="Регламентированное время ответа (SLA)")
    routing_reason: str