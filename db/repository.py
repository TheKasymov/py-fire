from sqlalchemy.orm import Session
from db.models import RoutingHistory
from typing import List, Dict

class RoutingRepository:
    @staticmethod
    def save_routing_results(db: Session, routed_tickets: List[Dict], original_tickets: List[Dict]):
        for rt in routed_tickets:
            ai_data = rt.get("ai_analysis", {})
            
            history_record = RoutingHistory(
                ticket_guid=rt["ticket_guid"],
                manager_fio=rt["manager_fio"],
                assigned_office=rt["assigned_office"],
                ai_ticket_type=ai_data.get("ticket_type"),
                ai_sentiment=ai_data.get("sentiment"),
                ai_complexity_score=ai_data.get("complexity_score"),
                sla_deadline=rt.get("sla_deadline", "Не задан"), # <--- ПЕРЕДАЕМ SLA
                routing_reason=rt["routing_reason"]
            )
            db.add(history_record)
        
        db.commit()