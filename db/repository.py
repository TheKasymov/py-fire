from sqlalchemy.orm import Session
from db.models import RoutingHistory
from schemas.models import RoutedTicket, TicketCreate

class RoutingRepository:
    @staticmethod
    def save_routing_results(db: Session, routed_tickets: list[RoutedTicket], original_tickets: list[TicketCreate]):
        # Создаем словарь оригинальных тикетов для быстрого поиска города и сегмента
        ticket_map = {t.guid: t for t in original_tickets}
        
        db_records = []
        for rt in routed_tickets:
            orig_ticket = ticket_map.get(rt.ticket_guid)
            
            record = RoutingHistory(
                ticket_guid=rt.ticket_guid,
                city=orig_ticket.city if orig_ticket else "Неизвестно",
                segment=orig_ticket.segment if orig_ticket else "Mass",
                
                ai_ticket_type=rt.ai_analysis.ticket_type,
                ai_sentiment=rt.ai_analysis.sentiment,
                ai_complexity_score=rt.ai_analysis.complexity_score,
                
                manager_fio=rt.manager_fio,
                assigned_office=rt.assigned_office,
                routing_reason=rt.routing_reason
            )
            db_records.append(record)
            
        # Массовая запись (Bulk insert) работает очень быстро!
        db.add_all(db_records)
        db.commit()