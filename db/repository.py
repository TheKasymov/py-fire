from typing import Any

from sqlalchemy.orm import Session
from db.models import RoutingHistory

class RoutingRepository:
    @staticmethod
    def _get_value(item: Any, *fields: str, default: Any = None) -> Any:
        if item is None:
            return default

        if isinstance(item, dict):
            for field in fields:
                value = item.get(field)
                if value is not None:
                    return value
            return default

        for field in fields:
            if hasattr(item, field):
                value = getattr(item, field)
                if value is not None:
                    return value
        return default

    @staticmethod
    def _to_int(value: Any, default: int = 0) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def save_routing_results(db: Session, routed_tickets: list[Any], original_tickets: list[Any]):
        # Создаем словарь оригинальных тикетов для быстрого поиска города и сегмента
        ticket_map: dict[str, Any] = {}
        for ticket in original_tickets:
            guid = RoutingRepository._get_value(ticket, "guid", "id")
            if guid:
                ticket_map[str(guid)] = ticket

        db_records = []
        for routed_ticket in routed_tickets:
            ticket_guid = str(
                RoutingRepository._get_value(
                    routed_ticket, "ticket_guid", "ticket_id", "guid", "id", default=""
                )
            )
            if not ticket_guid:
                continue

            orig_ticket = ticket_map.get(ticket_guid)
            ai_analysis = RoutingRepository._get_value(
                routed_ticket, "ai_analysis", "analysis", default={}
            )

            city = RoutingRepository._get_value(orig_ticket, "city", default="Неизвестно")
            segment = RoutingRepository._get_value(orig_ticket, "segment", default="Mass")

            record = RoutingHistory(
                ticket_guid=ticket_guid,
                city=city,
                segment=segment,

                ai_ticket_type=RoutingRepository._get_value(
                    ai_analysis, "ticket_type", "appeal_type", default="Консультация"
                ),
                ai_sentiment=RoutingRepository._get_value(
                    ai_analysis, "sentiment", default="Нейтральный"
                ),
                ai_complexity_score=RoutingRepository._to_int(
                    RoutingRepository._get_value(
                        ai_analysis, "complexity_score", "priority", default=0
                    ),
                    default=0,
                ),

                manager_fio=RoutingRepository._get_value(
                    routed_ticket,
                    "manager_fio",
                    "assigned_manager",
                    default="Очередь (нет подходящих)",
                ),
                assigned_office=RoutingRepository._get_value(
                    routed_ticket, "assigned_office", default=city or "Не определён"
                ),
                routing_reason=RoutingRepository._get_value(
                    routed_ticket, "routing_reason", default="Автоматическая маршрутизация"
                ),
            )
            db_records.append(record)

        if not db_records:
            return

        # Массовая запись (Bulk insert) работает очень быстро!
        db.add_all(db_records)
        db.commit()
