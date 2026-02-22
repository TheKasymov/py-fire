from typing import List, Dict, Any, Optional
from services.geo_balancer import geo_balancer
from ai_service.nlp_module import analyze_text

class TicketPipeline:
    @staticmethod
    async def process_all(tickets: List[Dict], managers: List[Dict], offices: List[Dict]) -> List[Dict]:
        results = []
        rr_index = {}

        # Переменная-счетчик для распределения неизвестных адресов 50/50
        fallback_city_toggle = 0

        for ticket in tickets:
            ai_analysis = await analyze_text(ticket['text'])
            geo_data = await geo_balancer.geocode(ticket['address'])

            # --- ИСПРАВЛЕНИЕ: Гео-фильтр 50/50 ---
            if geo_data:
                current_city = geo_data['city']
            else:
                # Если адрес неизвестен/зарубежный, чередуем Астану и Алматы
                current_city = "Астана" if fallback_city_toggle % 2 == 0 else "Алматы"
                fallback_city_toggle += 1

            suitable_managers = TicketPipeline._filter_managers(
                managers,
                current_city,
                ai_analysis,
                ticket.get('segment')
            )

            assigned_manager = TicketPipeline._apply_round_robin(
                suitable_managers,
                current_city,
                rr_index
            )

            results.append(
                TicketPipeline._build_routed_ticket(
                    ticket=ticket,
                    ai_analysis=ai_analysis,
                    assigned_manager=assigned_manager,
                    current_city=current_city,
                )
            )

        return results

    @staticmethod
    def _to_int(value: Any, default: int = 0) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _map_complexity_score(appeal_type: str, priority: int) -> int:
        if appeal_type == "Мошеннические действия":
            return 50
        if appeal_type in {
            "Претензия",
            "Жалоба",
            "Неработоспособность приложения",
        }:
            return 10
        # Легкие категории: Консультация, Смена данных, Спам
        return 5 if priority <= 5 else 10

    @staticmethod
    def _build_routed_ticket(
        ticket: Dict[str, Any],
        ai_analysis: Dict[str, Any],
        assigned_manager: Optional[Dict[str, Any]],
        current_city: str,
    ) -> Dict[str, Any]:
        ticket_guid = str(ticket.get("guid") or ticket.get("id") or "")
        appeal_type = ai_analysis.get("appeal_type") or "Консультация"
        sentiment = ai_analysis.get("sentiment") or "Нейтральный"
        priority = TicketPipeline._to_int(ai_analysis.get("priority"), default=1)
        complexity_score = TicketPipeline._map_complexity_score(appeal_type, priority)

        segment = (ticket.get("segment") or "Mass").strip()
        is_critical = segment in {"VIP", "Priority"} and priority >= 7

        if assigned_manager:
            manager_fio = assigned_manager.get("name") or "Очередь (нет подходящих)"
            assigned_office = assigned_manager.get("office") or current_city
            routing_reason = (
                f"Round Robin по офису '{current_city}', "
                f"сегмент '{segment}', тип '{appeal_type}'"
            )
        else:
            manager_fio = "Очередь (нет подходящих)"
            assigned_office = current_city
            routing_reason = (
                f"Нет подходящего менеджера для офиса '{current_city}' "
                "(фильтры: офис/сегмент/язык/роль)"
            )

        return {
            "ticket_guid": ticket_guid,
            "manager_fio": manager_fio,
            "assigned_office": assigned_office,
            "ai_analysis": {
                "ticket_type": appeal_type,
                "sentiment": sentiment,
                "complexity_score": complexity_score,
                "is_critical": is_critical,
            },
            "routing_reason": routing_reason,
        }

    @staticmethod
    def _filter_managers(managers, city, ai_data, segment):
        suitable = []
        appeal_type = ai_data.get('appeal_type', '')

        for m in managers:
            # 1. Гео-фильтр (Офис в том же городе)
            if m['office'].lower() != city.lower():
                continue

            # --- ИСПРАВЛЕНИЕ: Фильтр VIP/Priority ---
            if segment in ['VIP', 'Priority'] and 'VIP' not in m['skills']:
                continue

            # --- ИСПРАВЛЕНИЕ: Фильтр "Смена данных" -> Глав спец ---
            if appeal_type == 'Смена данных' and 'глав' not in m['role'].lower():
                continue

            # 3. Языковой фильтр
            ticket_lang = ai_data.get('language', 'RU')
            if ticket_lang in ['KZ', 'ENG'] and ticket_lang not in m['skills']:
                continue

            suitable.append(m)
        return suitable

    @staticmethod
    def _apply_round_robin(suitable_managers, city, rr_index):
        if not suitable_managers:
            return None

        # Сортируем по нагрузке
        suitable_managers.sort(
            key=lambda manager: TicketPipeline._to_int(manager.get("load"), default=0)
        )

        # Применяем Round Robin к топ-2 самым свободным
        candidates = suitable_managers[:2]
        idx = rr_index.get(city, 0) % len(candidates)
        rr_index[city] = idx + 1

        return candidates[idx]
