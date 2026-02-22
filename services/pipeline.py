from typing import List, Dict, Any, Optional
from services.geo_balancer import geo_balancer
from ai_service.nlp_module import analyze_text

class TicketPipeline:
    @staticmethod
    async def process_all(tickets: List[Dict], managers: List[Dict], offices: List[Dict]) -> List[Dict]:
        results = []
        
        # Инициализация баллов менеджеров (конвертируем текущую нагрузку в стартовые баллы)
        for m in managers:
            if 'current_score' not in m:
                m['current_score'] = int(m.get('load', 0)) * 5

        fallback_city_toggle = 0

        for ticket in tickets:
            ai_analysis = await analyze_text(ticket['text'])
            geo_data = await geo_balancer.geocode(ticket['address'])

            if geo_data:
                current_city = geo_data['city']
            else:
                current_city = "Астана" if fallback_city_toggle % 2 == 0 else "Алматы"
                fallback_city_toggle += 1

            segment = (ticket.get('segment') or "Mass").strip()
            
            # 1. Расчет параметров по ТЗ
            # 1. Расчет параметров по ТЗ
            sla_time, is_critical, complexity_score, risk_level = TicketPipeline._determine_sla_and_escalation(ai_analysis, segment)
            
            # 2. Маршрутизация
            assigned_manager = None
            routing_reason = ""

            # ИИ забирает VIP+ ИЛИ "Самые легкие запросы" (Без риска, 5 баллов)
            if segment == "VIP+" or (complexity_score == 5 and risk_level == "Без риска"):
                assigned_manager = {"name": "AI Агент", "office": "Виртуальный офис"}
                routing_reason = f"Мгновенный ответ ИИ (Сегмент: {segment}, Риск: {risk_level})."
            
            elif is_critical:
                suitable_managers = [m for m in managers if 'старш' in m.get('role', '').lower() or 'глав' in m.get('role', '').lower()]
                assigned_manager = TicketPipeline._apply_anti_burnout_routing(suitable_managers, complexity_score)
                routing_reason = f"Эскалация: сработали 4 условия. Уровень риска: {risk_level}."
            
            else:
                suitable_managers = TicketPipeline._filter_managers(managers, current_city, ai_analysis, segment)
                assigned_manager = TicketPipeline._apply_anti_burnout_routing(suitable_managers, complexity_score)
                routing_reason = f"Стандартное распределение. Риск: {risk_level}, Сложность: {complexity_score} баллов."
            # Обновляем нагрузку менеджера
            if assigned_manager and assigned_manager.get("name") != "AI Агент":
                assigned_manager["current_score"] += complexity_score

            # Формируем результат
            ticket_guid = str(ticket.get("guid") or ticket.get("id") or "")
            appeal_type = ai_analysis.get("appeal_type", "Консультация")
            sentiment = ai_analysis.get("sentiment", "Нейтральный")

            results.append({
                "ticket_guid": ticket_guid,
                "manager_fio": assigned_manager.get("name", "Очередь (нет свободных)") if assigned_manager else "Очередь (нет свободных)",
                "assigned_office": assigned_manager.get("office", current_city) if assigned_manager else current_city,
                "ai_analysis": {
                    "ticket_type": appeal_type,
                    "sentiment": sentiment,
                    "complexity_score": complexity_score,
                    "is_critical": is_critical,
                },
                "sla_deadline": sla_time,
                "routing_reason": routing_reason
            })

        return results

    @staticmethod
    def _determine_sla_and_escalation(ai_analysis: Dict, segment: str):
        appeal_type = ai_analysis.get("appeal_type", "Консультация")
        sentiment = ai_analysis.get("sentiment", "Нейтральный")
        priority = int(ai_analysis.get("priority", 1))

        # --- A. Расчет баллов (Срочность/Сложность) ---
        if appeal_type == "Мошеннические действия":
            complexity_score = 50
        elif appeal_type in ["Жалоба", "Претензия", "Неработоспособность приложения"]:
            complexity_score = 10
        else:
            complexity_score = 5  # Тех запрос, Консультация, Смена данных

        # --- B. Уровни Риска (ПО ТЗ: без риска, низкий, средний, высокий) ---
        if complexity_score == 50:
            risk_level = "Высокий"
        elif complexity_score == 10 and sentiment == "Негативный":
            risk_level = "Средний"
        elif complexity_score == 10:
            risk_level = "Низкий"
        else:
            risk_level = "Без риска"

        # --- C. SLA (Время ответа) ---
        if segment == "VIP+":
            sla_time = "Мгновенно (ИИ отвечает)"
        elif segment == "VIP":
            sla_time = "В течение 15 минут"
        elif segment == "Priority":
            sla_time = "В течение 3 часов"
        else:
            # Если запрос "Без риска" и легкий - пусть ИИ тоже забирает на себя
            if complexity_score == 5 and risk_level == "Без риска":
                 sla_time = "Мгновенно (Автоответ ИИ)"
            else:
                 sla_time = "В течение 24 часов"

        # --- D. 4 Условия Эскалации ---
        is_vip = segment in ["VIP", "VIP+"]
        is_emotional = sentiment == "Негативный"
        is_crisis_word = priority >= 8 or appeal_type == "Мошеннические действия"
        is_serious_topic = appeal_type in ["Мошеннические действия", "Жалоба", "Претензия", "Неработоспособность приложения"]

        is_critical = bool(is_vip and is_emotional and is_crisis_word and is_serious_topic)

        # Возвращаем еще и уровень риска
        return sla_time, is_critical, complexity_score, risk_level

    @staticmethod
    def _apply_anti_burnout_routing(suitable_managers: List[Dict], complexity_score: int) -> Optional[Dict]:
        if not suitable_managers:
            return None

        valid_managers = []
        for m in suitable_managers:
            # Используем безопасный .get()
            current_score = m.get('current_score', 0)
            
            # АНТИПЕРЕГОРАНИЕ: Если у менеджера накопилось 20+ баллов, он не получает сложные запросы
            if current_score >= 20 and complexity_score >= 10:
                continue
            valid_managers.append(m)

        # Если все перегружены, берем всех, чтобы запрос не завис
        managers_to_sort = valid_managers if valid_managers else suitable_managers

        # Сортируем по сумме баллов (чем меньше баллов, тем приоритетнее)
        managers_to_sort.sort(key=lambda x: x.get('current_score', 0))
        
        return managers_to_sort[0]

    @staticmethod
    def _filter_managers(managers, city, ai_data, segment):
        suitable = []
        appeal_type = ai_data.get('appeal_type', '')
        ticket_lang = ai_data.get('language', 'RU')

        for m in managers:
            if m.get('office', '').lower() != city.lower():
                continue
            if segment in ['VIP', 'Priority'] and 'VIP' not in m.get('skills', []):
                continue
            if appeal_type == 'Смена данных' and 'глав' not in m.get('role', '').lower():
                continue
            if ticket_lang in ['KZ', 'ENG'] and ticket_lang not in m.get('skills', []):
                continue
            
            suitable.append(m)
            
        return suitable