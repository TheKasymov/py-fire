import math
from typing import List, Dict, Any, Optional
from services.geo_balancer import geo_balancer 
from ai_service.nlp_module import analyze_text, generate_psychological_portrait

class TicketPipeline:
    @staticmethod
    async def process_all(tickets: List[Dict], managers: List[Dict], offices: List[Dict]) -> List[Dict]:
        results = []
        rr_index = {}
        
        # Переменная-счетчик для распределения неизвестных адресов 50/50
        fallback_city_toggle = 0 

        for ticket in tickets:
            # 1. AI Анализ (Тип, Тональность, Приоритет)
            ai_analysis = await analyze_text(ticket['text'])
            
            # 2. Психологический портрет (твоя новая фича)
            portrait = generate_psychological_portrait(
                ticket['text'], 
                ai_analysis['sentiment'], 
                ai_analysis['priority']
            )
            
            # 3. Гео-балансировка
            geo_data = await geo_balancer.geocode(ticket['address'])
            
            # --- Логика 50/50 для неизвестных адресов ---
            if geo_data:
                current_city = geo_data['city']
            else:
                current_city = "Астана" if fallback_city_toggle % 2 == 0 else "Алматы"
                fallback_city_toggle += 1
            
            # 4. Фильтрация менеджеров по хард-скиллам (VIP, Смена данных, Язык)
            suitable_managers = TicketPipeline._filter_managers(
                managers, 
                current_city, 
                ai_analysis, 
                ticket.get('segment')
            )
            
            # 5. Распределение Round Robin (балансировка нагрузки)
            assigned_manager = TicketPipeline._apply_round_robin(
                suitable_managers, 
                current_city, 
                rr_index
            )
            
            # 6. Формирование финального объекта (для БД и фронтенда)
            # Совмещаем бизнес-данные и твою новую аналитику
            routed_data = TicketPipeline._build_routed_ticket(
                ticket, ai_analysis, assigned_manager, current_city
            )
            
            # Добавляем дополнительные поля, которые нужны фронтенду
            results.append({
                **routed_data,
                "psychological_portrait": portrait,
                "geo": geo_data,
                "raw_text": ticket['text']
            })
            
        return results

    @staticmethod
    def _filter_managers(managers, city, ai_data, segment):
        suitable = []
        appeal_type = ai_data.get('appeal_type', '')
        
        for m in managers:
            # 1. Гео-фильтр
            if m['office'].lower() != city.lower():
                continue
            
            # 2. Фильтр VIP/Priority (ТЗ: только для VIP-скилла)
            if segment in ['VIP', 'Priority'] and 'VIP' not in m['skills']:
                continue
                
            # 3. Фильтр "Смена данных" (ТЗ: только для Главных спецов)
            if appeal_type == 'Смена данных' and 'глав' not in m['role'].lower():
                continue
                
            # 4. Языковой фильтр
            ticket_lang = ai_data.get('language', 'RU')
            if ticket_lang in ['KZ', 'ENG'] and ticket_lang not in m['skills']:
                continue
                
            suitable.append(m)
        return suitable

    @staticmethod
    def _apply_round_robin(suitable_managers, city, rr_index):
        if not suitable_managers:
            return None
        # Сортируем по нагрузке (у кого меньше обращений в работе)
        suitable_managers.sort(key=lambda x: int(x.get('load', 0)))
        candidates = suitable_managers[:2]
        idx = rr_index.get(city, 0) % len(candidates)
        rr_index[city] = idx + 1
        return candidates[idx]

    @staticmethod
    def _map_complexity_score(appeal_type: str, priority: int) -> int:
        """Расчет баллов сложности по ТЗ"""
        if appeal_type == "Мошеннические действия": return 50
        if appeal_type in {"Претензия", "Жалоба", "Неработоспособность приложения"}: return 10
        return 5 if priority <= 5 else 10

    @staticmethod
    def _build_routed_ticket(ticket, ai_analysis, assigned_manager, current_city):
        """Вспомогательный метод для сборки структуры данных"""
        appeal_type = ai_analysis.get("appeal_type", "Консультация")
        priority = int(ai_analysis.get("priority", 1))
        
        return {
            "ticket_id": ticket.get("id"),
            "assigned_manager": assigned_manager['name'] if assigned_manager else "Очередь (нет подходящих)",
            "assigned_office": assigned_manager['office'] if assigned_manager else current_city,
            "analysis": {
                "appeal_type": appeal_type,
                "sentiment": ai_analysis.get("sentiment", "Нейтральный"),
                "priority": priority,
                "complexity_score": TicketPipeline._map_complexity_score(appeal_type, priority),
                "language": ai_analysis.get("language", "RU")
            }
        }