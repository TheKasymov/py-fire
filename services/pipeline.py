import math
from typing import List, Dict, Any
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
            
            results.append({
                "ticket_id": ticket['id'],
                "analysis": ai_analysis,
                "geo": geo_data,
                "assigned_manager": assigned_manager['name'] if assigned_manager else "Очередь (нет подходящих)"
            })
            
        return results

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
        suitable_managers.sort(key=lambda x: x['load'])
        
        # Применяем Round Robin к топ-2 самым свободным
        candidates = suitable_managers[:2]
        idx = rr_index.get(city, 0) % len(candidates)
        rr_index[city] = idx + 1
        
        return candidates[idx]