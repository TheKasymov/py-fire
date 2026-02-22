import math
from typing import List, Dict, Any
# Важно: имя файла должно быть services/geo_balancer.py
from services.geo_balancer import geo_balancer 
from ai_service.nlp_module import analyze_text

class TicketPipeline:
    @staticmethod
    async def process_all(tickets: List[Dict], managers: List[Dict], offices: List[Dict]) -> List[Dict]:
        results = []
        
        # Индекс для Round Robin (храним в памяти на время обработки пачки)
        rr_index = {}

        for ticket in tickets:
            # 1. AI Анализ через локальную Ollama или Naive Bayes
            ai_analysis = await analyze_text(ticket['text'])
            
            # 2. Гео-балансировка (используем ваш класс GeoFallbackBalancer)
            # Передаем адрес из распарсенного тикета
            geo_data = await geo_balancer.geocode(ticket['address'])
            
            # Если адрес не найден, берем дефолтный город или Астану
            current_city = geo_data['city'] if geo_data else "Астана"
            
            # 3. Фильтрация менеджеров по правилам ТЗ
            suitable_managers = TicketPipeline._filter_managers(
                managers, 
                current_city, 
                ai_analysis, 
                ticket.get('segment')
            )
            
            # 4. Распределение (Round Robin)
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
        for m in managers:
            # 1. Гео-фильтр (Офис в том же городе)
            if m['office'].lower() != city.lower():
                continue
            
            # 2. VIP фильтр (Только для менеджеров с навыком VIP)
            if segment == 'VIP' and 'VIP' not in m['skills']:
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