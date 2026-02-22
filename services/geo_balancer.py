import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class GeoFallbackBalancer:
    def __init__(self):
        # Локальная база городов из твоего business_units.csv
        self.local_cache = {
            "актау": {"lat": 43.6481, "lon": 51.1706, "address": "г. Актау", "city": "Актау"},
            "актобе": {"lat": 50.2839, "lon": 57.1670, "address": "г. Актобе", "city": "Актобе"},
            "алматы": {"lat": 43.2220, "lon": 76.8512, "address": "г. Алматы", "city": "Алматы"},
            "астана": {"lat": 51.1801, "lon": 71.4460, "address": "г. Астана", "city": "Астана"},
            "атырау": {"lat": 47.0945, "lon": 51.9238, "address": "г. Атырау", "city": "Атырау"},
            "караганда": {"lat": 49.8019, "lon": 73.1021, "address": "г. Караганда", "city": "Караганда"},
            "кокшетау": {"lat": 53.2846, "lon": 69.3775, "address": "г. Кокшетау", "city": "Кокшетау"},
            "костанай": {"lat": 53.2198, "lon": 63.6283, "address": "г. Костанай", "city": "Костанай"},
            "кызылорда": {"lat": 44.8486, "lon": 65.4823, "address": "г. Кызылорда", "city": "Кызылорда"},
            "павлодар": {"lat": 52.3013, "lon": 76.9566, "address": "г. Павлодар", "city": "Павлодар"},
            "петропавловск": {"lat": 54.8732, "lon": 69.1430, "address": "г. Петропавловск", "city": "Петропавловск"},
            "тараз": {"lat": 42.9000, "lon": 71.3667, "address": "г. Тараз", "city": "Тараз"},
            "уральск": {"lat": 51.2333, "lon": 51.3667, "address": "г. Уральск", "city": "Уральск"},
            "усть-каменогорск": {"lat": 49.9483, "lon": 82.6279, "address": "г. Усть-Каменогорск", "city": "Усть-Каменогорск"},
            "шымкент": {"lat": 42.3417, "lon": 69.5901, "address": "г. Шымкент", "city": "Шымкент"}
        }

    async def geocode(self, address_query: str) -> Optional[Dict[str, Any]]:
        if not address_query:
            return None
            
        address_lower = address_query.lower()

        for city, data in self.local_cache.items():
            if city in address_lower:
                return data

        return None

# ИМЕННО ЭТУ ПЕРЕМЕННУЮ ИЩЕТ ПАЙПЛАЙН:
geo_balancer = GeoFallbackBalancer()