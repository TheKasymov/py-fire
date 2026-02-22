import asyncio
import logging
from typing import Optional, Dict, Any
from geopy.geocoders import Nominatim, Photon, ArcGIS
from geopy.exc import GeocoderTimedOut, GeocoderServiceError, GeocoderQuotaExceeded

logger = logging.getLogger(__name__)

class GeoFallbackBalancer:
    def __init__(self):
        # "Скамейка запасных" из бесплатных провайдеров
        # Они будут вызываться по очереди, если предыдущий упал
        self.providers = [
            Nominatim(user_agent="fire_primary_bot_1", timeout=5),
            Photon(user_agent="fire_backup_bot_2", timeout=5), # Бесплатный, без жестких лимитов
            ArcGIS(user_agent="fire_emergency_bot_3", timeout=5) # Бесплатный для базового поиска
        ]
        
        # Локальная база: сюда можно закинуть все частые адреса и офисы из CSV
        self.local_cache = {
            "достык 15": {"lat": 51.1801, "lon": 71.4460, "address": "ул. Достык 15, Астана", "city": "Астана"},
            "назарбаева 50": {"lat": 43.2220, "lon": 76.8512, "address": "ул. Назарбаева 50, Алматы", "city": "Алматы"},
            # Координаты центров городов (Last Resort)
            "астана": {"lat": 51.1801, "lon": 71.4460, "address": "г. Астана", "city": "Астана"},
            "алматы": {"lat": 43.2220, "lon": 76.8512, "address": "г. Алматы", "city": "Алматы"},
            "шымкент": {"lat": 42.3417, "lon": 69.5901, "address": "г. Шымкент", "city": "Шымкент"}
        }

    async def geocode(self, address_query: str) -> Optional[Dict[str, Any]]:
        if not address_query:
            return None
            
        address_lower = address_query.lower()

        # ШАГ 1: Проверка локального кэша (Моментальный ответ)
        # Если мы уже искали этот адрес или он захардкожен - отдаем сразу
        for key, data in self.local_cache.items():
            if key in address_lower:
                logger.info(f"⚡ Локальный кэш сработал для: {address_query}")
                return data

        # ШАГ 2: Перебор "скамейки запасных" внешних API
        loop = asyncio.get_running_loop()
        for provider in self.providers:
            provider_name = provider.__class__.__name__
            try:
                # Запускаем синхронный метод geopy асинхронно
                location = await loop.run_in_executor(None, provider.geocode, address_query)
                
                if location:
                    logger.info(f"🌐 API {provider_name} успешно нашел: {address_query}")
                    res = {
                        "lat": location.latitude, 
                        "lon": location.longitude, 
                        "address": location.address,
                        "city": "Астана" if "Астана" in location.address else "Алматы" # Упрощенно
                    }
                    # Сохраняем успешный ответ в кэш, чтобы больше не дергать API для этого адреса
                    self.local_cache[address_lower] = res 
                    return res
                    
            except (GeocoderTimedOut, GeocoderServiceError, GeocoderQuotaExceeded) as e:
                logger.warning(f"⚠️ Провайдер {provider_name} отвалился (Ошибка/Лимит). Выпускаем запасного...")
                continue # Идем к следующему провайдеру в списке
            except Exception as e:
                logger.error(f"❌ Неизвестная ошибка геокодера {provider_name}: {e}")
                continue

        # ШАГ 3: Last Resort (Крайний случай)
        # Все API лежат. Пытаемся грубо найти название города в тексте
        logger.error(f"🚨 Все внешние API недоступны! Включаем грубый локальный поиск для: {address_query}")
        for city in ["алматы", "астана", "шымкент"]:
            if city in address_lower:
                return self.local_cache[city]

        # Если даже город не угадали
        return None

# Инициализируем балансировщик один раз
geo_balancer = GeoFallbackBalancer()