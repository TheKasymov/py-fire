import csv
import io
from typing import List, Dict, Any

class CSVParser:
    @staticmethod
    def _to_clean_str(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value.strip()
        return str(value).strip()

    @staticmethod
    def _read_csv(file_obj) -> List[Dict[str, Any]]:
        """Вспомогательный метод для чтения CSV из байтового потока"""
        # Считываем байты и декодируем
        content = file_obj.read()
        if isinstance(content, bytes):
            # Пробуем разные кодировки, так как Excel часто сохраняет в cp1251
            try:
                text = content.decode('utf-8')
            except UnicodeDecodeError:
                text = content.decode('cp1251')
        else:
            text = content

        # Создаем объект StringIO для библиотеки csv
        file_like = io.StringIO(text)

        # Читаем с заголовками
        reader = csv.DictReader(file_like)

        # Очищаем ключи и значения от пробелов
        results = []
        for row in reader:
            clean_row = {
                CSVParser._to_clean_str(k): CSVParser._to_clean_str(v)
                for k, v in row.items()
                if k
            }
            results.append(clean_row)

        return results

    @staticmethod
    def parse_business_units(file) -> List[Dict[str, Any]]:
        raw_data = CSVParser._read_csv(file)
        # Приводим к единому формату, который ждет pipeline
        units = []
        for row in raw_data:
            units.append({
                "name": row.get("Офис"),
                "address": row.get("Адрес"),
                "city": row.get("Офис")  # В business_units.csv название офиса совпадает с городом
            })
        return units

    @staticmethod
    def parse_managers(file) -> List[Dict[str, Any]]:
        raw_data = CSVParser._read_csv(file)
        managers = []
        for row in raw_data:
            # Обрабатываем навыки (строка "VIP, ENG" -> список)
            skills_str = row.get("Навыки", "")
            skills = [s.strip() for s in skills_str.split(",")] if skills_str else []
            load_raw = row.get("Количество обращений в работе", "0")
            try:
                load = int(load_raw)
            except ValueError:
                load = 0

            managers.append({
                "name": row.get("ФИО"),
                "role": row.get("Должность"),
                "office": row.get("Офис"),
                "skills": skills,
                "load": load
            })
        return managers

    @staticmethod
    def parse_tickets(file) -> List[Dict[str, Any]]:
        raw_data = CSVParser._read_csv(file)
        tickets = []
        for row in raw_data:
            guid = row.get("GUID клиента")
            description = row.get("Описание", "")
            segment = row.get("Сегмент клиента") or "Mass"
            country = row.get("Страна", "")
            region = row.get("Область", "")
            city = row.get("Населённый пункт", "")
            street = row.get("Улица", "")
            house = row.get("Дом", "")

            tickets.append({
                "guid": guid,
                "id": guid,
                "text": description,
                "description": description,
                "segment": segment, # Mass, VIP и т.д.
                "country": country,
                "region": region,
                "city": city,
                "street": street,
                "house": house,
                "address": ", ".join(
                    [part for part in [country, region, city, street, house] if part]
                ),
                "manager_id": None # Будет заполнено алгоритмом
            })
        return tickets
