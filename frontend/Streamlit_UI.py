import streamlit as st
import requests
import os
import pandas as pd

# Получаем URL API из окружения (в Docker это будет http://api:8000)
API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="AI Маршрутизатор", page_icon="🔥", layout="wide")

st.title("🔥 F.I.R.E: Умная маршрутизация обращений")
st.markdown("Загрузите файлы с данными, чтобы ИИ распределил тикеты по свободным менеджерам.")

# --- БЛОК 1: ЗАГРУЗКА ФАЙЛОВ ---
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("1. База тикетов")
    tickets_file = st.file_uploader("Загрузите билеты (.csv)", type=["csv"], key="tickets")

with col2:
    st.subheader("2. Менеджеры")
    managers_file = st.file_uploader("Загрузите менеджеров (.csv)", type=["csv"], key="managers")

with col3:
    st.subheader("3. Офисы")
    units_file = st.file_uploader("Загрузите филиалы (.csv)", type=["csv"], key="units")

# --- БЛОК 2: ОТПРАВКА НА СЕРВЕР ---
def upload_file_to_api(file, doc_type):
    if file is not None:
        files = {'file': (file.name, file.getvalue(), 'text/csv')}
        try:
            response = requests.post(f"{API_URL}/api/v1/upload/{doc_type}", files=files)
            if response.status_code == 200:
                return True, response.json().get('processed_count', 0)
            return False, f"Ошибка API: {response.status_code}"
        except Exception as e:
            return False, f"Нет связи с API: {str(e)}"
    return False, "Файл не выбран"

if st.button("📥 1. Загрузить файлы на сервер", use_container_width=True):
    if not all([tickets_file, managers_file, units_file]):
        st.error("Пожалуйста, выберите все три файла перед загрузкой!")
    else:
        with st.spinner("Отправка данных..."):
            # Загружаем по очереди
            s1, msg1 = upload_file_to_api(tickets_file, "tickets")
            s2, msg2 = upload_file_to_api(managers_file, "managers")
            s3, msg3 = upload_file_to_api(units_file, "units")
            
            if s1 and s2 and s3:
                st.success(f"✅ Все файлы загружены! (Тикетов: {msg1}, Менеджеров: {msg2})")
                st.session_state['files_uploaded'] = True
            else:
                st.error(f"❌ Ошибка загрузки. \nТикеты: {msg1}\nМенеджеры: {msg2}\nОфисы: {msg3}")

# --- БЛОК 3: ЗАПУСК ИИ-МАРШРУТИЗАЦИИ ---
st.divider()

if st.session_state.get('files_uploaded', False):
    if st.button("🚀 2. Запустить ИИ-распределение", type="primary", use_container_width=True):
        with st.spinner("🤖 ИИ анализирует тикеты и подбирает менеджеров... Это может занять время."):
            try:
                res = requests.post(f"{API_URL}/api/v1/route-tickets/execute")
                if res.status_code == 200:
                    data = res.json()
                    st.success(f"🎉 Успех! Распределено тикетов: {data.get('routed_tickets')}")
                    
                    # Предлагаем посмотреть историю
                    st.info("Перейдите в Telegram-бот и нажмите /history, чтобы увидеть результаты, или посмотрите ниже.")
                else:
                    st.error(f"❌ Ошибка сервера: {res.text}")
            except Exception as e:
                st.error(f"❌ Ошибка соединения: {str(e)}")

# --- БЛОК 4: ИСТОРИЯ (GET API) ---
st.divider()
if st.button("🔄 Обновить последние 10 записей"):
    try:
        res = requests.get(f"{API_URL}/api/v1/routing-history?limit=10")
        if res.status_code == 200:
            history = res.json()
            if history:
                # Превращаем JSON в красивую таблицу Pandas
                df = pd.json_normalize(history)
                
                # Переименовываем колонки для красоты
                df = df.rename(columns={
                    "ticket_guid": "ID Тикета",
                    "manager_fio": "Менеджер",
                    "assigned_office": "Офис",
                    "routing_reason": "Причина маршрутизации",
                    "sla_deadline": "SLA",
                    "ai_analysis.ticket_type": "Категория",
                    "ai_analysis.complexity_score": "Сложность",
                    "ai_analysis.is_critical": "Критично?"
                })
                
                # Оставляем только нужные колонки
                display_df = df[["ID Тикета", "Менеджер", "Офис", "Категория", "Сложность", "SLA", "Критично?"]]
                st.dataframe(display_df, use_container_width=True)
            else:
                st.info("История пуста. Запустите распределение.")
    except Exception as e:
        st.error(f"Не удалось загрузить историю: {e}")