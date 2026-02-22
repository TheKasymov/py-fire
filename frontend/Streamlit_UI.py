import streamlit as st
import requests
import pandas as pd

# Настройка страницы
st.set_page_config(page_title="F.I.R.E. Dashboard", page_icon="🔥", layout="wide")

API_BASE_URL = "http://api:8000/api/v1"

st.title("🔥 F.I.R.E. — Freedom Intelligent Routing Engine")
st.markdown("Система интеллектуального распределения клиентских обращений")

# Вкладки для навигации
tab1, tab2 = st.tabs(["📄 Маршрутизация (Загрузка CSV)", "📊 ИИ-Ассистент (Star Task)"])

# --- ВКЛАДКА 1: МАРШРУТИЗАЦИЯ ---
with tab1:
    st.header("1. Загрузка данных")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        tickets_file = st.file_uploader("1. Обращения (tickets.csv)", type=["csv"])
    with col2:
        managers_file = st.file_uploader("2. Менеджеры (managers.csv)", type=["csv"])
    with col3:
        units_file = st.file_uploader("3. Офисы (business_units.csv)", type=["csv"])

    if st.button("🚀 Запустить распределение", type="primary"):
        if tickets_file and managers_file and units_file:
            with st.spinner("Анализируем обращения через ИИ и распределяем..."):
                # Подготовка файлов для отправки в API
                files = {
                    "tickets_file": (tickets_file.name, tickets_file.getvalue(), "text/csv"),
                    "managers_file": (managers_file.name, managers_file.getvalue(), "text/csv"),
                    "units_file": (units_file.name, units_file.getvalue(), "text/csv"),
                }
                
                try:
                    # Отправляем POST запрос на твой FastAPI
                    response = requests.post(f"{API_BASE_URL}/route-tickets", files=files)
                    
                    if response.status_code == 200:
                        results = response.json()
                        st.success(f"Успешно обработано {len(results)} обращений!")
                        
                        # Преобразуем JSON в красивую таблицу
                        table_data = []
                        for r in results:
                            with st.expander(f"📋 Детали тикета {r['ticket_id'][:8]}..."):
                                portrait = r.get("psychological_portrait", {})
                                
                                st.subheader("🧠 Психологический портрет клиента")
                                col_p1, col_p2 = st.columns(2)
                                
                                with col_p1:
                                    st.info(f"**Тип личности:** {portrait.get('profile_type')}")
                                    st.write(f"**Рекомендация:** {portrait.get('communication_recommendation')}")
                                
                                with col_p2:
                                    metrics = portrait.get("metrics", {})
                                    st.write(f"📈 Повторов слов: {metrics.get('word_repetition_count')}")
                                    st.write(f"❗ Эмоциональный фон: {'Высокий' if metrics.get('emotional_punctuation', 0) > 2 else 'Спокойный'}")
                            
                            analysis = r.get("analysis", {})
                            geo = r.get("geo") or {}
                            table_data.append({
                                "ID Обращения": r.get("ticket_id", "N/A")[:8] + "...",
                                "Тип": analysis.get("appeal_type", "-"),
                                "Тональность": analysis.get("sentiment", "-"),
                                "Приоритет": analysis.get("priority", "-"),
                                "Город (Гео)": geo.get("nearest_office", {}).get("name", "Не определён"),
                                "Назначенный Менеджер": r.get("assigned_manager", "-")
                            })
                            
                        df = pd.DataFrame(table_data)
                        st.dataframe(df, use_container_width=True)
                        
                    else:
                        st.error(f"Ошибка API: {response.text}")
                except Exception as e:
                    st.error(f"Не удалось подключиться к API. Убедитесь, что бэкенд запущен на порту 8000. Ошибка: {e}")
        else:
            st.warning("Пожалуйста, загрузите все три CSV файла.")


# --- ВКЛАДКА 2: STAR TASK (ИИ-АССИСТЕНТ) ---
with tab2:
    st.header("✨ ИИ-Ассистент для аналитики (Star Task)")
    st.markdown("Спросите ИИ о данных, например: *«Покажи распределение по приоритетам»* или *«Динамика по городам»*")
    
    query = st.text_input("Ваш запрос:")
    
    if st.button("Сгенерировать график", type="secondary"):
        if query:
            with st.spinner("ИИ анализирует базу и строит график..."):
                try:
                    # Обращаемся к новому эндпоинту, который ты добавил в main.py
                    res = requests.post(f"{API_BASE_URL}/ai-assistant/chart", json={"query": query})
                    
                    if res.status_code == 200:
                        chart_data = res.json()
                        
                        if "error" in chart_data:
                            st.warning(chart_data["error"])
                        else:
                            st.subheader(chart_data.get("title", "График"))
                            st.write(chart_data.get("description", ""))
                            
                            labels = chart_data.get("labels", [])
                            values = chart_data.get("values", [])
                            c_type = chart_data.get("chart_type", "bar")
                            
                            if labels and values:
                                # Подготавливаем DataFrame для графиков
                                df_chart = pd.DataFrame({"Показатель": labels, "Количество": values}).set_index("Показатель")
                                
                                # Отрисовка в зависимости от типа, который вернула Ollama
                                if c_type in ["bar", "pie", "doughnut"]:
                                    st.bar_chart(df_chart)
                                elif c_type == "line":
                                    st.line_chart(df_chart)
                                else:
                                    st.bar_chart(df_chart)
                            else:
                                st.info("Нет данных для отрисовки графика.")
                    else:
                        st.error(f"Ошибка генерации: {res.text}")
                except Exception as e:
                    st.error(f"Не удалось подключиться к API: {e}")
        else:
            st.warning("Введите запрос!")