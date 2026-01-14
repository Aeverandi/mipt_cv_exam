import streamlit as st
import os
from pathlib import Path
import datetime
import io
from modules.logs import safe_logger
from modules.ml import get_cached_models, detect_single_frame, generate_tab_internal, train_tab_internal
import cv2
import numpy as np
import tempfile
from PIL import Image
import imageio

# Настройка страницы
st.set_page_config(
    page_title="CV Exam Project",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# === КЭШИРОВАНИЕ МОДЕЛЕЙ ЧЕРЕЗ STREAMLIT ===
@st.cache_resource
def load_models_with_cache():
    """Функция-обертка для Streamlit кэширования"""
    yolo, sd, status = get_cached_models()
    return yolo, sd, status

# Загрузка моделей при запуске
with st.spinner("🔄 Загрузка моделей (впервые может занять 1-2 минуты)..."):
    yolo_model, sd_model, load_status = load_models_with_cache()

# === ГЛАВНАЯ СТРАНИЦА ===
def main_page():
    st.title("🎭 Система детекции и генерации изображений актёров")
    
    # Отображение статуса загрузки
    with st.expander("🔧 Статус загрузки моделей", expanded=True):
        if load_status["yolo_loaded"]:
            st.success("✅ YOLO модель успешно загружена")
        else:
            st.error("❌ Ошибка загрузки YOLO")
        
        if load_status["sd_loaded"]:
            device = load_status["device"].upper()
            st.success(f"✅ Stable Diffusion загружена ({device})")
            
            if device == "CPU":
                if load_status["accelerate_available"]:
                    st.success("✅ CPU offload активирован через accelerate")
                else:
                    st.warning("⚠️ CPU offload отключен (библиотека 'accelerate' не установлена)")
                st.info("⚡ Применены базовые оптимизации: attention slicing")
        else:
            st.error("❌ Ошибка загрузки Stable Diffusion")
            if load_status["errors"]:
                for error in load_status["errors"]:
                    st.caption(f"• {error}")
        
        if not load_status["accelerate_available"] and load_status["device"] == "cpu":
            st.info("""
            💡 Совет для ускорения на CPU:
            Установите библиотеку accelerate: `pip install accelerate==1.12.0`
            Это ускорит генерацию изображений в 1.5-2 раза
            """)

# === РЕЖИМ ДЕТЕКЦИИ ===
def detection_page():
    logger = safe_logger
    st.header("🔍 Детекция актёров")
    
    uploaded_file = st.file_uploader("Выберите файл", type=["jpg", "jpeg", "png", "gif", "mp4", "mov", "avi"])
    
    if uploaded_file is not None:
        file_name = uploaded_file.name
        file_size = uploaded_file.size
        is_video = file_name.lower().endswith((".mp4", ".mov", ".avi"))
        
        if st.button("Запустить детекцию"):
            # === ПРОГРЕССБАРЫ - СОЗДАЕМ ДО ВЫЗОВА ML ===
            progress_bar = st.progress(0)
            status_text = st.empty()
            result_container = st.container()
            
            try:
                # === ОБРАБОТКА ИЗОБРАЖЕНИЯ ===
                if not is_video:
                    status_text.text("Загрузка изображения...")
                    progress_bar.progress(20)
                    
                    image = Image.open(io.BytesIO(uploaded_file.getvalue())).convert("RGB")
                    img_array = np.array(image)
                    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                    
                    status_text.text("Детекция объектов...")
                    progress_bar.progress(50)
                    
                    results = yolo_model(img_bgr)
                    
                    # Сбор актёров
                    detected_actors = []
                    for result_box in results[0].boxes:
                        class_id = int(result_box.cls[0])
                        actor_name = yolo_model.names[class_id]
                        if actor_name not in detected_actors:
                            detected_actors.append(actor_name)
                    
                    # Аннотация
                    annotated_frame = results[0].plot()
                    annotated_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                    annotated_image = Image.fromarray(annotated_rgb)
                    
                    progress_bar.progress(90)
                    status_text.text("Подготовка результата...")
                    
                    with result_container:
                        st.subheader("Результат детекции")
                        st.image(annotated_image, caption=f"Обнаружены: {', '.join(detected_actors) if detected_actors else 'никто'}")
                    
                    progress_bar.progress(100)
                    status_text.text("Готово!")
                    
                    # Логирование
                    log_entry = f"ДЕТЕКЦИЯ | Файл: {file_name} | Обнаружены: {', '.join(detected_actors) if detected_actors else 'нет данных'}"
                    logger.info(log_entry)
                    st.success("✅ Результаты сохранены в лог")
                
                # === ОБРАБОТКА ВИДЕО ===
                else:
                    status_text.text("Загрузка видео...")
                    progress_bar.progress(5)
                    
                    # Сохраняем во временный файл для OpenCV
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_input:
                        tmp_input.write(uploaded_file.getvalue())
                        input_path = tmp_input.name
                    
                    cap = cv2.VideoCapture(input_path)
                    fps = max(1, int(cap.get(cv2.CAP_PROP_FPS)))  # Защита от fps=0
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    
                    # Уменьшаем разрешение для совместимости
                    target_width = min(width, 640)
                    target_height = min(height, 480)
                    scale = min(target_width / width, target_height / height)
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    
                    # Обработка кадров
                    processed_frames = []
                    detected_actors_set = set()
                    
                    status_text.text(f"Обработка видео: 0/{frame_count} кадров")
                    progress_bar.progress(10)
                    
                    for frame_idx in range(frame_count):
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        # Обработка одного кадра через ML функцию
                        frame_result = detect_single_frame(
                            yolo_model=yolo_model,
                            frame=frame,
                            scale=scale,
                            new_width=new_width,
                            new_height=new_height
                        )
                        
                        processed_frames.append(frame_result["annotated_frame"])
                        detected_actors_set.update(frame_result["actors"])
                        
                        # Обновление прогресса
                        progress = 10 + int((frame_idx + 1) / frame_count * 80)
                        progress_bar.progress(min(progress, 95))
                        status_text.text(f"Обработка: {frame_idx + 1}/{frame_count} кадров")
                    
                    cap.release()
                    os.unlink(input_path)
                    
                    detected_actors = list(detected_actors_set)
                    
                    # === СОЗДАНИЕ ВИДЕО С IMAGEIO ===
                    status_text.text("Создание видео...")
                    progress_bar.progress(95)
                    
                    # Создаём буфер в памяти
                    output_buffer = io.BytesIO()
                    
                    # Записываем видео в буфер
                    with imageio.get_writer(
                        output_buffer, 
                        format='mp4', 
                        fps=fps,
                        codec='libx264',  # H.264 - поддерживается всеми браузерами
                        quality=7,        # Качество 0-10
                        pixelformat='yuv420p'  # Обязательно для совместимости
                    ) as writer:
                        for frame in processed_frames:
                            writer.append_data(frame)
                    
                    # Получаем байты видео
                    output_video_bytes = output_buffer.getvalue()
                    
                    # Отображаем видео
                    with result_container:
                        st.subheader("Результат детекции")
                        st.video(output_video_bytes)
                        st.caption(f"Обнаружены: {', '.join(detected_actors) if detected_actors else 'никто'}")
                    
                    progress_bar.progress(100)
                    status_text.text("Готово!")
                    
                    # Логирование
                    log_entry = f"ДЕТЕКЦИЯ | Файл: {file_name} | Тип: видео | Обнаружены: {', '.join(detected_actors) if detected_actors else 'нет данных'}"
                    logger.info(log_entry)
                    st.success("✅ Результаты сохранены в лог")
            
            except Exception as e:
                st.error(f"❌ Ошибка детекции: {str(e)}")
                logger.error(f"ДЕТЕКЦИЯ_ОШИБКА | Файл: {file_name} | Ошибка: {str(e)}")
    
    # === СПОЙЛЕР С ЛОГАМИ ===
    with st.expander("📄 Последние записи лога"):
        log_content = logger.read_last_lines(n=15)
        st.text_area("Содержимое лога", log_content, height=300, key="detection_log_display")
        
        if logger.get_log_info()['exists']:
            with open(logger.log_path, 'rb') as f:
                st.download_button(
                    label="📥 Скачать полный лог",
                    data=f.read(),
                    file_name=f"detection_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
                    mime="text/plain"
                )

# === РЕЖИМ ГЕНЕРАЦИИ ===
def generation_page():
    logger = safe_logger
    st.header("🎨 Генерация изображений")
    
    if sd_model is None:
        st.error("❌ Stable Diffusion не загружена. Генерация недоступна.")
        return
    
    # Получение списка актёров
    embeddings_path = Path("models/sd_embeddings")
    if not embeddings_path.exists():
        embeddings_path = Path.cwd() / "models" / "sd_embeddings"
    
    actor_names = [f.stem for f in embeddings_path.glob("*.bin")] if embeddings_path.exists() else []
    
    if not actor_names:
        st.warning("⚠️ Нет эмбеддингов в папке. Используются демо-актёры.")
        actor_names = ["tom_hanks", "angelina_jolie", "brad_pitt"]
    
    selected_actor = st.selectbox("Выберите актёра", actor_names, format_func=lambda x: x.replace("_", " ").title())
    
    # === ИНИЦИАЛИЗАЦИЯ ПРОМПТОВ В APP.PY ===
    default_prompt = (
        f"a high-quality professional portrait photograph of {selected_actor.replace('_', ' ')}, "
        "looking directly at camera, natural expression, cinematic lighting, "
        "4k resolution, detailed skin texture, professional color grading"
    )
    default_negative_prompt = (
        "blurry, low quality, distorted face, extra limbs, disfigured, "
        "bad anatomy, duplicate, morbid, mutilated, out of frame, "
        "extra fingers, mutated hands, poorly drawn hands, poorly drawn face, "
        "text, watermark, signature, cartoon, drawing, anime, 3d render"
    )
    
    # === СПОЙЛЕР С НАСТРОЙКАМИ ===
    with st.expander("📝 Настройки генерации"):
        prompt = st.text_area("Позитивный промпт", value=default_prompt, height=100, key=f"prompt_{selected_actor}")
        negative_prompt = st.text_area("Негативный промпт", value=default_negative_prompt, height=100, key=f"negative_prompt_{selected_actor}")
        num_steps = st.slider("Количество шагов генерации", 15, 75, 35, key=f"steps_{selected_actor}")
        
        # Автонастройка разрешения в зависимости от устройства
        is_cpu = sd_model.device.type == "cpu"
        resolution = st.selectbox(
            "Разрешение изображения",
            [256, 384, 512],
            index=0 if is_cpu else 2,
            help="Для CPU рекомендуется 256x256 для ускорения генерации",
            key=f"resolution_{selected_actor}"
        )
    
    if st.button("Сгенерировать изображение"):
        # === ПРОГРЕССБАРЫ - СОЗДАЕМ ДО ВЫЗОВА ML ===
        progress_bar = st.progress(0)
        status_text = st.empty()
        result_container = st.container()
        
        try:
            # Callback для прогресс-бара
            def update_progress(step, timestep, latents):
                progress = int(step / num_steps * 100)
                progress_bar.progress(min(progress, 99))
                status_text.text(f"Генерация: шаг {step}/{num_steps}")
            
            result = generate_tab_internal(
                sd_model=sd_model,
                actor_name=selected_actor,
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_steps=num_steps,
                resolution=resolution,
                embeddings_dir=str(embeddings_path),
                progress_callback=update_progress,
                status_callback=lambda text: status_text.text(text)
            )
            
            if result["success"]:
                with result_container:
                    st.subheader(f"✨ Сгенерированное изображение: {selected_actor.replace('_', ' ')}")
                    st.image(result["generated_image"], caption=f"Размер: {resolution}x{resolution} px", width=256)
                    
                    # Кнопка скачивания
                    img_byte_arr = io.BytesIO()
                    result["generated_image"].save(img_byte_arr, format='PNG')
                    st.download_button(
                        label="📥 Скачать изображение",
                        data=img_byte_arr.getvalue(),
                        file_name=f"generated_{selected_actor}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                        mime="image/png",
                        type="primary"
                    )
                
                status_text.text("✅ Генерация успешно завершена!")
                progress_bar.progress(100)
                
                # Логирование
                log_entry = f"ГЕНЕРАЦИЯ | Актёр: {selected_actor} | Промпт: {prompt[:50]}..."
                logger.info(log_entry)
                st.success("✅ Изображение сгенерировано и сохранено в лог")
            else:
                st.error(f"❌ Ошибка генерации: {result['error']}")
                logger.error(f"ГЕНЕРАЦИЯ_ОШИБКА | Актёр: {selected_actor} | Ошибка: {result['error']}")
        
        except Exception as e:
            st.error(f"❌ Критическая ошибка генерации: {str(e)}")
            logger.error(f"ГЕНЕРАЦИЯ_КРИТИЧЕСКАЯ_ОШИБКА | Актёр: {selected_actor} | Ошибка: {str(e)}")
    
    # === СПОЙЛЕР С ЛОГАМИ ===
    with st.expander("📄 Последние записи лога"):
        log_content = logger.read_last_lines(n=15)
        st.text_area("Содержимое лога", log_content, height=300, key="generation_log_display")
        
        if logger.get_log_info()['exists']:
            with open(logger.log_path, 'rb') as f:
                st.download_button(
                    label="📥 Скачать полный лог",
                    data=f.read(),
                    file_name=f"generation_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
                    mime="text/plain"
                )

# === РЕЖИМ ДООБУЧЕНИЯ ===
def training_page():
    logger = safe_logger
    st.header("🔄 Дообучение модели")
    
    uploaded_zip = st.file_uploader("Загрузите ZIP с фото", type="zip")
    actor_name = st.text_input("Имя актёра (латиницей, например: tom_hanks)")
    epochs = st.slider("Количество эпох", 1, 50, 10)
    
    if st.button("Начать дообучение", disabled=(not uploaded_zip or not actor_name)):
        # === ПРОГРЕССБАРЫ - СОЗДАЕМ ДО ВЫЗОВА ML ===
        progress_bar = st.progress(0)
        status_text = st.empty()
        result_container = st.container()
        
        try:
            # Callback для прогресс-бара
            def update_progress(value, text):
                progress_bar.progress(min(value, 100))
                status_text.text(text)
            
            result = train_tab_internal(
                yolo_model=yolo_model,
                zip_data=uploaded_zip.getvalue() if uploaded_zip else None,
                actor_name=actor_name,
                epochs=epochs,
                progress_callback=update_progress
            )
            
            if result["success"]:
                st.success(f"✅ Модель дообучена на {result['images_processed']} изображениях")
                st.metric("Итоговый mAP", f"{result['metrics']['mAP']:.3f}")
                
                # Логирование
                log_entry = f"ДООБУЧЕНИЕ | Актёр: {actor_name} | Эпохи: {epochs} | mAP: {result['metrics']['mAP']:.3f}"
                logger.info(log_entry)
                st.success("✅ Результаты дообучения сохранены в лог")
            else:
                st.error(f"❌ Ошибка дообучения: {result['error']}")
                logger.error(f"ДООБУЧЕНИЕ_ОШИБКА | Актёр: {actor_name} | Ошибка: {result['error']}")
        
        except Exception as e:
            st.error(f"❌ Критическая ошибка дообучения: {str(e)}")
            logger.error(f"ДООБУЧЕНИЕ_КРИТИЧЕСКАЯ_ОШИБКА | Актёр: {actor_name} | Ошибка: {str(e)}")
    
    # === СПОЙЛЕР С ЛОГАМИ ===
    with st.expander("📄 Последние записи лога"):
        log_content = logger.read_last_lines(n=15)
        st.text_area("Содержимое лога", log_content, height=300, key="training_log_display")
        
        if logger.get_log_info()['exists']:
            with open(logger.log_path, 'rb') as f:
                st.download_button(
                    label="📥 Скачать полный лог",
                    data=f.read(),
                    file_name=f"training_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
                    mime="text/plain"
                )

# === НАВИГАЦИЯ ===
st.sidebar.title("🚀 Компьютерное зрение: Экзамен")
mode = st.sidebar.radio(
    "Выберите режим",
    ["🏠 Главная", "👁️ Детекция", "🎨 Генерация", "🔄 Дообучение"],
    index=0
)

if mode == "🏠 Главная":
    main_page()
elif mode == "👁️ Детекция":
    detection_page()
elif mode == "🎨 Генерация":
    generation_page()
elif mode == "🔄 Дообучение":
    training_page()

# Отладочная информация (скрыта по умолчанию)
with st.sidebar.expander("🔧 Отладка", expanded=False):
    st.caption(f"Текущая директория: {Path.cwd()}")
    st.caption(f"YOLO модель: {'загружена' if yolo_model else 'не загружена'}")
    st.caption(f"SD модель: {'загружена' if sd_model else 'не загружена'}")