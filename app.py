import streamlit as st
import os
from pathlib import Path
import datetime
import io
import cv2
import numpy as np
import tempfile
from PIL import Image
import imageio

# Извините, вычленил некоторые процессы - тяжело работалось с более чем 1000 строками кода уже...
from modules.logs import safe_logger
from modules.ml import get_cached_models, detect_single_frame, generate_tab_internal, detect_faces, prepare_yolo_dataset, train_yolo_model

# Настройка страницы
st.set_page_config(
    page_title="CV Exam Project",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# === КЭШИРОВАНИЕ МОДЕЛЕЙ ===
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
    st.write("Загрузите изображение (JPG, PNG, GIF) или видео (MP4, MOV, AVI) до 25 МБ и до 10 секунд.")
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
    st.write("Выберите актёра из списка, чтобы сгенерировать его изображение")
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


# === РЕЖИМ ДООБУЧЕНИЯ (ДВА ЭТАПА) ===
# === РЕЖИМ ДООБУЧЕНИЯ (ДВА ЭТАПА) ===
def training_page():
    logger = safe_logger
    st.header("🔄 Дообучение модели")

    # === ИНИЦИАЛИЗАЦИЯ СОСТОЯНИЙ ===
    if "training_stage" not in st.session_state:
        st.session_state.training_stage = "upload"  # upload, annotate, train

    if "annotated_images" not in st.session_state:
        st.session_state.annotated_images = []

    if "accepted_images" not in st.session_state:
        st.session_state.accepted_images = []

    if "current_image_idx" not in st.session_state:
        st.session_state.current_image_idx = 0

    # === ЭТАП 1: ЗАГРУЗКА АРХИВА ===
    if st.session_state.training_stage == "upload":
        st.subheader("📁 Шаг 1: Загрузка данных")
        st.write("Загрузите ZIP-архив с фотографиями одного человека для дообучения модели")

        uploaded_zip = st.file_uploader("Загрузите ZIP с фото", type="zip", key="zip_uploader")
        actor_name = st.text_input(
            "Имя актёра (латиницей, например: ben_afflek)",
            placeholder="ben_afflek",
            key="actor_name_input"
        )

        # === ПРОВЕРКА ГОТОВНОСТИ ===
        is_ready = uploaded_zip is not None and bool(actor_name.strip())

        if st.button("🔍 Выполнить разметку", disabled=not is_ready, type="primary", use_container_width=True):
            with st.spinner("🔄 Распаковка архива и детекция лиц..."):
                try:
                    if uploaded_zip is None:
                        st.error("Файл не загружен")
                        return

                    if not actor_name.strip():
                        st.error("Имя актёра не может быть пустым")
                        return

                    # Проверка размера файла (максимум 50 МБ для обучения)
                    if uploaded_zip.size > 50 * 1024 * 1024:
                        st.error(
                            f"Файл слишком большой! Максимальный размер: 50 МБ. Ваш файл: {uploaded_zip.size / (1024 * 1024):.1f} МБ")
                        logger.warning(
                            f"ДООБУЧЕНИЕ | Отклонён большой файл: {uploaded_zip.name} ({uploaded_zip.size // 1024} КБ)")
                        return

                    # Создаем временную директорию
                    with tempfile.TemporaryDirectory() as tmp_dir:
                        # Сохраняем ZIP файл
                        zip_path = os.path.join(tmp_dir, uploaded_zip.name)
                        with open(zip_path, "wb") as f:
                            f.write(uploaded_zip.getvalue())

                        # Распаковка архива
                        import zipfile
                        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                            zip_ref.extractall(tmp_dir)

                        # Поиск изображений
                        image_files = []
                        valid_extensions = {".jpg", ".jpeg", ".png"}
                        for root, _, files in os.walk(tmp_dir):
                            for f in files:
                                if Path(f).suffix.lower() in valid_extensions:
                                    image_files.append(os.path.join(root, f))

                        if not image_files:
                            st.error("В архиве не найдено изображений (JPG/PNG)!")
                            logger.warning("ДООБУЧЕНИЕ | В архиве нет изображений")
                            return

                        st.info(f"📂 Найдено изображений: {len(image_files)}")

                        # Загрузка изображений и детекция лиц
                        annotated_images = []
                        from PIL import Image
                        import time

                        progress_bar = st.progress(0)
                        status_text = st.empty()

                        for idx, img_path in enumerate(image_files):
                            try:
                                # Открываем изображение
                                img = Image.open(img_path).convert("RGB")

                                # Обновление прогресса
                                progress = int((idx + 1) / len(image_files) * 100)
                                progress_bar.progress(progress)
                                status_text.text(f"Обработка изображения {idx + 1}/{len(image_files)}")

                                # === ИСПОЛЬЗУЕМ УНИВЕРСАЛЬНУЮ ФУНКЦИЮ ДЕТЕКЦИИ ===
                                start_time = time.time()
                                faces = detect_faces(img)
                                detection_time = time.time() - start_time

                                logger.info(
                                    f"ML | Детекция для {os.path.basename(img_path)}: {len(faces)} лиц, время: {detection_time:.2f}с")

                                # Добавляем изображения с разметкой
                                for face in faces:
                                    bbox = face["box"]  # [x, y, width, height]

                                    # Проверяем, что bbox в пределах изображения
                                    img_width, img_height = img.size
                                    x, y, w, h = bbox
                                    if x < 0 or y < 0 or x + w > img_width or y + h > img_height:
                                        continue

                                    annotated_images.append({
                                        "original_path": img_path,
                                        "image": img.copy(),
                                        "bbox": bbox,
                                        "method_used": face.get("method", "unknown"),
                                        "confidence": face.get("confidence", 1.0),
                                        "accepted": True  # По умолчанию принимаем
                                    })
                            except Exception as e:
                                logger.warning(f"Не удалось обработать изображение {img_path}: {str(e)}")
                                continue

                        if not annotated_images:
                            st.error("Не удалось детектировать лица ни на одном изображении!")
                            logger.error("ДООБУЧЕНИЕ | Не найдено лиц на изображениях")
                            return

                        # Фильтрация дубликатов (одно лицо на изображение)
                        unique_images = {}
                        for item in annotated_images:
                            img_path = item["original_path"]
                            if img_path not in unique_images:
                                unique_images[img_path] = item
                            else:
                                # Оставляем лицо с максимальным confidence
                                if item.get("confidence", 0) > unique_images[img_path].get("confidence", 0):
                                    unique_images[img_path] = item

                        annotated_images = list(unique_images.values())

                        # Сохраняем в состояние
                        st.session_state.annotated_images = annotated_images
                        st.session_state.accepted_images = list(annotated_images)  # По умолчанию принимаем все
                        st.session_state.actor_name = actor_name.strip()
                        st.session_state.current_image_idx = 0
                        st.session_state.training_stage = "annotate"

                        # Статистика детекции
                        methods_used = set(img["method_used"] for img in annotated_images)
                        total_images = len(image_files)
                        detected_images = len(annotated_images)
                        detection_rate = detected_images / total_images * 100

                        st.success(
                            f"✅ Обнаружено лиц на {detected_images} из {total_images} изображений ({detection_rate:.1f}%)")
                        st.info(f"ℹ️ Использован метод детекции: {', '.join(methods_used)}")

                        # Автоматически переходим к проверке разметки
                        time.sleep(1)
                        st.rerun()  # ЗАМЕНА experimental_rerun НА rerun

                except Exception as e:
                    st.error(f"❌ Ошибка разметки: {str(e)}")
                    logger.error(f"ДООБУЧЕНИЕ_ОШИБКА | Этап: разметка | Ошибка: {str(e)}")

    # === ЭТАП 2: ПРОВЕРКА РАЗМЕТКИ ===
    if st.session_state.training_stage == "annotate":
        st.subheader("✅ Шаг 2: Проверка разметки")
        st.write("Проверьте разметку на каждом изображении. Отклоните изображения с некорректной разметкой.")

        if not st.session_state.annotated_images:
            st.warning("Нет изображений для проверки. Вернитесь к первому шагу.")
            if st.button("↩️ Вернуться к загрузке", use_container_width=True):
                st.session_state.training_stage = "upload"
                st.rerun()  # ЗАМЕНА experimental_rerun НА rerun
            return

        # Статистика
        total_images = len(st.session_state.annotated_images)
        accepted_count = sum(1 for img in st.session_state.annotated_images if img.get("accepted", False))

        col1, col2, col3 = st.columns(3)
        col1.metric("Всего изображений", total_images)
        col2.metric("Принято", accepted_count)
        col3.metric("Процент принятых", f"{accepted_count / total_images * 100:.0f}%")

        st.progress(accepted_count / total_images if total_images > 0 else 0)

        # Навигация по изображениям
        if total_images > 1:
            col1, col2, col3 = st.columns([1, 2, 1])
            with col1:
                if st.button("⬅️ Предыдущее", disabled=(st.session_state.current_image_idx <= 0),
                             use_container_width=True):
                    st.session_state.current_image_idx = max(0, st.session_state.current_image_idx - 1)
                    st.rerun()  # ЗАМЕНА experimental_rerun НА rerun
            with col2:
                st.markdown(f"### Изображение {st.session_state.current_image_idx + 1} из {total_images}")
            with col3:
                if st.button("Следующее ➡️", disabled=(st.session_state.current_image_idx >= total_images - 1),
                             use_container_width=True):
                    st.session_state.current_image_idx = min(total_images - 1, st.session_state.current_image_idx + 1)
                    st.rerun()  # ЗАМЕНА experimental_rerun НА rerun

        # Отображение текущего изображения с разметкой
        current_item = st.session_state.annotated_images[st.session_state.current_image_idx]
        img = current_item["image"].copy()
        bbox = current_item["bbox"]

        # Рисуем bbox на изображении
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        x, y, w, h = bbox
        draw.rectangle([x, y, x + w, y + h], outline="red", width=3)

        # Добавляем подпись с информацией
        confidence = current_item.get("confidence", 1.0)
        method = current_item.get("method_used", "unknown")
        draw.text((x, y - 25), f"Лицо ({method.upper()})", fill="red")
        if method == "mtcnn_gpu":
            draw.text((x, y - 10), f"Уверенность: {confidence:.2f}", fill="red")

        # Отображение
        st.image(img, caption=f"Разметка для: {st.session_state.actor_name}",
                 width='content')  # ЗАМЕНА use_column_width НА width=None

        # Кнопки принятия/отклонения с состоянием
        current_accepted = current_item.get("accepted", True)

        col1, col2 = st.columns(2)
        with col1:
            # ИСПРАВЛЕНО: кнопка активна, когда изображение НЕ принято
            if st.button("✅ Принять разметку", type="primary", use_container_width=True, disabled=current_accepted):
                current_item["accepted"] = True
                st.success("Изображение принято для обучения")
                st.rerun()  # ЗАМЕНА experimental_rerun НА rerun

        with col2:
            # ИСПРАВЛЕНО: кнопка активна, когда изображение принято
            if st.button("❌ Отклонить разметку", use_container_width=True, disabled=not current_accepted):
                current_item["accepted"] = False
                st.warning("Изображение исключено из обучения")
                st.rerun()  # ЗАМЕНА experimental_rerun НА rerun

        # Отображение текущего статуса
        status_emoji = "✅" if current_accepted else "❌"
        status_text = "принято" if current_accepted else "отклонено"
        st.markdown(f"### Статус изображения: {status_emoji} {status_text}")

        # Сводка по всем изображениям
        with st.expander("📊 Сводка по всем изображениям"):
            accepted_images = [img for img in st.session_state.annotated_images if img.get("accepted", False)]
            st.write(f"**Принято:** {len(accepted_images)} изображений")
            st.write(f"**Отклонено:** {total_images - len(accepted_images)} изображений")

            if accepted_images:
                methods_count = {}
                for img in accepted_images:
                    method = img.get("method_used", "unknown")
                    methods_count[method] = methods_count.get(method, 0) + 1

                st.write("**Методы детекции:**")
                for method, count in methods_count.items():
                    st.write(f"- {method.upper()}: {count} изображений")

        # Кнопки навигации
        col1, col2 = st.columns(2)
        with col1:
            if st.button("↩️ Вернуться к загрузке", use_container_width=True):
                st.session_state.training_stage = "upload"
                st.rerun()  # ЗАМЕНА experimental_rerun НА rerun
        with col2:
            min_images = 3
            if st.button("🚀 Начать обучение", disabled=(accepted_count < min_images), use_container_width=True):
                if accepted_count < min_images:
                    st.warning(f"Для обучения требуется минимум {min_images} изображения с корректной разметкой!")
                else:
                    st.session_state.training_stage = "train"
                    st.rerun()  # ЗАМЕНА experimental_rerun НА rerun

    # === ЭТАП 3: ОБУЧЕНИЕ МОДЕЛИ ===
    if st.session_state.training_stage == "train":
        st.subheader("🚀 Шаг 3: Обучение модели")
        st.write(f"Обучение модели на данных актёра: {st.session_state.actor_name}")

        accepted_images = [img for img in st.session_state.annotated_images if img.get("accepted", False)]
        accepted_count = len(accepted_images)

        if accepted_count == 0:
            st.error("Нет принятых изображений для обучения!")
            if st.button("↩️ Вернуться к проверке разметки", use_container_width=True):
                st.session_state.training_stage = "annotate"
                st.rerun()  # ЗАМЕНА experimental_rerun НА rerun
            return

        st.info(f"✅ **Принято для обучения:** {accepted_count} изображений")

        epochs = st.slider("Количество эпох обучения", min_value=5, max_value=100, value=20, key="epochs")
        batch_size_options = [4, 8, 16, 32]
        batch_size = st.selectbox("Размер батча", batch_size_options, index=1, key="batch_size")

        st.caption("""
        💡 **Рекомендации:**
        - Для небольшого количества изображений (3-10) используйте 15-25 эпох
        - Для большого количества изображений (>10) используйте 30-50 эпох
        - Размер батча 8 оптимален для большинства GPU
        """)

        if st.button("🔥 Запустить обучение", type="primary", use_container_width=True):
            # Прогресс-бар и статус
            progress_bar = st.progress(0)
            status_text = st.empty()
            result_container = st.container()

            try:
                # Подготовка датасета
                status_text.text("Подготовка датасета...")
                progress_bar.progress(5)

                # Создаем временную директорию для обучения
                dataset_dir = Path.cwd() / "temp_training_data"
                dataset_dir.mkdir(parents=True, exist_ok=True)

                # Подготовка датасета в формате YOLO
                images_dir, labels_dir, data_yaml_path = prepare_yolo_dataset(
                    accepted_images,
                    st.session_state.actor_name,
                    base_dir=dataset_dir
                )

                status_text.text(f"Датасет подготовлен: {accepted_count} изображений")
                progress_bar.progress(15)

                # Callback для прогресса
                def progress_callback(percent, text):
                    progress_bar.progress(percent)
                    status_text.text(text)

                # Обучение модели
                status_text.text("Начало обучения...")
                training_result = train_yolo_model(
                    data_yaml_path=data_yaml_path,
                    epochs=epochs,
                    batch_size=batch_size,
                    progress_callback=progress_callback,
                    status_callback=lambda text: status_text.text(text)
                )

                if training_result["success"]:
                    progress_bar.progress(100)
                    status_text.text("✅ Обучение завершено!")

                    # Отображение результатов
                    with result_container:
                        st.success("🎉 Модель успешно дообучена!")

                        # Метрики
                        metrics = training_result["metrics"]
                        col1, col2, col3, col4 = st.columns(4)

                        with col1:
                            st.metric("Precision", f"{metrics['precision']:.3f}")
                            st.caption("Точность детекции")

                        with col2:
                            st.metric("Recall", f"{metrics['recall']:.3f}")
                            st.caption("Полнота детекции")

                        with col3:
                            st.metric("mAP@0.5", f"{metrics['map50']:.3f}")
                            st.caption("Средняя точность при IoU=0.5")

                        with col4:
                            st.metric("mAP@0.5-0.95", f"{metrics['map50_95']:.3f}")
                            st.caption("Средняя точность при IoU=0.5-0.95")

                        # График обучения
                        results_path = metrics.get("results_path")
                        if results_path and os.path.exists(results_path):
                            st.subheader("📊 Графики обучения")
                            st.image(results_path, caption="Результаты обучения YOLO",
                                     width='content')  # ЗАМЕНА use_column_width НА width=None

                        # Загрузка новой модели
                        status_text.text("Загрузка новой модели...")
                        from ultralytics import YOLO
                        global yolo_model

                        new_model_path = training_result["model_path"]
                        try:
                            yolo_model = YOLO(new_model_path)
                            st.success("✅ Новая модель загружена и готова к использованию в детекции!")
                        except Exception as e:
                            logger.error(f"ДООБУЧЕНИЕ | Ошибка загрузки новой модели: {str(e)}")
                            st.warning(
                                "⚠️ Не удалось загрузить новую модель. Перезагрузите приложение для применения изменений.")

                        # Логирование
                        log_entry = (
                            f"ДООБУЧЕНИЕ_УСПЕШНО | Актёр: {st.session_state.actor_name} | "
                            f"Изображений: {accepted_count} | Эпохи: {epochs} | "
                            f"Precision: {metrics['precision']:.3f} | Recall: {metrics['recall']:.3f} | "
                            f"mAP@0.5: {metrics['map50']:.3f} | mAP@0.5-0.95: {metrics['map50_95']:.3f}"
                        )
                        logger.info(log_entry)


                else:
                    raise Exception(training_result["error"])

            except Exception as e:
                progress_bar.progress(0)
                status_text.text("❌ Ошибка обучения!")
                st.error(f"Ошибка обучения: {str(e)}")
                logger.error(
                    f"ДООБУЧЕНИЕ_ОШИБКА | Этап: обучение | Актёр: {st.session_state.actor_name} | Ошибка: {str(e)}")

    # === СПОЙЛЕР С ЛОГАМИ ===
    with st.expander("📄 Последние записи лога"):
        log_content = logger.read_last_lines(n=15)
        st.text_area("Содержимое лога", log_content, height=300, key="training_log")

        if logger.get_log_info()['exists']:
            with open(logger.log_path, 'rb') as f:
                st.download_button(
                    label="📥 Скачать полный лог",
                    data=f.read(),
                    file_name=f"training_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
                    mime="text/plain",
                    use_container_width=True
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
    st.caption(f"YOLO модель: {f'загружена ({yolo_model.ckpt_path})' if yolo_model else 'не загружена'}")
    st.caption(f"SD модель: {f'загружена ({sd_model.config._name_or_path})' if sd_model else 'не загружена'}")
