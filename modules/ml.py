import torch
import os
from pathlib import Path
from PIL import Image
import numpy as np
import cv2
import tempfile
import time
from typing import Optional, Tuple, List, Dict
from diffusers import AutoPipelineForText2Image
from ultralytics import YOLO
from typing import List, Dict, Optional

from modules.logs import safe_logger

# Глобальные переменные для кэширования
_cached_models = {
    "yolo": None,
    "sd": None,
    "load_status": None
}

######################################
########## Загрузка моделей ##########
######################################

def load_models_internal() -> Tuple[Optional[YOLO], Optional[AutoPipelineForText2Image], Dict]:
    """
    Внутренняя функция загрузки моделей БЕЗ Streamlit декораторов
    Возвращает кортеж: (yolo_model, sd_model, load_status)
    """
    logger = safe_logger
    logger.info("ML | Начало загрузки моделей")
    
    load_status = {
        "yolo_loaded": False,
        "sd_loaded": False,
        "device": "cpu",
        "accelerate_available": False,
        "errors": []
    }
    
    # 1. Проверка accelerate
    try:
        import accelerate
        load_status["accelerate_available"] = True
        logger.info("ML | accelerate доступен")
    except ImportError as e:
        logger.warning(f"ML | accelerate не найден: {str(e)}")
    
    # 2. Загрузка YOLO
    try:
        yolo_path = Path("models/YOLO/best.pt")
        if not yolo_path.exists():
            # Ищем в корне проекта
            project_root = Path.cwd()
            yolo_path = project_root / "models" / "best.pt"
            logger.warning(f"ML | YOLO модель не найдена в {yolo_path.parent}. Ищем в {project_root}")
        
        yolo_model = YOLO(str(yolo_path))
        load_status["yolo_loaded"] = True
        logger.info(f"ML | YOLO модель загружена с {yolo_path}")
    except Exception as e:
        logger.error(f"ML | Ошибка загрузки YOLO: {str(e)}")
        load_status["errors"].append(f"YOLO: {str(e)}")
        yolo_model = None
    
    # 3. Загрузка Stable Diffusion
    sd_model = None
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        load_status["device"] = device
        
        logger.info(f"ML | Устройство для SD: {device.upper()}, тип данных: {dtype}")
        
        sd_model = AutoPipelineForText2Image.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=dtype,
            safety_checker=None,
            use_safetensors=True,
            variant="fp16" if device == "cuda" else None
        ).to(device)
        
        # Применение оптимизаций
        if device == "cpu":
            logger.info("ML | Применение оптимизаций для CPU")
            sd_model.enable_attention_slicing()
            
            if load_status["accelerate_available"]:
                try:
                    sd_model.enable_model_cpu_offload()
                    logger.info("ML | CPU offload активирован")
                except Exception as e:
                    logger.warning(f"ML | CPU offload отключен: {str(e)}")
        
        load_status["sd_loaded"] = True
        logger.info(f"ML | Stable Diffusion загружена на {device.upper()}")
    except Exception as e:
        logger.error(f"ML | Ошибка загрузки SD: {str(e)}")
        load_status["errors"].append(f"SD: {str(e)}")
    
    return yolo_model, sd_model, load_status

def get_cached_models():
    """
    Безопасное получение кэшированных моделей с ленивой загрузкой
    """
    global _cached_models
    
    if _cached_models["yolo"] is None or _cached_models["sd"] is None:
        logger = safe_logger
        logger.info("ML | Кэшированные модели не найдены. Загружаем...")
        
        yolo, sd, status = load_models_internal()
        _cached_models["yolo"] = yolo
        _cached_models["sd"] = sd
        _cached_models["load_status"] = status
    
    return (
        _cached_models["yolo"],
        _cached_models["sd"],
        _cached_models["load_status"]
    )

######################################
############## ДЕТЕКЦИЯ ##############
######################################

def detect_single_frame(yolo_model, frame, scale, new_width, new_height) -> Dict:
    """
    Обработка одного кадра видео - чистая ML логика без UI
    """
    result = {
        "success": False,
        "error": None,
        "actors": [],
        "annotated_frame": None
    }
    
    try:
        # Детекция
        results = yolo_model(frame)
        
        # Сбор актёров
        actors_set = set()
        for result_det in results:
            for box in result_det.boxes:
                class_id = int(box.cls[0])
                actor_name = yolo_model.names[class_id]
                actors_set.add(actor_name)
        
        # Аннотация
        annotated_frame = results[0].plot()
        
        # Уменьшение разрешения
        if scale < 1.0:
            annotated_frame = cv2.resize(annotated_frame, (new_width, new_height))
        
        # Конвертация BGR -> RGB для imageio
        rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        
        result["actors"] = list(actors_set)
        result["annotated_frame"] = rgb_frame
        result["success"] = True
        
        return result
    
    except Exception as e:
        result["error"] = str(e)
        return result

#####################################
############# ГЕНЕРАЦИЯ #############
#####################################

#####################################
############# ГЕНЕРАЦИЯ #############
#####################################

def generate_tab_internal(
        sd_model,
        actor_name: str,
        prompt: str,
        negative_prompt: str,
        num_steps: int,
        resolution: int,
        embeddings_dir: str = "models/sd_embeddings",
        progress_callback=None,
        status_callback=None
) -> Dict:
    """
    Внутренняя функция генерации с callback для прогресса
    """
    logger = safe_logger
    logger.info(f"ML | Запуск генерации для актёра: {actor_name}")

    result = {
        "success": False,
        "error": None,
        "generated_image": None,
        "prompt_used": "",
        "negative_prompt_used": ""
    }

    try:
        # Проверка доступности модели
        if sd_model is None:
            raise ValueError("Stable Diffusion модель не загружена")

        # Обновление статуса
        if status_callback:
            status_callback("Подготовка к генерации...")

        # Загрузка эмбеддинга (безопасная версия)
        embeddings_path = Path(embeddings_dir)
        if not embeddings_path.exists():
            project_root = Path.cwd()
            embeddings_path = project_root / "models" / "sd_embeddings"

        embedding_path = embeddings_path / f"{actor_name}.bin"
        token_name = f"<{actor_name}>"

        if embedding_path.exists():
            if status_callback:
                status_callback(f"Загрузка эмбеддинга: {actor_name}")

            # === КРИТИЧЕСКИ ВАЖНОЕ ИСПРАВЛЕНИЕ: БЕЗОПАСНАЯ ЗАГРУЗКА ЭМБЕДДИНГА ===
            try:
                # Проверяем, существует ли токен уже в словаре
                tokenizer = sd_model.tokenizer
                text_encoder = sd_model.text_encoder

                if token_name in tokenizer.get_vocab():
                    logger.info(f"ML | Токен {token_name} уже существует в словаре. Пропускаем загрузку эмбеддинга.")
                else:
                    # Загружаем эмбеддинг только если токена нет
                    sd_model.load_textual_inversion(
                        str(embedding_path),
                        token=token_name
                    )
                    logger.info(f"ML | Эмбеддинг загружен: {actor_name}")
            except Exception as e:
                # Если ошибка про существующий токен - игнорируем
                if "already in tokenizer vocabulary" in str(e).lower():
                    logger.info(f"ML | Токен {token_name} уже загружен. Продолжаем генерацию.")
                else:
                    logger.warning(f"ML | Не удалось загрузить эмбеддинг: {str(e)}")
                    logger.warning("ML | Продолжаем генерацию без персонализации.")
        else:
            logger.warning(f"ML | Эмбеддинг не найден для {actor_name}")

        # Генерация
        if status_callback:
            status_callback(f"Генерация изображения ({num_steps} шагов)...")

        generator = torch.Generator(device=sd_model.device).manual_seed(42)

        # Автоматическое снижение разрешения для CPU
        height = width = resolution
        if sd_model.device.type == "cpu" and resolution > 256:
            height = width = 256
            logger.info(f"ML | CPU оптимизация: разрешение снижено до 256x256")

        # Автоматическое снижение шагов для CPU
        effective_steps = num_steps
        if sd_model.device.type == "cpu" and num_steps > 35:
            effective_steps = 35
            logger.info(f"ML | CPU оптимизация: шаги снижены с {num_steps} до {effective_steps}")

        with torch.autocast(sd_model.device.type):
            output = sd_model(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=effective_steps,
                guidance_scale=7.5,
                height=height,
                width=width,
                generator=generator,
                callback=progress_callback,
                callback_steps=1
            )

        # Обработка результата
        generated_image = output.images[0]
        result["generated_image"] = generated_image
        result["prompt_used"] = prompt
        result["negative_prompt_used"] = negative_prompt
        result["success"] = True

        logger.info(f"ML | Генерация успешна для {actor_name}")
        return result

    except Exception as e:
        logger.error(f"ML | Ошибка генерации для {actor_name}: {str(e)}")
        result["error"] = str(e)
        return result


######################################
########## ФУНКЦИИ ДООБУЧЕНИЯ ########
######################################

def detect_faces(image: Image.Image) -> List[Dict]:
    """
    Универсальная функция детекции лиц.
    Автоматически выбирает метод в зависимости от доступного оборудования (GPU/CPU).
    """
    logger = safe_logger

    # Определяем доступное устройство
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"ML | Детекция лиц: используемое устройство - {device}")

    try:
        if device == 'cuda':
            # Используем MTCNN из facenet_pytorch для GPU
            return detect_faces_mtcnn(image)
        else:
            # Используем OpenCV Haar Cascade для CPU
            return detect_faces_opencv(image)
    except Exception as e:
        logger.error(f"ML | Ошибка в универсальном детекторе лиц: {str(e)}")
        logger.warning("ML | Попытка запасного метода (OpenCV)")
        try:
            return detect_faces_opencv(image)
        except Exception as fallback_e:
            logger.error(f"ML | Ошибка запасного метода: {str(fallback_e)}")
            return []


def detect_faces_mtcnn(image: Image.Image) -> List[Dict]:
    """
    Детекция лиц с помощью MTCNN (facenet_pytorch) для GPU
    """
    logger = safe_logger
    logger.info("ML | Используем MTCNN (GPU) для детекции лиц")

    try:
        from facenet_pytorch import MTCNN

        # Инициализация детектора (параметры оптимизированы для качества)
        mtcnn = MTCNN(
            min_face_size=20,  # Минимальный размер лица
            thresholds=[0.6, 0.7, 0.7],  # Пороги для 3 сетей каскада
            factor=0.709,  # Параметр масштабирования
            post_process=False,  # Не нормализовать изображения
            device='cuda',  # Явно указываем GPU
            keep_all=True  # Возвращать все найденные лица
        )

        # Конвертация PIL в RGB numpy array
        img_array = np.array(image)

        # Детекция лиц
        boxes, probs = mtcnn.detect(img_array)

        results = []
        if boxes is not None:
            for i, box in enumerate(boxes):
                # Фильтрация по confidence (только лица с высокой вероятностью)
                if probs[i] < 0.9:
                    continue

                # Форматируем bbox: [x, y, width, height]
                x1, y1, x2, y2 = box
                bbox = [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]

                results.append({
                    "box": bbox,
                    "confidence": float(probs[i]),
                    "method": "mtcnn_gpu"
                })

        logger.info(f"ML | MTCNN обнаружил лиц: {len(results)}")
        return results

    except ImportError as e:
        logger.error(f"ML | Ошибка импорта facenet_pytorch: {str(e)}")
        logger.info("ML | Установите facenet-pytorch для GPU детекции: pip install facenet-pytorch==2.5.3")
        raise
    except Exception as e:
        logger.error(f"ML | Ошибка детекции MTCNN: {str(e)}")
        raise


def detect_faces_opencv(image: Image.Image) -> List[Dict]:
    """
    Детекция лиц с помощью OpenCV Haar Cascade для CPU
    """
    logger = safe_logger
    logger.info("ML | Используем OpenCV (CPU) для детекции лиц")

    try:
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        except AttributeError:
            # Резервный вариант для некоторых сборок OpenCV
            cascade_path = os.path.join(cv2.__path__[0], 'data', 'haarcascade_frontalface_default.xml')

        if not os.path.exists(cascade_path):
            logger.error(f"ML | Файл каскада не найден: {cascade_path}")
            raise FileNotFoundError(f"Файл каскада не найден: {cascade_path}")

        # Загружаем каскад
        face_cascade = cv2.CascadeClassifier(cascade_path)

        # Конвертация PIL в numpy array (RGB -> GRAY)
        img_array = np.array(image)
        img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

        # Детекция лиц с оптимальными параметрами для CPU
        faces = face_cascade.detectMultiScale(
            img_gray,
            scaleFactor=1.1,  # Шаг изменения размера изображения
            minNeighbors=3,  # Минимальное количество соседей для подтверждения лица
            minSize=(30, 30),  # Минимальный размер лица
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        results = []
        for (x, y, w, h) in faces:
            # Фильтрация по размеру (исключаем очень маленькие или очень большие лица)
            img_width, img_height = image.size
            face_area = w * h
            img_area = img_width * img_height

            # Лицо должно занимать от 1% до 40% площади изображения
            if face_area < 0.01 * img_area or face_area > 0.4 * img_area:
                continue

            results.append({
                "box": [x, y, w, h],
                "confidence": 1.0,  # OpenCV не предоставляет точные значения confidence
                "method": "opencv_cpu"
            })

        logger.info(f"ML | OpenCV обнаружил лиц: {len(results)}")
        return results

    except ImportError as e:
        logger.error(f"ML | Ошибка импорта OpenCV: {str(e)}")
        logger.info("ML | Установите OpenCV: pip install opencv-python-headless==4.8.0.76")
        raise
    except Exception as e:
        logger.error(f"ML | Ошибка детекции OpenCV: {str(e)}")
        raise

# Для обучения
def prepare_yolo_dataset(
        accepted_images: List[Dict],
        actor_name: str,
        base_dir: Path = None
) -> Tuple[Path, Path, Dict]:
    """
    Подготовка датасета в формате YOLO
    Возвращает пути к изображениям, аннотациям и конфигурацию датасета
    """
    logger = safe_logger
    logger.info(f"ML | Подготовка датасета для актёра: {actor_name}")

    try:
        if base_dir is None:
            base_dir = Path.cwd() / "temp_training_data"

        # Создаем структуру папок
        images_dir = base_dir / "images" / "train"
        labels_dir = base_dir / "labels" / "train"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        # Создаем файл классов
        classes_path = base_dir / "classes.txt"
        with open(classes_path, 'w', encoding='utf-8') as f:
            f.write(actor_name)

        # Сохраняем изображения и аннотации
        for idx, item in enumerate(accepted_images):
            image = item["image"]
            bbox = item["bbox"]

            # Имя файла
            img_filename = f"{actor_name}_{idx}.jpg"
            img_path = images_dir / img_filename

            # Сохраняем изображение
            image.save(img_path, quality=95)

            # Создаем аннотацию в формате YOLO
            img_width, img_height = image.size
            x_center = (bbox[0] + bbox[2] / 2) / img_width
            y_center = (bbox[1] + bbox[3] / 2) / img_height
            width = bbox[2] / img_width
            height = bbox[3] / img_height

            # Ограничиваем значения [0, 1]
            x_center = max(0, min(1, x_center))
            y_center = max(0, min(1, y_center))
            width = max(0, min(1, width))
            height = max(0, min(1, height))

            # Записываем аннотацию
            label_path = labels_dir / f"{actor_name}_{idx}.txt"
            with open(label_path, 'w', encoding='utf-8') as f:
                f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

        logger.info(f"ML | Датасет подготовлен: {len(accepted_images)} изображений")

        # Создаем конфигурацию датасета
        data_config = {
            "path": str(base_dir.resolve()),
            "train": "images/train",
            "val": "images/train",  # Для простоты используем те же изображения для валидации
            "test": "",
            "names": {0: actor_name},
            "nc": 1
        }

        # Сохраняем конфигурацию
        data_yaml_path = base_dir / "dataset.yaml"
        import yaml
        with open(data_yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(data_config, f, allow_unicode=True)

        return images_dir, labels_dir, str(data_yaml_path)

    except Exception as e:
        logger.error(f"ML | Ошибка подготовки датасета: {str(e)}")
        raise


def train_yolo_model(
    data_yaml_path: str,
    epochs: int = 10,
    batch_size: int = 8,
    progress_callback=None,
    status_callback=None
) -> Dict:
    """
    Дообучение YOLO модели на новом датасете
    """
    logger = safe_logger
    logger.info(f"ML | Начало дообучения, эпохи: {epochs}, batch_size: {batch_size}")

    try:
        from ultralytics import YOLO
        import time

        # Загружаем предобученную модель
        model = YOLO("models/YOLO/best.pt")  # Используем текущую модель

        # Обучение с правильным синтаксисом
        status_callback("Начало обучения модели...")

        # Выполняем обучение без callbacks
        results = model.train(
            data=data_yaml_path,
            epochs=epochs,
            batch=batch_size,
            imgsz=640,
            patience=10,  # Ранняя остановка при отсутствии улучшений
            save=True,
            plots=True,
            exist_ok=True,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

        # Эмулируем прогресс обучения для UI
        if progress_callback:
            for epoch in range(epochs):
                percent = int((epoch + 1) / epochs * 90)  # 90% для обучения
                progress_callback(percent, f"Эпоха {epoch + 1}/{epochs}")
                time.sleep(0.1)  # Небольшая задержка для визуализации

        # Получаем метрики
        metrics = {
            "precision": results.results_dict.get("metrics/precision(B)", 0.0),
            "recall": results.results_dict.get("metrics/recall(B)", 0.0),
            "map50": results.results_dict.get("metrics/mAP50(B)", 0.0),
            "map50_95": results.results_dict.get("metrics/mAP50-95(B)", 0.0),
            "results_path": str(Path(results.save_dir) / "results.png")
        }

        # Сохраняем модель
        model_path = Path("models/YOLO/new.pt")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(model_path)

        logger.info(f"ML | Дообучение завершено. Метрики: {metrics}")
        return {
            "success": True,
            "model_path": str(model_path),
            "metrics": metrics
        }

    except Exception as e:
        logger.error(f"ML | Ошибка дообучения: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }