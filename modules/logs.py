import os
import sys
import logging
from pathlib import Path
import locale
from datetime import datetime

class SafeLogger:
    _instance = None
    _initialized = False
    
    def __new__(cls, log_dir="logs", log_name="app.log"):
        if cls._instance is None:
            cls._instance = super(SafeLogger, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, log_dir="logs", log_name="app.log"):
        if self._initialized:
            return
            
        self.project_root = Path.cwd()
        self.log_dir = self.project_root / log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.log_dir / log_name
        
        # Определяем кодировку системы
        self.encoding = self._detect_encoding()
        
        # Инициализируем логгер
        self.logger = self._setup_logger()
        self._initialized = True
    
    def _detect_encoding(self):
        """Автоопределение кодировки для текущей системы"""
        if os.name == 'nt':  # Windows
            try:
                enc = locale.getpreferredencoding()
                if any(code in enc.lower() for code in ['1251', 'cp1251', 'windows-1251']):
                    return 'cp1251'
                elif 'utf' in enc.lower():
                    return 'utf-8-sig'  # UTF-8 с BOM для Windows
                else:
                    return 'utf-8-sig'
            except:
                return 'cp1251'
        else:  # Linux/Mac
            return 'utf-8'
    
    def _setup_logger(self):
        """Настройка логгера без буферизации и дублирования"""
        # Удаляем ВСЕ существующие обработчики
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        
        # Создаем логгер с уникальным именем
        logger = logging.getLogger('cv_exam_safe_logger')
        logger.setLevel(logging.INFO)
        
        # Удаляем все существующие обработчики у этого логгера
        logger.handlers.clear()
        
        # Форматтер
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Обработчик файла
        file_handler = logging.FileHandler(
            filename=str(self.log_path.resolve()),
            encoding=self.encoding,
            mode='a',
            errors='replace'
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)
        
        # Обработчик консоли
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.INFO)
        
        # Добавляем обработчики
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        # Отключаем пропагацию
        logger.propagate = False
        
        # Гарантируем создание файла
        self._ensure_file_exists()
        
        return logger
    
    def _ensure_file_exists(self):
        """Гарантирует существование лог-файла с правильной кодировкой"""
        if not self.log_path.exists():
            with open(self.log_path, 'w', encoding=self.encoding) as f:
                if self.encoding == 'utf-8-sig':
                    f.write('\ufeff')
                f.write(f"[ИНФО] Лог-файл создан: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    def info(self, message):
        self.logger.info(message)
        self._flush_handlers()
    
    def error(self, message):
        self.logger.error(message)
        self._flush_handlers()
    
    def warning(self, message):
        self.logger.warning(message)
        self._flush_handlers()
    
    def _flush_handlers(self):
        for handler in self.logger.handlers:
            handler.flush()
    
    def read_last_lines(self, n=10):
        """Чтение последних n строк лога"""
        if not self.log_path.exists():
            return "Лог-файл не существует"
        
        if self.log_path.stat().st_size == 0:
            return "Лог-файл пуст"
        
        encodings_to_try = [self.encoding, 'utf-8', 'utf-8-sig', 'cp1251', 'latin-1']
        
        for enc in encodings_to_try:
            try:
                with open(self.log_path, 'r', encoding=enc, errors='replace') as f:
                    lines = f.readlines()
                    last_lines = lines[-n:] if len(lines) > n else lines
                    return ''.join(last_lines)
            except:
                continue
        
        return "Не удалось прочитать лог-файл"
    
    def get_log_info(self):
        info = {
            'path': str(self.log_path.resolve()),
            'exists': self.log_path.exists(),
            'size': self.log_path.stat().st_size if self.log_path.exists() else 0,
            'encoding': self.encoding,
            'permissions': oct(os.stat(self.log_path).st_mode)[-3:] if self.log_path.exists() else 'N/A'
        }
        return info

# Глобальный экземпляр логгера
safe_logger = SafeLogger(log_dir="logs", log_name="app.log")