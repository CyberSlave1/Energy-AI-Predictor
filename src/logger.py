import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class Logger(metaclass=Singleton):
    def init_logging(self, log_dir: str = "logs", level: str = "DEBUG", size_mb: int = 1, max_files: int = 1) -> None:
        # Абсолютний шлях до logs в корені проєкту
        base_path = Path(__file__).resolve().parent.parent  # вихід з /src
        full_log_dir = base_path / log_dir
        full_log_dir.mkdir(parents=True, exist_ok=True)

        log_file_path = full_log_dir / "app.log"
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger("Rotating Log")
        self.logger.setLevel(level.upper())

        if not self.logger.handlers:
            file_handler = RotatingFileHandler(str(log_file_path), maxBytes=size_mb * 1024 * 1024, backupCount=max_files)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

    def _ensure_logger_initialized(self) -> None:
        if not hasattr(self, 'logger'):
            self.init_logging()

    def log_debug(self, msg: str) -> None:
        self._ensure_logger_initialized()
        self.logger.debug(msg)

    def log_info(self, msg: str) -> None:
        self._ensure_logger_initialized()
        self.logger.info(msg)

    def log_warning(self, msg: str) -> None:
        self._ensure_logger_initialized()
        self.logger.warning(msg)

    def log_error(self, msg: str) -> None:
        self._ensure_logger_initialized()
        self.logger.error(msg)