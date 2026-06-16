import os
import sys
import time
from datetime import datetime
from enum import Enum
from typing import Optional, TextIO, Union
class Colors:
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    RESET = '\033[0m'
    BOLD = '\033[1m'
class LogLevel(Enum):
    DEBUG = (10, Colors.BLUE)
    OPTIMIZE = (25, Colors.CYAN)
    INFO = (20, Colors.GREEN)
    WARNING = (30, Colors.YELLOW)
    ERROR = (40, Colors.RED)
    CRITICAL = (50, Colors.MAGENTA)
class SimpleLogger:
    def __init__(
        self,
        name: str = "AutoEnv",
        log_level: Union[int, LogLevel] = LogLevel.INFO,
        log_file: Optional[str] = None,
        log_dir: str = "workspace/logs",
        console_output: bool = True
    ):
        self.name = name
        if isinstance(log_level, LogLevel):
            self.log_level = log_level.value[0]
        else:
            self.log_level = log_level
        self.console_output = console_output
        self.file_output = None
        self.level_display_names = {
            LogLevel.DEBUG: "DEBUG",
            LogLevel.OPTIMIZE: "OPTIMIZE",
            LogLevel.INFO: "INFO",
            LogLevel.WARNING: "WARNING",
            LogLevel.ERROR: "ERROR",
            LogLevel.CRITICAL: "CRITICAL"
        }
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            if log_file is None:
                current_date = datetime.now().strftime("%Y-%m-%d")
                log_file = f"{name}_{current_date}.log"
            file_path = os.path.join(log_dir, log_file)
            self.file_output = open(file_path, 'a', encoding='utf-8')
    def _log(self, level: LogLevel, message: str) -> None:
        if level.value[0] < self.log_level:
            return
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        level_name = self.level_display_names.get(level, level.name)
        formatted_msg = f"{timestamp} - {level_name} - {message}"
        if self.console_output:
            color = level.value[1]
            if level == LogLevel.CRITICAL:
                colored_msg = f"{Colors.BOLD}{color}{formatted_msg}{Colors.RESET}"
            else:
                colored_msg = f"{color}{formatted_msg}{Colors.RESET}"
            print(colored_msg)
        if self.file_output:
            self.file_output.write(formatted_msg + "\n")
            self.file_output.flush()
    def log_to_file(self, level: LogLevel, message: str) -> None:
        if level.value[0] < self.log_level:
            return
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        level_name = self.level_display_names.get(level, level.name)
        formatted_msg = f"{timestamp} - {level_name} - {message}"
        if self.file_output:
            self.file_output.write(formatted_msg + "\n")
            self.file_output.flush()
    def debug(self, message: str) -> None:
        self._log(LogLevel.DEBUG, message)
    def info(self, message: str) -> None:
        self._log(LogLevel.INFO, message)
    def optimize(self, message: str) -> None:
        self._log(LogLevel.OPTIMIZE, message)
    def warning(self, message: str) -> None:
        self._log(LogLevel.WARNING, message)
    def error(self, message: str, exc_info: bool = False) -> None:
        self._log(LogLevel.ERROR, message)
        if exc_info:
            import traceback
            self._log(LogLevel.ERROR, traceback.format_exc())
    def critical(self, message: str) -> None:
        self._log(LogLevel.CRITICAL, message)
    def agent_action(self, message: str) -> None:
        if self.log_level <= LogLevel.INFO.value[0]:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            formatted_msg = f"{timestamp} - AGENT_ACTION - {message}"
            if self.console_output:
                colored_msg = f"{Colors.BOLD}{Colors.CYAN}{formatted_msg}{Colors.RESET}"
                print(colored_msg)
        if self.file_output:
            self.file_output.write(formatted_msg + "\n")
            self.file_output.flush()
    def agent_thinking(self, message: str) -> None:
        if self.log_level <= LogLevel.INFO.value[0]:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            formatted_msg = f"{timestamp} - AGENT_THINKING - {message}"
            if self.console_output:
                colored_msg = f"{Colors.BOLD}{Colors.WHITE}{formatted_msg}{Colors.RESET}"
                print(colored_msg)
            if self.file_output:
                self.file_output.write(formatted_msg + "\n")
                self.file_output.flush()
    def __del__(self):
        if self.file_output:
            self.file_output.close()
logger = SimpleLogger()
def logger_to_optimize(message: str, file_path: Optional[str] = None, console: bool = True) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_msg = f"{timestamp} - OPTIMIZE - {message}"
    if console:
        try:
            colored_msg = f"{Colors.BOLD}{Colors.CYAN}{formatted_msg}{Colors.RESET}"
            print(colored_msg)
        except Exception:
            print(formatted_msg)
    if not file_path:
        log_dir = os.path.join("workspace", "logs")
        os.makedirs(log_dir, exist_ok=True)
        file_path = os.path.join(log_dir, "optimize.log")
    else:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
    try:
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(formatted_msg + "\n")
    except Exception:
        pass
def test_logger():
    test_log_dir = "test_logs"
    if not os.path.exists(test_log_dir):
        os.makedirs(test_log_dir)
    test_logger = SimpleLogger(
        name="test_logger",
        log_level=LogLevel.DEBUG,
        log_file="test_logger.log",
        log_dir=test_log_dir
    )
    print("\n===== Testing SimpleLogger =====\n")
    test_logger.debug("This is a DEBUG message - Should appear in BLUE")
    test_logger.info("This is an INFO message - Should appear in GREEN")
    test_logger.warning("This is a WARNING message - Should appear in YELLOW")
    test_logger.error("This is an ERROR message - Should appear in RED")
    test_logger.critical("This is a CRITICAL message - Should appear in BOLD MAGENTA")
    print("\n===== Testing Log Level Filtering =====\n")
    filtered_logger = SimpleLogger(
        name="filtered_logger",
        log_level=LogLevel.WARNING,
        log_file="filtered_logger.log",
        log_dir=test_log_dir
    )
    filtered_logger.debug("This DEBUG message should NOT appear")
    filtered_logger.info("This INFO message should NOT appear")
    filtered_logger.warning("This WARNING message should appear in YELLOW")
    filtered_logger.error("This ERROR message should appear in RED")
    filtered_logger.critical("This CRITICAL message should appear in BOLD MAGENTA")
    print("\n===== Verifying File Output =====\n")
    log_file_path = os.path.join(test_log_dir, "test_logger.log")
    if os.path.exists(log_file_path):
        with open(log_file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            print(f"Last 5 lines from log file ({log_file_path}):")
            for line in lines[-5:]:
                print(f"  {line.strip()}")
        print(f"\nLog file successfully created at: {log_file_path}")
    else:
        print(f"ERROR: Log file was not created at: {log_file_path}")
    print("\n===== Testing Complete =====\n")
def test_in_app_scenario():
    logger = SimpleLogger(name="app_logger")
    print("\n===== Simulating Application Logs =====\n")
    logger.info("Application starting up...")
    logger.debug("Loading configuration from config.json")
    time.sleep(0.5)
    logger.info("Configuration loaded successfully")
    logger.info("Processing data files...")
    time.sleep(0.5)
    logger.warning("Memory usage is high (85%)")
    try:
        result = 100 / 0
    except Exception as e:
        logger.error(f"Error during calculation: {str(e)}")
    logger.critical("Database connection lost! System cannot continue.")
    logger.info("Application shutting down")
    print("\n===== Simulation Complete =====\n")
if __name__ == "__main__":
    test_logger()
    test_in_app_scenario()
