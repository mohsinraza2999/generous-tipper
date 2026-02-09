import logging
import logging.handlers
from pathlib import Path
# -----------------------------
# Configure logging
# -----------------------------
log_path=Path("logs")
log_path.mkdir(parents=True, exist_ok=True)  # ensure logs directory exists

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Rotating file handler
file_handler = logging.handlers.RotatingFileHandler(
    "logs/applogs.log",       # log file path
    maxBytes=5 * 1024 * 1024,  # 5 MB per log file
    backupCount=3              # keep 3 backups (training.log.1, training.log.2, etc.)
)
file_handler.setLevel(logging.INFO)

# Formatter
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Attach handlers (avoid duplicates if already added)
if not logger.handlers:
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
