import json
import os
import logging
import logging.handlers
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

_logging_initialized = False

class JSONLogger:
    """
    Saves detection results to a uniquely named JSON file per run.
    """
    @staticmethod
    def save(output_dir: str, result: dict) -> str:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now(timezone.utc)
        result["timestamp"] = timestamp.isoformat()
        filename = f"detection_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        output_path = os.path.join(output_dir, filename)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4, ensure_ascii=False)
        logger.info(f"Result saved: {output_path}")
        return output_path

def setup_logging(log_dir: str = "logs", level: int = logging.INFO):
    global _logging_initialized
    if _logging_initialized:
        return
    _logging_initialized = True
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "pipeline.log")
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    fh = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    root.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    ch.setFormatter(fmt)
    root.addHandler(ch)
    logging.info(f"Logging initialized — file: {log_file}")
