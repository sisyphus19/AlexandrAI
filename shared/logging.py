import logging
import json
import time
import sys
from datetime import datetime, timezone
from typing import Optional, Dict, Any

class JSONFormatter(logging.Formatter):
    """
    Structured JSON log formatter for production microservice observability.
    Outputs log records with request_id, service name, severity, timestamp, and latency metrics.
    """
    def __init__(self, service_name: str = "alexandrai-service"):
        super().__init__()
        self.service_name = service_name

    def format(self, record: logging.LogRecord) -> str:
        log_object: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "service": getattr(record, "service", self.service_name),
            "severity": record.levelname,
            "request_id": getattr(record, "request_id", "N/A"),
            "logger": record.name,
            "message": record.getMessage(),
        }

        if hasattr(record, "latency_ms") and record.latency_ms is not None:
            log_object["latency_ms"] = round(record.latency_ms, 2)

        if record.exc_info:
            log_object["exception"] = self.formatException(record.exc_info)

        if hasattr(record, "extra_fields") and isinstance(record.extra_fields, dict):
            log_object.update(record.extra_fields)

        return json.dumps(log_object)


def get_logger(service_name: str, log_level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(service_name)
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JSONFormatter(service_name=service_name))
        logger.addHandler(handler)
        logger.propagate = False
        
    return logger
