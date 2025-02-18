from llama_index.core.ingestion import IngestionCache, IngestionPipeline
from rnd_chat_with_doc.modules.config.global_settings import get_config
from rnd_chat_with_doc.modules.util.logging_manager import log_action

config = get_config()


def get_cache():
    try:
        path = config.get("CACHE_FILE")
        cached_hash = IngestionCache.from_persist_path(persist_path=path)
        log_action(type="CACHE", action="Cache file found.")
    except:
        log_action(type="CACHE", action="Cache file NOT found.")
        cached_hash = ""

    return cached_hash

