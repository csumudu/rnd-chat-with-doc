from rnd_chat_with_doc.modules.util.file_utils import get_abs_path


paths = {
    "LOG_FILE": "data/session_data/user_actions.log",
    "SESSION_FILE": "data/session_data/user_session_state.yaml",
    "CACHE_FILE": "data/cache/pipeline_cache.json",
    "CONVERSATION_FILE": "data/cache/chat_history.json",
    "QUIZ_FILE": "data/cache/quiz.csv",
    "SLIDES_FILE": "data/cache/slides.json",
    "STORAGE_PATH": "data/ingestion_storage/",
    "INDEX_STORAGE": "data/index_storage",
    "LARGE_DOC_STORAGE": "data/large",
    "JSON_DATA": "data/json",
}
QUIZ_SIZE = 5
ITEMS_ON_SLIDE = 4


def get_config():
    config = {k: get_abs_path(__file__, v, 2) for k, v in paths.items()}
    return config
