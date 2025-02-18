from typing import Dict
from rnd_chat_with_doc.modules.config.global_settings import SESSION_FILE
import yaml
import os

from rnd_chat_with_doc.modules.util.file_utils import get_abs_path


def save_session(state: Dict[str, any]) -> None:
    to_be_saved = {key: value for key, value in state.items()}
    with open(SESSION_FILE, "w") as file:
        yaml.dump(to_be_saved, file)


def load_session(state: Dict[str, any]) -> bool:
    if os.path.exists(SESSION_FILE):
        with open(SESSION_FILE, "r") as file:
            try:
                existing = yaml.safe_load(file) or {}
                for k, v in existing.items():
                    state[k] = v
                return True
            except yaml.YAMLError:
                return False
    return False


if __name__ == "__main__":
    SESSION_FILE = get_abs_path(
        __file__, "data/session_data/user_session_state.yaml", 2
    )
    state = {
        "user_id": 12345,
        "session_token": "abcde12345",
        "preferences": {"theme": "dark", "notifications": True},
        "last_login": "2024-08-13T14:23:00Z",
    }

    save_session(state)

    existing = {}
    load_session(existing)
    print("loaded-->", existing)
