from rnd_chat_with_doc.modules.config.global_settings import get_config
from rnd_chat_with_doc.modules.util.file_utils import get_abs_path
from datetime import datetime

config = get_config()


def log_action(type: str, action: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("Now-->", datetime.now())
    log_entry = f"{timestamp} : {type} : {action}\n"
    with open(config.get("LOG_FILE"), "a") as file:
        print(log_entry)
        file.write(log_entry)


def log(msg="APP", *params) -> None:
    log_action(type=msg, action=f"{' '.join(map(str,params))}")


def reset_log() -> None:
    with open(config.get("LOG_FILE"), "w") as file:
        pass


if __name__ == "__main__":
    log_action("USER", "Log in")
    log_action("LLM", "Perform LLM query")
    log("RES",1 ,2 ,3)
    # reset_log()
