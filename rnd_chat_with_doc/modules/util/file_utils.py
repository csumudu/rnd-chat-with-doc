import os


def get_abs_path(currentFie: str, relativePath: str, levelsToGoDown: int = 0) -> str:
    current_dir = os.path.dirname(os.path.abspath(currentFie))
    file_path = os.path.join(
        current_dir,
        *([".."] * levelsToGoDown),
        relativePath
    )

    target_dir = os.path.normpath(file_path)
    return target_dir
