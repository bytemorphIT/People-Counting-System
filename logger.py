
from datetime import datetime
from config import DEBUG_PATH

def log_debug(message):
    with open(DEBUG_PATH, "a") as f:
        f.write(f"[{datetime.now()}] {message}\n")

def save_count(entry, exit_):
    fname = datetime.now().strftime("count_log_%Y-%m-%d.txt")
    with open(fname, "w") as f:
        f.write(f"People Entered: {entry}\nPeople Exited: {exit_}\n")
