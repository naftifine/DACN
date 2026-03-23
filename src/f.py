import time
import sys
from dotenv import load_dotenv, set_key
import subprocess

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_FILE = os.path.join(BASE_DIR, "..", ".env")


def get_all_tomtom_keys():
    keys = []
    with open(ENV_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("#"):
                key = line.replace("#", "").strip()
                if key:
                    keys.append(key)
    return keys


def get_current_key():
    with open(ENV_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("TOMTOM_KEY="):
                return line.split("=", 1)[1].strip()
    return None


def rotate_key():
    keys = get_all_tomtom_keys()
    current = get_current_key()

    if not keys:
        print("No backup keys found")
        return

    if current not in keys:
        next_key = keys[0]
    else:
        idx = keys.index(current)
        next_key = keys[(idx + 1) % len(keys)]

    set_key(ENV_FILE, "TOMTOM_KEY", next_key, quote_mode="never")
    print(f"Switched to: {next_key}")


def run_job():
    load_dotenv(override=True)

    # chạy file h.py với mode dynamic
    subprocess.run(["python", "h.py", "dynamic"])


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else None

    if mode == "dynamic":
        # chạy 1 lần (manual)
        rotate_key()
        run_job()
    else:
        # auto 30p
        while True:
            rotate_key()
            run_job()
            time.sleep(900)
