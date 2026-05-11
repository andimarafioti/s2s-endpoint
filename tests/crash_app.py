import os
import threading
import time

from fastapi import FastAPI

app = FastAPI()
CRASH_AFTER_S = int(os.environ.get("CRASH_AFTER_S", "60"))


@app.get("/health")
def health():
    return {"status": "ok"}


def _crash():
    time.sleep(CRASH_AFTER_S)
    os._exit(1)


threading.Thread(target=_crash, daemon=True).start()
