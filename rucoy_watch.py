import os
import re
import shlex
import signal
import subprocess
import time

import cv2
import numpy as np
import pytesseract
import requests
from pytesseract import Output

STREAM_URL = os.environ["STREAM_URL"]
TG_TOKEN = os.environ["TG_TOKEN"]
TG_CHAT_ID = os.environ["TG_CHAT_ID"]
MAX_SECONDS = int(os.environ.get("MAX_SECONDS", 5 * 3600 + 45 * 60))

FRAME = "frame.jpg"
INTERVAL = 3          # секунд между кадрами
COOLDOWN = 240        # не повторять одну и ту же находку столько секунд
BIG_TEXT_RATIO = 0.08 # высота "большой" надписи относительно высоты кадра

REGIONS = {
    "sa": "South America",
    "na": "North America",
    "eu": "Europe",
    "as": "Asia",
}
CODE_RE = re.compile(r"(sa|na|eu|as)(\d{1,2})")
CHAT_RE = re.compile(
    r"found\s+(.+?)\s*\[?\s*(Europe|Asia|North\s*Am\w*|South\s*Am\w*)\s*(\d{1,2})",
    re.IGNORECASE,
)

last_sent = {}
pending_code = None


def tg_photo(path, caption):
    try:
        with open(path, "rb") as f:
            requests.post(
                f"https://api.telegram.org/bot{TG_TOKEN}/sendPhoto",
                data={"chat_id": TG_CHAT_ID, "caption": caption},
                files={"photo": f},
                timeout=30,
            )
    except Exception as e:
        print("telegram error:", e)


def tg_text(text):
    try:
        requests.post(
            f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
            data={"chat_id": TG_CHAT_ID, "text": text},
            timeout=30,
        )
    except Exception as e:
        print("telegram error:", e)


def region_name(raw):
    r = raw.lower().replace(" ", "")
    if r.startswith("north"):
        return "North America"
    if r.startswith("south"):
        return "South America"
    return "Europe" if r.startswith("eu") else "Asia"


def find_big_code(img):
    """Крупная белая надпись стримера вида sa8 / na3 / eu2 / as1."""
    h = img.shape[0]
    mask = cv2.inRange(img, (200, 200, 200), (255, 255, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    ocr_img = 255 - mask
    cfg = (
        "--psm 11 -c tessedit_char_whitelist="
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    )
    data = pytesseract.image_to_data(ocr_img, config=cfg, output_type=Output.DICT)
    big = []
    for i, t in enumerate(data["text"]):
        t = t.strip().lower()
        if t and data["height"][i] >= h * BIG_TEXT_RATIO:
            big.append(t)
    candidates = big + [a + b for a, b in zip(big, big[1:])]
    for c in candidates:
        m = CODE_RE.fullmatch(c)
        if m:
            return m.group(1), int(m.group(2))
    return None


def find_chat_line(img):
    """Строка игрового чата: 'X found Haunted Willow [South America 8]'."""
    h, w = img.shape[:2]
    crop = img[int(h * 0.60):, : int(w * 0.65)]
    crop = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray, config="--psm 6")
    m = CHAT_RE.search(text)
    if m:
        boss = m.group(1).strip(" :-[")
        return boss, region_name(m.group(2)), int(m.group(3))
    return None


def should_send(key):
    now = time.time()
    if now - last_sent.get(key, 0) < COOLDOWN:
        return False
    last_sent[key] = now
    return True


def process(img):
    global pending_code

    chat = find_chat_line(img)
    if chat:
        boss, region, num = chat
        if should_send(("chat", boss.lower(), region, num)):
            tg_photo(FRAME, f"🔥 Босс: {boss}\n🌍 Сервер: {region} {num}")
            pending_code = None
            return

    code = find_big_code(img)
    if code is None:
        pending_code = None
        return
    # надпись должна продержаться 2 кадра подряд, чтобы не ловить мусор
    if code == pending_code:
        region = REGIONS[code[0]]
        if should_send(("big", region, code[1])):
            tg_photo(FRAME, f"🌍 Сервер: {region} {code[1]}\n(надпись стримера: {code[0]}{code[1]})")
    pending_code = code


def start_capture():
    cmd = (
        f"streamlink --stdout {shlex.quote(STREAM_URL)} 720p,480p,best "
        f"| ffmpeg -loglevel error -i pipe:0 -vf fps=1/{INTERVAL} "
        f"-q:v 2 -update 1 -y {FRAME}"
    )
    return subprocess.Popen(cmd, shell=True, preexec_fn=os.setsid)


def main():
    end = time.time() + MAX_SECONDS
    if os.environ.get("TEST") == "1":
        tg_text("✅ Бот запущен и следит за стримом")
    last_mtime = 0
    while time.time() < end:
        p = start_capture()
        while p.poll() is None and time.time() < end:
            try:
                m = os.path.getmtime(FRAME)
            except OSError:
                m = 0
            if m != last_mtime:
                last_mtime = m
                img = cv2.imread(FRAME)
                if img is not None:
                    try:
                        process(img)
                    except Exception as e:
                        print("process error:", e)
            time.sleep(1)
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
        except Exception:
            pass
        if time.time() < end:
            print("stream offline or dropped, retry in 60s")
            time.sleep(60)


if __name__ == "__main__":
    main()
