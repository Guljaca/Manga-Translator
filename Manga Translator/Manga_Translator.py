#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import requests
import pyautogui
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import threading
import gc
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QLineEdit, QSpinBox, QDoubleSpinBox, QProgressBar, QSystemTrayIcon, QMenu,
    QMainWindow, QGroupBox, QFormLayout, QMessageBox, QComboBox, QCheckBox,
    QFontComboBox, QFileDialog, QScrollArea
)
from PyQt6.QtCore import Qt, QSize, QTimer, QThread, pyqtSignal, QPoint, QRect
from PyQt6.QtGui import QPainter, QPen, QColor, QBrush, QIcon, QAction, QPixmap, QFont, QPainterPath, QFontMetrics

# Try to import torch for GPU detection
print("Python executable:", sys.executable)
try:
    import torch
    print("Torch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU name:", torch.cuda.get_device_name(0))
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️ PyTorch не установлен, GPU недоступен")

# ----------------------- [MULTI-OCR] Импорт OCR движков -----------------------
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    print("⚠️ pytesseract не установлен, Tesseract недоступен")

try:
    import winocr
    WINDOWS_OCR_AVAILABLE = True
except ImportError:
    WINDOWS_OCR_AVAILABLE = False
    print("⚠️ winocr не установлен, Windows OCR недоступен")

# ----------------------- НАСТРОЙКА ПУТЕЙ -----------------------
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
MOCA_MODEL_DIR = MODELS_DIR / "manga-ocr-base"
YOLO_MODEL_PATH = MODELS_DIR / "comic-speech-bubble-detector.pt"
EASYOCR_MODEL_DIR = MODELS_DIR / "easyocr_models"
TESSERACT_LANG_DIR = MODELS_DIR / "tessdata"
CONFIG_FILE = BASE_DIR / "config.json"

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "0"
os.environ["QT_SCALE_FACTOR"] = "1"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
TESSERACT_LANG_DIR.mkdir(exist_ok=True)

# ----------------------------------------------------------------------
# Списки поддерживаемых языков
# ----------------------------------------------------------------------
EASYOCR_LANGUAGES = {
    "Japanese": "ja",
    "Chinese (Simplified)": "ch_sim",
    "Chinese (Traditional)": "ch_tra",
    "English": "en",
    "Korean": "ko",
    "Russian": "ru",
    "French": "fr",
    "German": "de",
    "Spanish": "es",
    "Italian": "it",
    "Arabic": "ar",
    "Greek": "el",
    "Dutch": "nl",
    "Polish": "pl",
    "Portuguese": "pt",
    "Turkish": "tr",
    "Vietnamese": "vi"
}

TESSERACT_LANG_MAP = {
    "ja": "jpn",
    "ch_sim": "chi_sim",
    "ch_tra": "chi_tra",
    "en": "eng",
    "ko": "kor",
    "ru": "rus",
    "fr": "fra",
    "de": "deu",
    "es": "spa",
    "it": "ita",
    "ar": "ara",
    "el": "ell",
    "nl": "nld",
    "pl": "pol",
    "pt": "por",
    "tr": "tur",
    "vi": "vie"
}

SOURCE_LANGUAGES = [
    "Japanese", "Chinese (Simplified)", "Chinese (Traditional)",
    "English", "Korean", "Russian", "French", "German", "Spanish"
]

SOURCE_TO_EASYOCR = {
    "Japanese": "ja",
    "Chinese (Simplified)": "ch_sim",
    "Chinese (Traditional)": "ch_tra",
    "English": "en",
    "Korean": "ko",
    "Russian": "ru",
    "French": "fr",
    "German": "de",
    "Spanish": "es"
}

# ----------------------------------------------------------------------
# Настройки по умолчанию
# ----------------------------------------------------------------------
default_settings = {
    "LM_STUDIO_URL": "http://localhost:1234/v1",
    "MODEL_NAME": "llmfan46/gemma-4-26B-A4B-it-ultra-uncensored-heretic-GGUF",
    "CONTEXT_SIZE": 3,
    "TEMPERATURE": 0.3,
    "TIMEOUT": 30,
    "CONFIDENCE_THRESHOLD": 0.5,
    "AUTO_CHECK_INTERVAL": 2.0,
    "SOURCE_LANG": "Chinese (Simplified)",
    "TARGET_LANG": "Russian",
    "MODE": "manga",
    "MANGA_OCR_BACKEND": "manga_ocr",
    "TEXT_OCR_BACKEND": "easyocr",
    "OCR_LANG": "ch_sim",
    "EASYOCR_CONFIDENCE": 0.2,
    "TEXT_BOX_WIDTH_FACTOR": 1.5,
    "TEXT_BOX_HEIGHT_FACTOR": 1.5,
    "ENABLE_OVERLAP_RELOCATION": True,
    "FONT_FAMILY": "Arial",
    "FONT_SIZE": 12,
    "OCR_ENGINES": ["easyocr"],
    "OCR_STRATEGY": "best_confidence",
    "TESSERACT_CONFIDENCE": 0.3,
    "WINDOWS_OCR_CONFIDENCE": 0.3,
    "TESSERACT_PATH": "",
    "USE_GPU_OCR": True,
    "USE_GPU_MODEL": False,
    # Новые настройки для асимметричной (параллельной) обработки
    "ASYMMETRIC_TRANSLATION": False,
    "MAX_CONCURRENT_REQUESTS": 5
}

settings = default_settings.copy()
translation_history = []
history_lock = threading.Lock()

mocr = None
yolo = None
easyocr_reader = None
current_ocr_lang = None

# ----------------------------------------------------------------------
# Загрузка / сохранение настроек
# ----------------------------------------------------------------------
def load_settings():
    global settings
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                saved = json.load(f)
            for key in default_settings:
                if key in saved:
                    settings[key] = saved[key]
            if "USE_GPU" in saved and "USE_GPU_OCR" not in saved:
                settings["USE_GPU_OCR"] = saved["USE_GPU"]
            # Добавленные ключи, если их нет в сохранённом файле
            if "ASYMMETRIC_TRANSLATION" not in settings:
                settings["ASYMMETRIC_TRANSLATION"] = False
            if "MAX_CONCURRENT_REQUESTS" not in settings:
                settings["MAX_CONCURRENT_REQUESTS"] = 5
            print("✅ Настройки загружены из config.json")
        except Exception as e:
            print(f"⚠️ Ошибка загрузки настроек: {e}")
    if "USE_GPU_OCR" not in settings:
        settings["USE_GPU_OCR"] = TORCH_AVAILABLE and torch.cuda.is_available()
        print(f"🔧 Автоопределение GPU для OCR: {settings['USE_GPU_OCR']}")

def save_settings_to_file():
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=2, ensure_ascii=False)
        print("✅ Настройки сохранены в config.json")
    except Exception as e:
        print(f"⚠️ Ошибка сохранения настроек: {e}")

# ----------------------------------------------------------------------
# Управление моделями OCR
# ----------------------------------------------------------------------
def unload_all_ocr():
    global mocr, easyocr_reader
    if mocr is not None:
        del mocr
        mocr = None
    if easyocr_reader is not None:
        del easyocr_reader
        easyocr_reader = None
    gc.collect()
    print("🧹 Все OCR-модели выгружены")

def unload_mode_models(mode):
    global yolo
    if mode == "text":
        if yolo is not None:
            del yolo
            yolo = None
            gc.collect()
            print("🧹 YOLO выгружен")

def is_gpu_available_for_ocr():
    """Проверяет, доступен ли GPU и разрешён ли в настройках для OCR."""
    if not settings.get("USE_GPU_OCR", False):
        return False
    if not TORCH_AVAILABLE:
        return False
    return torch.cuda.is_available()

def ensure_yolo():
    global yolo
    if yolo is None:
        device = 'cuda' if is_gpu_available_for_ocr() else 'cpu'
        print(f"🔄 Загрузка YOLO (детектор баблов) на устройстве {device}...")
        from ultralytics import YOLO
        yolo = YOLO(str(YOLO_MODEL_PATH))
        if device == 'cuda':
            yolo.to('cuda')
        print(f"✅ YOLO загружен на {device}")
    return yolo

def ensure_manga_ocr():
    global mocr
    if mocr is None:
        print("🔄 Загрузка Manga OCR...")
        from manga_ocr import MangaOcr
        device = 'cuda' if is_gpu_available_for_ocr() else 'cpu'
        print(f"   используем устройство: {device}")
        try:
            mocr = MangaOcr(pretrained_model_name_or_path=str(MOCA_MODEL_DIR), device=device)
        except TypeError:
            mocr = MangaOcr(pretrained_model_name_or_path=str(MOCA_MODEL_DIR))
            if device == 'cuda' and hasattr(mocr, 'model') and hasattr(mocr.model, 'to'):
                mocr.model.to('cuda')
                print("   Manga OCR принудительно перемещён на GPU")
        print("✅ Manga OCR загружена")
    return mocr

def ensure_easyocr():
    global easyocr_reader, current_ocr_lang
    lang = settings["OCR_LANG"]
    if easyocr_reader is None or current_ocr_lang != lang:
        if easyocr_reader is not None:
            del easyocr_reader
            gc.collect()
        use_gpu = is_gpu_available_for_ocr()
        print(f"🔄 Загрузка EasyOCR (язык: {lang}, GPU={use_gpu}, кэш в {EASYOCR_MODEL_DIR})...")
        import easyocr
        EASYOCR_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        easyocr_reader = easyocr.Reader([lang], gpu=use_gpu, model_storage_directory=str(EASYOCR_MODEL_DIR))
        current_ocr_lang = lang
        print(f"✅ EasyOCR загружен (GPU={use_gpu})")
    return easyocr_reader

def set_tesseract_path():
    if not TESSERACT_AVAILABLE:
        return False
    tesseract_path = settings.get("TESSERACT_PATH", "")
    if tesseract_path and os.path.exists(tesseract_path):
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
        return True
    else:
        import shutil
        default_path = shutil.which("tesseract")
        if default_path:
            pytesseract.pytesseract.tesseract_cmd = default_path
            return True
        fallback = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        if os.path.exists(fallback):
            pytesseract.pytesseract.tesseract_cmd = fallback
            return True
        return False

# ----------------------------------------------------------------------
# OCR функции для получения цельного текста (используется в manga-режиме и multi)
# ----------------------------------------------------------------------
def ocr_with_tesseract(img_cv, lang_code):
    if not TESSERACT_AVAILABLE:
        return "", 0.0
    if not set_tesseract_path():
        return "", 0.0
    tesseract_lang = TESSERACT_LANG_MAP.get(lang_code, "eng")
    tesseract_dir = os.path.dirname(pytesseract.pytesseract.tesseract_cmd)
    tessdata_dir = os.path.join(tesseract_dir, "tessdata")
    lang_file = os.path.join(tessdata_dir, f"{tesseract_lang}.traineddata")
    if not os.path.exists(lang_file):
        alt_tessdata = TESSERACT_LANG_DIR
        alt_lang_file = alt_tessdata / f"{tesseract_lang}.traineddata"
        if alt_lang_file.exists():
            tessdata_dir = str(alt_tessdata)
        else:
            return "", 0.0
    try:
        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        os.environ['TESSDATA_PREFIX'] = tessdata_dir
        custom_config = r'--oem 3 --psm 6'
        data = pytesseract.image_to_data(img_rgb, lang=tesseract_lang, config=custom_config, output_type=pytesseract.Output.DICT)
        texts = []
        confs = []
        for i, conf in enumerate(data['conf']):
            if conf != '-1' and int(conf) > 0:
                text = data['text'][i].strip()
                if text:
                    texts.append(text)
                    confs.append(int(conf))
        if texts:
            full_text = " ".join(texts)
            avg_conf = sum(confs) / len(confs) / 100.0
            return full_text, avg_conf
        return "", 0.0
    except Exception as e:
        print(f"⚠️ Tesseract ошибка: {e}")
        return "", 0.0

def ocr_with_windows_ocr(img_cv, lang_code):
    if not WINDOWS_OCR_AVAILABLE or sys.platform != "win32":
        return "", 0.0
    winocr_lang_map = {
        "ch_sim": "zh-CN",
        "ch_tra": "zh-TW",
        "ja": "ja-JP",
        "en": "en-US",
        "ko": "ko-KR",
        "ru": "ru-RU",
    }
    win_lang = winocr_lang_map.get(lang_code, "en-US")
    try:
        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        result = winocr.recognize_pil_sync(pil_img, win_lang)
        if result and result.get('text'):
            text = result['text'].strip()
            avg_conf = min(0.9, len(text) / 100.0)
            return text, avg_conf
        return "", 0.0
    except Exception as e:
        print(f"⚠️ Windows OCR ошибка: {e}")
        return "", 0.0

def ocr_with_easyocr(img_cv, lang_code):
    reader = ensure_easyocr()
    try:
        results = reader.readtext(img_cv, paragraph=True)
        if not results:
            return "", 0.0
        text = results[0][1]
        word_results = reader.readtext(img_cv, paragraph=False)
        if word_results:
            confs = [conf for (_, _, conf) in word_results if conf > 0]
            avg_conf = sum(confs)/len(confs) if confs else 0.5
        else:
            avg_conf = 0.5
        return text.strip(), avg_conf
    except Exception as e:
        print(f"⚠️ EasyOCR ошибка: {e}")
        return "", 0.0

def multi_ocr_recognize(img_cv, lang_code, engines=None, strategy="best_confidence"):
    """Используется ТОЛЬКО для manga-режима. Возвращает один текст."""
    if engines is None:
        engines = settings.get("OCR_ENGINES", ["easyocr"])
    results = []
    if "easyocr" in engines:
        text, conf = ocr_with_easyocr(img_cv, lang_code)
        if text and conf >= settings.get("EASYOCR_CONFIDENCE", 0.2):
            results.append(("easyocr", text, conf))
    if "tesseract" in engines and TESSERACT_AVAILABLE:
        text, conf = ocr_with_tesseract(img_cv, lang_code)
        if text and conf >= settings.get("TESSERACT_CONFIDENCE", 0.3):
            results.append(("tesseract", text, conf))
    if "windows_ocr" in engines and WINDOWS_OCR_AVAILABLE and sys.platform == "win32":
        text, conf = ocr_with_windows_ocr(img_cv, lang_code)
        if text and conf >= settings.get("WINDOWS_OCR_CONFIDENCE", 0.3):
            results.append(("windows_ocr", text, conf))
    if not results:
        return ""
    if strategy == "best_confidence":
        best = max(results, key=lambda x: x[2])
        return best[1]
    elif strategy == "voting":
        longest = max(results, key=lambda x: len(x[1]))
        return longest[1]
    else:
        return results[0][1]

# ----------------------------------------------------------------------
# ПОБЛОЧНЫЕ OCR (для текстового режима)
# ----------------------------------------------------------------------
def ocr_with_tesseract_blocks(img_cv, lang_code):
    if not TESSERACT_AVAILABLE:
        return []
    if not set_tesseract_path():
        return []
    tesseract_lang = TESSERACT_LANG_MAP.get(lang_code, "eng")
    tesseract_dir = os.path.dirname(pytesseract.pytesseract.tesseract_cmd)
    tessdata_dir = os.path.join(tesseract_dir, "tessdata")
    lang_file = os.path.join(tessdata_dir, f"{tesseract_lang}.traineddata")
    if not os.path.exists(lang_file):
        alt_tessdata = TESSERACT_LANG_DIR
        alt_lang_file = alt_tessdata / f"{tesseract_lang}.traineddata"
        if alt_lang_file.exists():
            tessdata_dir = str(alt_tessdata)
        else:
            return []
    try:
        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        os.environ['TESSDATA_PREFIX'] = tessdata_dir
        custom_config = r'--oem 3 --psm 6'
        data = pytesseract.image_to_data(img_rgb, lang=tesseract_lang, config=custom_config, output_type=pytesseract.Output.DICT)
        blocks = {}
        n_boxes = len(data['level'])
        for i in range(n_boxes):
            level = data['level'][i]
            if level == 5:
                text = data['text'][i].strip()
                if not text:
                    continue
                conf = int(data['conf'][i]) / 100.0 if data['conf'][i] != '-1' else 0.0
                if conf < settings.get("TESSERACT_CONFIDENCE", 0.3):
                    continue
                x = data['left'][i]
                y = data['top'][i]
                w = data['width'][i]
                h = data['height'][i]
                blocks[i] = (x, y, w, h, text, conf)
        result = list(blocks.values())
        result.sort(key=lambda b: (b[1], b[0]))
        return result
    except Exception as e:
        print(f"⚠️ Tesseract блоки ошибка: {e}")
        return []

def ocr_with_windows_ocr_blocks(img_cv, lang_code):
    if not WINDOWS_OCR_AVAILABLE or sys.platform != "win32":
        return []
    winocr_lang_map = {
        "ch_sim": "zh-CN",
        "ch_tra": "zh-TW",
        "ja": "ja-JP",
        "en": "en-US",
        "ko": "ko-KR",
        "ru": "ru-RU",
    }
    win_lang = winocr_lang_map.get(lang_code, "en-US")
    try:
        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        result = winocr.recognize_pil_sync(pil_img, win_lang)
        if not result or not result.get('lines'):
            return []
        
        blocks = []
        for line in result['lines']:
            words = line.get('words', [])
            if not words:
                continue
            line_text = ' '.join([w['text'] for w in words]).strip()
            if not line_text:
                continue
            all_rects = [w['bounding_rect'] for w in words if 'bounding_rect' in w]
            if not all_rects:
                continue
            x = min(r[0] for r in all_rects)
            y = min(r[1] for r in all_rects)
            x2 = max(r[0] + r[2] for r in all_rects)
            y2 = max(r[1] + r[3] for r in all_rects)
            w = x2 - x
            h = y2 - y
            confs = [w.get('confidence', 0.5) for w in words if 'confidence' in w]
            avg_conf = sum(confs)/len(confs) if confs else 0.5
            if avg_conf < settings.get("WINDOWS_OCR_CONFIDENCE", 0.3):
                continue
            pad = 5
            x = max(0, x - pad)
            y = max(0, y - pad)
            w = min(img_cv.shape[1] - x, w + 2*pad)
            h = min(img_cv.shape[0] - y, h + 2*pad)
            blocks.append((x, y, w, h, line_text, avg_conf))
        blocks.sort(key=lambda b: (b[1], b[0]))
        return blocks
    except Exception as e:
        print(f"⚠️ Windows OCR блоки ошибка: {e}")
        return []

def ocr_with_easyocr_blocks(img_cv, lang_code):
    reader = ensure_easyocr()
    confidence_threshold = settings.get("EASYOCR_CONFIDENCE", 0.2)
    results = reader.readtext(img_cv, paragraph=False)
    blocks = []
    for (bbox, text, conf) in results:
        if conf < confidence_threshold or len(text.strip()) < 2:
            continue
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        x = int(min(xs))
        y = int(min(ys))
        w = int(max(xs) - x)
        h = int(max(ys) - y)
        pad = 15
        x = max(0, x - pad)
        y = max(0, y - pad)
        w = min(img_cv.shape[1] - x, w + 2*pad)
        h = min(img_cv.shape[0] - y, h + 2*pad)
        blocks.append((x, y, w, h, text, conf))
    return blocks

# ----------------------------------------------------------------------
# Функции OCR для режимов
# ----------------------------------------------------------------------
def ocr_on_image_manga(img_cv):
    backend = settings["MANGA_OCR_BACKEND"]
    if backend == "manga_ocr":
        model = ensure_manga_ocr()
        pil_img = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
        return model(pil_img)
    elif backend == "easyocr":
        reader = ensure_easyocr()
        result = reader.readtext(img_cv, paragraph=True)
        if result:
            return " ".join([item[1] for item in result])
        return ""
    elif backend == "multi":
        lang_code = settings["OCR_LANG"]
        return multi_ocr_recognize(img_cv, lang_code, engines=settings["OCR_ENGINES"], strategy=settings["OCR_STRATEGY"])
    else:
        return ""

def ocr_on_image_text(img_cv):
    backend = settings["TEXT_OCR_BACKEND"]
    
    if backend == "multi":
        print("⚠️ [Текстовый режим] Multi-OCR отключён, так как не поддерживает поблочное распознавание. Используется EasyOCR.")
        backend = "easyocr"
    
    if backend == "easyocr":
        return ocr_with_easyocr_blocks(img_cv, settings["OCR_LANG"])
    elif backend == "tesseract":
        if not TESSERACT_AVAILABLE:
            print("❌ Tesseract недоступен, переключаем на EasyOCR")
            return ocr_with_easyocr_blocks(img_cv, settings["OCR_LANG"])
        return ocr_with_tesseract_blocks(img_cv, settings["OCR_LANG"])
    elif backend == "windows_ocr":
        if not WINDOWS_OCR_AVAILABLE or sys.platform != "win32":
            print("❌ Windows OCR недоступен, переключаем на EasyOCR")
            return ocr_with_easyocr_blocks(img_cv, settings["OCR_LANG"])
        return ocr_with_windows_ocr_blocks(img_cv, settings["OCR_LANG"])
    else:
        print(f"❌ Неизвестный бэкенд {backend}, используем EasyOCR")
        return ocr_with_easyocr_blocks(img_cv, settings["OCR_LANG"])

def detect_bubbles(yolo_model, image_np):
    device = 'cuda' if is_gpu_available_for_ocr() else 'cpu'
    results = yolo_model(image_np, conf=settings["CONFIDENCE_THRESHOLD"], iou=0.3, imgsz=640, verbose=False, device=device)
    boxes = []
    if results and results[0].boxes is not None:
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            w, h = x2 - x1, y2 - y1
            if w * h > 100:
                pad = 10
                x1 = max(0, x1 - pad)
                y1 = max(0, y1 - pad)
                w = min(image_np.shape[1] - x1, w + 2*pad)
                h = min(image_np.shape[0] - y1, h + 2*pad)
                boxes.append((x1, y1, w, h))
    boxes.sort(key=lambda b: (b[1], b[0]))
    return boxes

# ----------------------------------------------------------------------
# Функция перевода с поддержкой контекста (для синхронного и асинхронного режимов)
# ----------------------------------------------------------------------
def get_translation(text, use_context=True):
    """Выполняет перевод текста через LM Studio.
       use_context=False отключает использование истории и не добавляет перевод в историю (для параллельных запросов)."""
    if not text or len(text.strip()) < 2:
        return ""
    src = settings["SOURCE_LANG"]
    tgt = settings["TARGET_LANG"]
    messages = [{"role": "system", "content": (
        f"You are a professional {src}-to-{tgt} translator. "
        f"Respond only with the {tgt} translation. Do NOT provide explanations."
    )}]
    if use_context:
        with history_lock:
            recent = translation_history[-settings["CONTEXT_SIZE"]:] if translation_history else []
            if recent:
                ctx = "\n".join([f"Original ({src}): {o}\n{tgt}: {r}" for o, r in recent])
                messages.append({"role": "user", "content": f"Context:\n{ctx}"})
    messages.append({"role": "user", "content": f"Translate from {src} to {tgt}:\n{text}"})

    try:
        resp = requests.post(f"{settings['LM_STUDIO_URL']}/chat/completions", json={
            "model": settings["MODEL_NAME"], "messages": messages, "temperature": settings["TEMPERATURE"]
        }, timeout=settings["TIMEOUT"])
        if resp.status_code != 200:
            return "[Ошибка перевода]"
        data = resp.json()
        trans = data['choices'][0]['message']['content'].strip()
        if trans and trans != "[Ошибка перевода]":
            if use_context:
                with history_lock:
                    translation_history.append((text, trans))
                    if len(translation_history) > 100:
                        translation_history.pop(0)
        return trans
    except Exception:
        return "[Ошибка перевода]"

# ----------------------------------------------------------------------
# Смещение пересекающихся прямоугольников (упрощённый надёжный алгоритм)
# ----------------------------------------------------------------------
def rects_overlap(r1, r2):
    x1, y1, w1, h1 = r1
    x2, y2, w2, h2 = r2
    return not (x1 + w1 <= x2 or x2 + w2 <= x1 or y1 + h1 <= y2 or y2 + h2 <= y1)

# ----------------------------------------------------------------------
# Функция для вычисления размера блока по тексту
# ----------------------------------------------------------------------
def get_text_block_size(text, font):
    metrics = QFontMetrics(font)
    width_factor = settings.get("TEXT_BOX_WIDTH_FACTOR", 1.5)
    height_factor = settings.get("TEXT_BOX_HEIGHT_FACTOR", 1.5)
    text_pad_h = int(10 * width_factor)
    text_pad_v = int(10 * height_factor)
    
    lines = text.split('\n')
    if not lines:
        lines = [text]
    max_line_width = max(metrics.horizontalAdvance(line) for line in lines)
    required_width = int(max_line_width * width_factor) + 2 * text_pad_h
    required_height = int(metrics.height() * len(lines) * height_factor) + 2 * text_pad_v
    return required_width, required_height

# ----------------------------------------------------------------------
# Разрешение пересечений с учётом реальных размеров
# ----------------------------------------------------------------------
def resolve_overlaps(overlays, font, step=5, max_shift=150):
    if not overlays:
        return overlays
    print(f"[OVL] Разрешение пересечений для {len(overlays)} блоков...")
    start_time = time.time()
    
    blocks = []
    for (x, y, w, h, text) in overlays:
        final_w, final_h = get_text_block_size(text, font)
        center_x = x + w // 2
        center_y = y + h // 2
        new_x = center_x - final_w // 2
        new_y = center_y - final_h // 2
        blocks.append([new_x, new_y, final_w, final_h, text])
    
    result = [blocks[0]]
    for current in blocks[1:]:
        x, y, w, h, text = current
        found_place = False
        for shift in range(step, max_shift + 1, step):
            for dy in (shift, -shift, 0):
                for dx in (0, shift, -shift):
                    if dx == 0 and dy == 0:
                        continue
                    new_x = x + dx
                    new_y = y + dy
                    overlap = False
                    for (ox, oy, ow, oh, _) in result:
                        if not (new_x + w <= ox or ox + ow <= new_x or new_y + h <= oy or oy + oh <= new_y):
                            overlap = True
                            break
                    if not overlap:
                        result.append([new_x, new_y, w, h, text])
                        found_place = True
                        break
                if found_place:
                    break
            if found_place:
                break
        if not found_place:
            result.append([x, y, w, h, text])
    
    result = [tuple(b) for b in result]
    print(f"[OVL] Разрешение пересечений завершено за {time.time()-start_time:.3f} сек")
    return result

# ----------------------------------------------------------------------
# Группировка оверлеев
# ----------------------------------------------------------------------
def group_overlays(overlays):
    if not overlays:
        return []
    print(f"[OVL] Группировка {len(overlays)} блоков...")
    start = time.time()
    rects = [(x, y, x+w, y+h, text) for (x, y, w, h, text) in overlays]
    n = len(rects)
    parent = list(range(n))
    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u
    def union(u, v):
        ru, rv = find(u), find(v)
        if ru != rv:
            parent[rv] = ru

    for i in range(n):
        x1i, y1i, x2i, y2i, _ = rects[i]
        for j in range(i+1, n):
            x1j, y1j, x2j, y2j, _ = rects[j]
            if not (x2i <= x1j or x2j <= x1i or y2i <= y1j or y2j <= y1i):
                union(i, j)

    groups = {}
    for i in range(n):
        root = find(i)
        groups.setdefault(root, []).append(rects[i])

    result = []
    for grp in groups.values():
        group_items = []
        for (x1, y1, x2, y2, text) in grp:
            group_items.append((x1, y1, x2-x1, y2-y1, text))
        result.append(group_items)
    print(f"[OVL] Группировка завершена: {len(result)} групп, время {time.time()-start:.3f} сек")
    return result

# ----------------------------------------------------------------------
# CompositeOverlay
# ----------------------------------------------------------------------
class CompositeOverlay(QWidget):
    def __init__(self, group_items):
        super().__init__()
        self.items = group_items
        font_family = settings.get("FONT_FAMILY", "Arial")
        font_size = settings.get("FONT_SIZE", 12)
        self.font = QFont(font_family, font_size)
        
        base_x = min(item[0] for item in group_items)
        base_y = min(item[1] for item in group_items)
        
        self.local_rects = [(x - base_x, y - base_y, w, h, text) for (x, y, w, h, text) in group_items]
        
        self.merged_path = QPainterPath()
        for rx, ry, rw, rh, _ in self.local_rects:
            rect_path = QPainterPath()
            rect_path.addRect(rx, ry, rw, rh)
            self.merged_path = self.merged_path.united(rect_path)
        
        max_x = max(rx + rw for rx, ry, rw, rh, _ in self.local_rects)
        max_y = max(ry + rh for rx, ry, rw, rh, _ in self.local_rects)
        self.setGeometry(base_x, base_y, max_x, max_y)
        
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.show()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillPath(self.merged_path, QBrush(QColor(0, 0, 0, 200)))
        painter.setPen(QPen(QColor(255, 0, 127), 2))
        painter.drawPath(self.merged_path)
        painter.setFont(self.font)
        painter.setPen(QPen(QColor(255, 255, 255)))
        
        width_factor = settings.get("TEXT_BOX_WIDTH_FACTOR", 1.5)
        height_factor = settings.get("TEXT_BOX_HEIGHT_FACTOR", 1.5)
        text_pad_h = int(10 * width_factor)
        text_pad_v = int(10 * height_factor)
        
        for rx, ry, rw, rh, text in self.local_rects:
            if not text:
                continue
            text_rect = QRect(rx + text_pad_h//2, ry + text_pad_v//2, rw - text_pad_h, rh - text_pad_v)
            painter.drawText(text_rect, Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap, text)

class CaptureFrame(QWidget):
    def __init__(self, x, y, w, h, parent_app):
        super().__init__()
        self.parent_app = parent_app
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setGeometry(x, y, w, h)
        
        self.drag_pos = None
        self.resizing = False
        self.resize_edge = None
        self.resize_margin = 10
        
        self.btn_translate = QPushButton("OCR", self)
        self.btn_auto = QPushButton("Авто", self)
        self.btn_clear = QPushButton("Очистить", self)
        self.btn_close = QPushButton("✕", self)
        
        for btn in (self.btn_translate, self.btn_auto, self.btn_clear, self.btn_close):
            btn.setStyleSheet("""
                QPushButton {
                    background-color: rgba(30,30,30,200);
                    color: white;
                    border: 1px solid #ff007f;
                    border-radius: 4px;
                    padding: 4px 8px;
                    font-size: 11px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #ff007f;
                    color: black;
                }
            """)
        
        self.btn_translate.clicked.connect(self.parent_app.manual_translate)
        self.btn_auto.clicked.connect(self.parent_app.toggle_auto_mode)
        self.btn_clear.clicked.connect(clear_overlays)
        self.btn_close.clicked.connect(self.close_frame)
        
        self.update_button_positions()
        self.resize_timer = QTimer()
        self.resize_timer.setSingleShot(True)
        self.resize_timer.timeout.connect(self.update_button_positions)
        
    def update_button_positions(self):
        w = self.width()
        h = self.height()
        btn_w, btn_h = 55, 26
        spacing = 5
        total_w = btn_w * 4 + spacing * 3
        start_x = w - total_w - 10
        y_pos = h - btn_h - 10
        self.btn_translate.setGeometry(start_x, y_pos, btn_w, btn_h)
        self.btn_auto.setGeometry(start_x + btn_w + spacing, y_pos, btn_w, btn_h)
        self.btn_clear.setGeometry(start_x + (btn_w+spacing)*2, y_pos, btn_w, btn_h)
        self.btn_close.setGeometry(start_x + (btn_w+spacing)*3, y_pos, btn_w, btn_h)
        
    def resizeEvent(self, event):
        self.resize_timer.start(50)
        super().resizeEvent(event)
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setPen(QPen(QColor(255,0,127), 3))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(2, 2, self.width()-4, self.height()-4)
        
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            pos = event.position().toPoint()
            w, h = self.width(), self.height()
            if pos.x() >= w - self.resize_margin and pos.y() >= h - self.resize_margin:
                self.resizing, self.resize_edge = True, 'bottom-right'
            elif pos.x() <= self.resize_margin and pos.y() <= self.resize_margin:
                self.resizing, self.resize_edge = True, 'top-left'
            elif pos.x() >= w - self.resize_margin:
                self.resizing, self.resize_edge = True, 'right'
            elif pos.y() >= h - self.resize_margin:
                self.resizing, self.resize_edge = True, 'bottom'
            elif pos.x() <= self.resize_margin:
                self.resizing, self.resize_edge = True, 'left'
            elif pos.y() <= self.resize_margin:
                self.resizing, self.resize_edge = True, 'top'
            else:
                self.drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()
            
    def mouseMoveEvent(self, event):
        if self.resizing:
            gpos = event.globalPosition().toPoint()
            rect = self.geometry()
            if self.resize_edge == 'right':
                rect.setWidth(max(100, gpos.x() - rect.x()))
            elif self.resize_edge == 'bottom':
                rect.setHeight(max(100, gpos.y() - rect.y()))
            elif self.resize_edge == 'bottom-right':
                rect.setSize(QSize(max(100, gpos.x() - rect.x()), max(100, gpos.y() - rect.y())))
            elif self.resize_edge == 'left':
                delta = rect.x() - gpos.x()
                if delta < rect.width() - 100:
                    rect.setX(gpos.x())
                    rect.setWidth(rect.width() + delta)
            elif self.resize_edge == 'top':
                delta = rect.y() - gpos.y()
                if delta < rect.height() - 100:
                    rect.setY(gpos.y())
                    rect.setHeight(rect.height() + delta)
            elif self.resize_edge == 'top-left':
                dx, dy = rect.x() - gpos.x(), rect.y() - gpos.y()
                if dx < rect.width() - 100:
                    rect.setX(gpos.x())
                    rect.setWidth(rect.width() + dx)
                if dy < rect.height() - 100:
                    rect.setY(gpos.y())
                    rect.setHeight(rect.height() + dy)
            self.setGeometry(rect)
            self.parent_app.update_region(rect.x(), rect.y(), rect.width(), rect.height())
        elif self.drag_pos:
            self.move(event.globalPosition().toPoint() - self.drag_pos)
            g = self.geometry()
            self.parent_app.update_region(g.x(), g.y(), g.width(), g.height())
        event.accept()
        
    def mouseReleaseEvent(self, event):
        self.resizing = False
        self.drag_pos = None
        event.accept()
        
    def close_frame(self):
        clear_overlays()
        self.parent_app.reset_frame()
        self.close()

class RegionSelector(QWidget):
    def __init__(self, parent_app):
        super().__init__()
        self.parent_app = parent_app
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setGeometry(100,100,400,300)
        self.drag_pos = None
        self.resizing = False
        self.resize_edge = None
        self.resize_margin = 10
        self.btn = QPushButton("✓ Зафиксировать область", self)
        self.btn.setGeometry(10, self.height()-40, 150, 30)
        self.btn.setStyleSheet("background:#2d2d2d; color:white; border:1px solid #ff007f; border-radius:4px;")
        self.btn.clicked.connect(self.accept_region)
        self.label = QLabel("Перетащите и измените размер рамки\nЗатем нажмите кнопку", self)
        self.label.setGeometry(10,10,380,50)
        self.label.setStyleSheet("color:white; background:rgba(0,0,0,0.6); padding:4px; border-radius:4px;")
        self.label.setWordWrap(True)
        self.resize_timer = QTimer()
        self.resize_timer.setSingleShot(True)
        self.resize_timer.timeout.connect(self.update_button_position)
        
    def update_button_position(self):
        self.btn.move(10, self.height()-40)
        self.label.resize(self.width()-20, self.label.height())
        
    def resizeEvent(self, event):
        self.resize_timer.start(50)
        super().resizeEvent(event)
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setBrush(QBrush(QColor(0,0,0,80)))
        painter.setPen(QPen(QColor(255,0,127), 3))
        painter.drawRect(0,0,self.width()-1,self.height()-1)
        
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            pos = event.position().toPoint()
            w, h = self.width(), self.height()
            if pos.x() >= w - self.resize_margin and pos.y() >= h - self.resize_margin:
                self.resizing, self.resize_edge = True, 'bottom-right'
            elif pos.x() <= self.resize_margin and pos.y() <= self.resize_margin:
                self.resizing, self.resize_edge = True, 'top-left'
            elif pos.x() >= w - self.resize_margin:
                self.resizing, self.resize_edge = True, 'right'
            elif pos.y() >= h - self.resize_margin:
                self.resizing, self.resize_edge = True, 'bottom'
            elif pos.x() <= self.resize_margin:
                self.resizing, self.resize_edge = True, 'left'
            elif pos.y() <= self.resize_margin:
                self.resizing, self.resize_edge = True, 'top'
            else:
                self.drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()
            
    def mouseMoveEvent(self, event):
        if self.resizing:
            gpos = event.globalPosition().toPoint()
            rect = self.geometry()
            if self.resize_edge == 'right':
                rect.setWidth(max(100, gpos.x() - rect.x()))
            elif self.resize_edge == 'bottom':
                rect.setHeight(max(100, gpos.y() - rect.y()))
            elif self.resize_edge == 'bottom-right':
                rect.setSize(QSize(max(100, gpos.x() - rect.x()), max(100, gpos.y() - rect.y())))
            elif self.resize_edge == 'left':
                delta = rect.x() - gpos.x()
                if delta < rect.width() - 100:
                    rect.setX(gpos.x())
                    rect.setWidth(rect.width() + delta)
            elif self.resize_edge == 'top':
                delta = rect.y() - gpos.y()
                if delta < rect.height() - 100:
                    rect.setY(gpos.y())
                    rect.setHeight(rect.height() + delta)
            elif self.resize_edge == 'top-left':
                dx, dy = rect.x() - gpos.x(), rect.y() - gpos.y()
                if dx < rect.width() - 100:
                    rect.setX(gpos.x())
                    rect.setWidth(rect.width() + dx)
                if dy < rect.height() - 100:
                    rect.setY(gpos.y())
                    rect.setHeight(rect.height() + dy)
            self.setGeometry(rect)
        elif self.drag_pos:
            self.move(event.globalPosition().toPoint() - self.drag_pos)
        event.accept()
        
    def mouseReleaseEvent(self, event):
        self.resizing = False
        self.drag_pos = None
        event.accept()
        
    def accept_region(self):
        rect = self.geometry()
        self.parent_app.create_capture_frame(rect.x(), rect.y(), rect.width(), rect.height())
        self.close()

class TranslationThread(QThread):
    finished = pyqtSignal(list, float)
    progress = pyqtSignal(int, int)

    def __init__(self, region, mode):
        super().__init__()
        self.region = region
        self.mode = mode
        self.start_time = None

    def run(self):
        self.start_time = time.time()
        x, y, w, h = self.region
        print(f"\n📸 Скриншот ({w}x{h}), режим: {self.mode}")
        img = pyautogui.screenshot(region=(x, y, w, h))
        np_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        if self.mode == "manga":
            unload_mode_models("manga")
            yolo_model = ensure_yolo()
            blocks = detect_bubbles(yolo_model, np_img)
            total = len(blocks)
            self.progress.emit(0, total)
            # Сначала распознаём текст во всех баблах (последовательно)
            bubble_data = []  # (bx, by, bw, bh, original_text)
            for idx, (bx, by, bw, bh) in enumerate(blocks):
                crop = np_img[by:by+bh, bx:bx+bw]
                if crop.size == 0:
                    continue
                try:
                    orig = ocr_on_image_manga(crop)
                except Exception as e:
                    print(f"OCR ошибка: {e}")
                    continue
                if not orig or len(orig.strip()) < 2:
                    continue
                print(f"   📖 {orig}")
                bubble_data.append((x + bx, y + by, bw, bh, orig))
                self.progress.emit(idx + 1, total)

            if not bubble_data:
                self.finished.emit([], time.time() - self.start_time)
                return

            # Перевод
            if settings.get("ASYMMETRIC_TRANSLATION", False):
                # Параллельная обработка
                max_workers = settings.get("MAX_CONCURRENT_REQUESTS", 5)
                print(f"🔄 Асимметричный режим: параллельный перевод {len(bubble_data)} блоков (макс. {max_workers} запросов)")
                executor = ThreadPoolExecutor(max_workers=max_workers)
                futures = {}
                for i, (bx, by, bw, bh, orig) in enumerate(bubble_data):
                    future = executor.submit(get_translation, orig, False)  # без контекста
                    futures[future] = i
                overlays = [None] * len(bubble_data)
                completed = 0
                for future in as_completed(futures):
                    idx = futures[future]
                    trans = future.result()
                    bx, by, bw, bh, orig = bubble_data[idx]
                    if trans and trans != "[Ошибка перевода]":
                        overlays[idx] = (bx, by, bw, bh, trans)
                        print(f"   🌐 {trans}")
                    completed += 1
                    self.progress.emit(completed, len(bubble_data))
                # Убираем None (неудачные переводы)
                overlays = [ov for ov in overlays if ov is not None]
                elapsed = time.time() - self.start_time
                self.finished.emit(overlays, elapsed)
            else:
                # Синхронный режим с историей
                overlays = []
                total = len(bubble_data)
                for idx, (bx, by, bw, bh, orig) in enumerate(bubble_data):
                    trans = get_translation(orig, use_context=True)
                    if trans and trans != "[Ошибка перевода]":
                        overlays.append((bx, by, bw, bh, trans))
                        print(f"   🌐 {trans}")
                    self.progress.emit(idx + 1, total)
                elapsed = time.time() - self.start_time
                self.finished.emit(overlays, elapsed)

        else:  # text mode
            unload_mode_models("text")
            blocks = ocr_on_image_text(np_img)
            total = len(blocks)
            self.progress.emit(0, total)
            if not blocks:
                self.finished.emit([], time.time() - self.start_time)
                return

            # blocks: (x, y, w, h, text, conf)
            text_blocks = [(x + bx, y + by, bw, bh, text) for (bx, by, bw, bh, text, _) in blocks]

            if settings.get("ASYMMETRIC_TRANSLATION", False):
                max_workers = settings.get("MAX_CONCURRENT_REQUESTS", 5)
                print(f"🔄 Асимметричный режим: параллельный перевод {len(text_blocks)} блоков (макс. {max_workers} запросов)")
                executor = ThreadPoolExecutor(max_workers=max_workers)
                futures = {}
                for i, (bx, by, bw, bh, orig) in enumerate(text_blocks):
                    future = executor.submit(get_translation, orig, False)
                    futures[future] = i
                overlays = [None] * len(text_blocks)
                completed = 0
                for future in as_completed(futures):
                    idx = futures[future]
                    trans = future.result()
                    bx, by, bw, bh, orig = text_blocks[idx]
                    if trans and trans != "[Ошибка перевода]":
                        overlays[idx] = (bx, by, bw, bh, trans)
                        print(f"   🌐 {trans}")
                    completed += 1
                    self.progress.emit(completed, len(text_blocks))
                overlays = [ov for ov in overlays if ov is not None]
                elapsed = time.time() - self.start_time
                self.finished.emit(overlays, elapsed)
            else:
                overlays = []
                total = len(text_blocks)
                for idx, (bx, by, bw, bh, orig) in enumerate(text_blocks):
                    trans = get_translation(orig, use_context=True)
                    if trans and trans != "[Ошибка перевода]":
                        overlays.append((bx, by, bw, bh, trans))
                        print(f"   🌐 {trans}")
                    self.progress.emit(idx + 1, total)
                elapsed = time.time() - self.start_time
                self.finished.emit(overlays, elapsed)

class SettingsWindow(QMainWindow):
    def __init__(self, parent_app):
        super().__init__()
        self.parent_app = parent_app
        self.setWindowTitle("Настройки переводчика")
        self.setWindowFlags(Qt.WindowType.WindowStaysOnTopHint)
        self.setMinimumSize(720, 600)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setCentralWidget(scroll)
        
        central = QWidget()
        scroll.setWidget(central)
        layout = QVBoxLayout(central)
        layout.setSpacing(15)

        # LM Studio
        group1 = QGroupBox("LM Studio")
        form1 = QFormLayout()
        self.url_edit = QLineEdit(settings["LM_STUDIO_URL"])
        self.model_edit = QLineEdit(settings["MODEL_NAME"])
        form1.addRow("URL сервера:", self.url_edit)
        form1.addRow("Название модели:", self.model_edit)
        group1.setLayout(form1)
        layout.addWidget(group1)

        # Языки перевода
        group_lang = QGroupBox("Языки перевода")
        form_lang = QFormLayout()
        self.source_lang_combo = QComboBox()
        self.source_lang_combo.addItems(SOURCE_LANGUAGES)
        self.source_lang_combo.setCurrentText(settings["SOURCE_LANG"])
        self.target_lang_combo = QComboBox()
        self.target_lang_combo.addItems(["Russian", "English", "Ukrainian", "German", "French", "Spanish", "Chinese (Simplified)", "Chinese (Traditional)"])
        self.target_lang_combo.setCurrentText(settings["TARGET_LANG"])
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["manga", "text"])
        self.mode_combo.setCurrentText(settings["MODE"])
        form_lang.addRow("С какого языка:", self.source_lang_combo)
        form_lang.addRow("На какой язык:", self.target_lang_combo)
        form_lang.addRow("Режим (manga/text):", self.mode_combo)
        group_lang.setLayout(form_lang)
        layout.addWidget(group_lang)

        # OCR движки
        group_ocr = QGroupBox("OCR движки (поддержка нескольких одновременно)")
        form_ocr = QFormLayout()

        self.manga_ocr_combo = QComboBox()
        self.manga_ocr_combo.addItems(["manga_ocr", "easyocr", "multi"])
        self.manga_ocr_combo.setCurrentText(settings["MANGA_OCR_BACKEND"])

        self.text_ocr_combo = QComboBox()
        self.text_ocr_combo.addItems(["easyocr", "tesseract", "windows_ocr"])
        self.text_ocr_combo.setCurrentText(settings["TEXT_OCR_BACKEND"])

        self.easyocr_lang_combo = QComboBox()
        self.easyocr_lang_combo.setEditable(True)
        for name, code in EASYOCR_LANGUAGES.items():
            self.easyocr_lang_combo.addItem(f"{name} ({code})", code)
        self._set_combo_by_value(self.easyocr_lang_combo, settings["OCR_LANG"])

        self.check_easyocr = QCheckBox("EasyOCR")
        self.check_tesseract = QCheckBox("Tesseract")
        self.check_windows_ocr = QCheckBox("Windows OCR (только Win)")
        self.check_easyocr.setChecked("easyocr" in settings["OCR_ENGINES"])
        self.check_tesseract.setChecked("tesseract" in settings["OCR_ENGINES"])
        self.check_windows_ocr.setChecked("windows_ocr" in settings["OCR_ENGINES"])
        if not TESSERACT_AVAILABLE:
            self.check_tesseract.setEnabled(False)
            self.check_tesseract.setToolTip("pytesseract не установлен")
        if not WINDOWS_OCR_AVAILABLE or sys.platform != "win32":
            self.check_windows_ocr.setEnabled(False)
            self.check_windows_ocr.setToolTip("Windows OCR доступен только на Windows с winocr")

        engines_layout = QHBoxLayout()
        engines_layout.addWidget(self.check_easyocr)
        engines_layout.addWidget(self.check_tesseract)
        engines_layout.addWidget(self.check_windows_ocr)

        self.strategy_combo = QComboBox()
        self.strategy_combo.addItems(["best_confidence", "voting"])
        self.strategy_combo.setCurrentText(settings["OCR_STRATEGY"])

        self.easyocr_conf = QDoubleSpinBox()
        self.easyocr_conf.setRange(0.0, 0.9)
        self.easyocr_conf.setSingleStep(0.05)
        self.easyocr_conf.setValue(settings.get("EASYOCR_CONFIDENCE", 0.2))

        self.tesseract_conf = QDoubleSpinBox()
        self.tesseract_conf.setRange(0.0, 0.9)
        self.tesseract_conf.setSingleStep(0.05)
        self.tesseract_conf.setValue(settings.get("TESSERACT_CONFIDENCE", 0.3))

        self.windows_ocr_conf = QDoubleSpinBox()
        self.windows_ocr_conf.setRange(0.0, 0.9)
        self.windows_ocr_conf.setSingleStep(0.05)
        self.windows_ocr_conf.setValue(settings.get("WINDOWS_OCR_CONFIDENCE", 0.3))

        self.tesseract_path_edit = QLineEdit()
        self.tesseract_path_edit.setPlaceholderText("Например: C:/Program Files/Tesseract-OCR/tesseract.exe")
        self.tesseract_path_edit.setText(settings.get("TESSERACT_PATH", ""))
        self.tesseract_path_button = QPushButton("Обзор...")
        self.tesseract_path_button.clicked.connect(self.browse_tesseract)
        tesseract_path_layout = QHBoxLayout()
        tesseract_path_layout.addWidget(self.tesseract_path_edit)
        tesseract_path_layout.addWidget(self.tesseract_path_button)

        form_ocr.addRow("OCR для режима manga:", self.manga_ocr_combo)
        form_ocr.addRow("OCR для режима text:", self.text_ocr_combo)
        form_ocr.addRow("Язык EasyOCR:", self.easyocr_lang_combo)
        form_ocr.addRow("Активные движки (multi):", engines_layout)
        form_ocr.addRow("Стратегия выбора:", self.strategy_combo)
        form_ocr.addRow("Порог уверенности EasyOCR:", self.easyocr_conf)
        form_ocr.addRow("Порог уверенности Tesseract:", self.tesseract_conf)
        form_ocr.addRow("Порог уверенности Windows OCR:", self.windows_ocr_conf)
        form_ocr.addRow("Путь к tesseract.exe:", tesseract_path_layout)

        group_ocr.setLayout(form_ocr)
        layout.addWidget(group_ocr)

        warn_label = QLabel("⚠️ Multi-OCR (выбор движков и стратегия) работает ТОЛЬКО в режиме manga.\n"
                            "В текстовом режиме multi-OCR отключён автоматически, используется выбранный бэкенд (easyocr/tesseract/windows_ocr).\n"
                            "✅ Все три движка в текстовом режиме поддерживают поблочное распознавание.")
        warn_label.setStyleSheet("color: orange; font-size: 10px; background: rgba(0,0,0,0.5); padding: 4px;")
        warn_label.setWordWrap(True)
        layout.addWidget(warn_label)

        group2 = QGroupBox("Параметры перевода и оверлеев")
        form2 = QFormLayout()
        self.context_spin = QSpinBox()
        self.context_spin.setRange(1, 20)
        self.context_spin.setValue(settings["CONTEXT_SIZE"])
        self.temp_spin = QDoubleSpinBox()
        self.temp_spin.setRange(0.0, 2.0)
        self.temp_spin.setSingleStep(0.05)
        self.temp_spin.setValue(settings["TEMPERATURE"])
        self.timeout_spin = QSpinBox()
        self.timeout_spin.setRange(10, 120)
        self.timeout_spin.setValue(settings["TIMEOUT"])
        self.yolo_conf_spin = QDoubleSpinBox()
        self.yolo_conf_spin.setRange(0.1, 0.9)
        self.yolo_conf_spin.setSingleStep(0.05)
        self.yolo_conf_spin.setValue(settings["CONFIDENCE_THRESHOLD"])
        self.auto_interval_spin = QDoubleSpinBox()
        self.auto_interval_spin.setRange(0.5, 10.0)
        self.auto_interval_spin.setSingleStep(0.5)
        self.auto_interval_spin.setValue(settings["AUTO_CHECK_INTERVAL"])

        self.width_factor_spin = QDoubleSpinBox()
        self.width_factor_spin.setRange(0.5, 3.0)
        self.width_factor_spin.setSingleStep(0.1)
        self.width_factor_spin.setValue(settings.get("TEXT_BOX_WIDTH_FACTOR", 1.5))
        self.height_factor_spin = QDoubleSpinBox()
        self.height_factor_spin.setRange(0.5, 3.0)
        self.height_factor_spin.setSingleStep(0.1)
        self.height_factor_spin.setValue(settings.get("TEXT_BOX_HEIGHT_FACTOR", 1.5))

        form2.addRow("Размер контекста:", self.context_spin)
        form2.addRow("Температура:", self.temp_spin)
        form2.addRow("Таймаут (сек):", self.timeout_spin)
        form2.addRow("Порог уверенности YOLO:", self.yolo_conf_spin)
        form2.addRow("Интервал авто (сек):", self.auto_interval_spin)
        form2.addRow("Коэф. ширины текста:", self.width_factor_spin)
        form2.addRow("Коэф. высоты текста:", self.height_factor_spin)
        group2.setLayout(form2)
        layout.addWidget(group2)

        # ---- Новая группа: Асимметричная обработка ----
        group_async = QGroupBox("Асимметричная обработка (параллельные переводы)")
        form_async = QFormLayout()
        self.async_check = QCheckBox("Включить асимметричную обработку")
        self.async_check.setChecked(settings.get("ASYMMETRIC_TRANSLATION", False))
        self.async_check.setToolTip("Отправлять несколько блоков текста в LM Studio параллельно.\nКонтекст перевода при этом не используется.")
        self.max_concurrent_spin = QSpinBox()
        self.max_concurrent_spin.setRange(1, 20)
        self.max_concurrent_spin.setValue(settings.get("MAX_CONCURRENT_REQUESTS", 5))
        self.max_concurrent_spin.setToolTip("Максимальное количество одновременных запросов к LM Studio")
        form_async.addRow("", self.async_check)
        form_async.addRow("Макс. параллельных запросов:", self.max_concurrent_spin)
        group_async.setLayout(form_async)
        layout.addWidget(group_async)
        # ---------------------------------------------

        group_display = QGroupBox("Отображение")
        form_display = QFormLayout()
        self.font_combo = QFontComboBox()
        self.font_combo.setCurrentFont(QFont(settings.get("FONT_FAMILY", "Arial")))
        self.font_size_spin = QSpinBox()
        self.font_size_spin.setRange(8, 48)
        self.font_size_spin.setValue(settings.get("FONT_SIZE", 12))
        self.relocation_check = QCheckBox()
        self.relocation_check.setChecked(settings.get("ENABLE_OVERLAP_RELOCATION", True))
        form_display.addRow("Шрифт:", self.font_combo)
        form_display.addRow("Размер шрифта:", self.font_size_spin)
        form_display.addRow("Смещать пересекающиеся блоки:", self.relocation_check)
        group_display.setLayout(form_display)
        layout.addWidget(group_display)

        # --- Группа ускорения (раздельная настройка для OCR и модели) ---
        group_accel = QGroupBox("Ускорение")
        form_accel = QFormLayout()
        
        # Чекбокс для OCR
        self.gpu_ocr_checkbox = QCheckBox("Использовать GPU для OCR (YOLO, EasyOCR, MangaOCR)")
        self.gpu_ocr_checkbox.setChecked(settings.get("USE_GPU_OCR", False))
        self.gpu_ocr_checkbox.setEnabled(True)
        if not TORCH_AVAILABLE:
            self.gpu_ocr_checkbox.setToolTip("PyTorch не установлен, GPU не будет работать")
        elif not torch.cuda.is_available():
            self.gpu_ocr_checkbox.setToolTip("CUDA не обнаружена. Установите CUDA Toolkit и драйверы NVIDIA.")
        else:
            self.gpu_ocr_checkbox.setToolTip(f"GPU найден: {torch.cuda.get_device_name(0)}")
        
        # Чекбокс для модели – теперь активный, но с предупреждением
        self.gpu_model_checkbox = QCheckBox("Использовать GPU для модели (LM Studio)")
        self.gpu_model_checkbox.setChecked(settings.get("USE_GPU_MODEL", False))
        self.gpu_model_checkbox.setEnabled(True)
        self.gpu_model_checkbox.setToolTip("Ускорение модели настраивается в самой LM Studio.\nЭтот флаг только для информации и не влияет на работу программы.\nДля реального GPU перейдите в настройки LM Studio → GPU Offload.")
        self.gpu_model_checkbox.stateChanged.connect(self.on_gpu_model_toggled)
        
        form_accel.addRow("", self.gpu_ocr_checkbox)
        form_accel.addRow("", self.gpu_model_checkbox)
        group_accel.setLayout(form_accel)
        layout.addWidget(group_accel)
        
        info_accel = QLabel("ℹ️ Для модели (перевода текста) GPU управляется через LM Studio. Откройте LM Studio → Настройки → GPU Offload.\n"
                            "Для OCR можно отдельно включить GPU — это ускорит распознавание текста и детекцию баблов.\n"
                            "Асимметричная обработка ускоряет перевод за счёт параллельных запросов, но отключает контекст перевода.")
        info_accel.setStyleSheet("color: #aaa; font-size: 10px; background: rgba(0,0,0,0.3); padding: 4px;")
        info_accel.setWordWrap(True)
        layout.addWidget(info_accel)

        btn_layout = QHBoxLayout()
        save_btn = QPushButton("Сохранить")
        cancel_btn = QPushButton("Отмена")
        save_btn.clicked.connect(self.save_settings)
        cancel_btn.clicked.connect(self.hide)
        btn_layout.addWidget(save_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)

        info = QLabel("📌 Для китайского языка рекомендуется EasyOCR или Windows OCR. Tesseract может давать низкое качество.\n"
                      "📌 Укажите путь к tesseract.exe, если он не в PATH. Языковые файлы Tesseract должны быть в папке tessdata рядом с tesseract.exe.\n"
                      "📌 Включение GPU для OCR ускоряет работу YOLO, EasyOCR и Manga OCR. Требуется NVIDIA GPU и установленный CUDA Toolkit.")
        info.setStyleSheet("color: orange; font-size: 10px;")
        layout.addWidget(info)

        self.source_lang_combo.currentTextChanged.connect(self.on_source_lang_changed)

    def on_gpu_model_toggled(self, state):
        checked = (state == Qt.CheckState.Checked.value)
        if checked:
            msg = "Эта настройка не управляет LM Studio. Чтобы использовать GPU для модели,\nоткройте LM Studio → Настройки → GPU Offload и установите нужные слои."
        else:
            msg = "Отключение этого флага не отключает GPU в LM Studio.\nУправление GPU для модели происходит внутри LM Studio."
        QMessageBox.information(self, "Информация об ускорении модели", msg)
        settings["USE_GPU_MODEL"] = (state == Qt.CheckState.Checked.value)

    def _set_combo_by_value(self, combo, value):
        index = combo.findData(value)
        if index >= 0:
            combo.setCurrentIndex(index)
        else:
            combo.setEditText(value)

    def browse_tesseract(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Выберите tesseract.exe", "", "Executable (*.exe)")
        if file_path:
            self.tesseract_path_edit.setText(file_path)

    def on_source_lang_changed(self, new_source):
        reply = QMessageBox.question(
            self, "Обновить язык OCR?",
            f"Вы изменили исходный язык на '{new_source}'.\n"
            "Обновить язык EasyOCR в соответствии с выбранным языком?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            easy_rec = SOURCE_TO_EASYOCR.get(new_source, "en")
            self._set_combo_by_value(self.easyocr_lang_combo, easy_rec)
            QMessageBox.information(self, "Готово", f"Язык EasyOCR установлен на: {easy_rec}")

    def save_settings(self):
        global settings
        old_mode = settings["MODE"]
        old_manga_ocr = settings["MANGA_OCR_BACKEND"]
        old_text_ocr = settings["TEXT_OCR_BACKEND"]
        old_ocr_lang = settings["OCR_LANG"]
        old_gpu_ocr = settings.get("USE_GPU_OCR", False)

        new_ocr_lang = self.easyocr_lang_combo.currentData()
        if new_ocr_lang is None:
            new_ocr_lang = self.easyocr_lang_combo.currentText().strip()

        settings["LM_STUDIO_URL"] = self.url_edit.text().strip()
        settings["MODEL_NAME"] = self.model_edit.text().strip()
        settings["CONTEXT_SIZE"] = self.context_spin.value()
        settings["TEMPERATURE"] = self.temp_spin.value()
        settings["TIMEOUT"] = self.timeout_spin.value()
        settings["CONFIDENCE_THRESHOLD"] = self.yolo_conf_spin.value()
        settings["AUTO_CHECK_INTERVAL"] = self.auto_interval_spin.value()
        settings["SOURCE_LANG"] = self.source_lang_combo.currentText()
        settings["TARGET_LANG"] = self.target_lang_combo.currentText()
        settings["MODE"] = self.mode_combo.currentText()
        settings["MANGA_OCR_BACKEND"] = self.manga_ocr_combo.currentText()
        settings["TEXT_OCR_BACKEND"] = self.text_ocr_combo.currentText()
        settings["OCR_LANG"] = new_ocr_lang
        settings["EASYOCR_CONFIDENCE"] = self.easyocr_conf.value()
        settings["TESSERACT_CONFIDENCE"] = self.tesseract_conf.value()
        settings["WINDOWS_OCR_CONFIDENCE"] = self.windows_ocr_conf.value()
        settings["TEXT_BOX_WIDTH_FACTOR"] = self.width_factor_spin.value()
        settings["TEXT_BOX_HEIGHT_FACTOR"] = self.height_factor_spin.value()
        settings["ENABLE_OVERLAP_RELOCATION"] = self.relocation_check.isChecked()
        settings["FONT_FAMILY"] = self.font_combo.currentFont().family()
        settings["FONT_SIZE"] = self.font_size_spin.value()
        settings["OCR_STRATEGY"] = self.strategy_combo.currentText()
        settings["TESSERACT_PATH"] = self.tesseract_path_edit.text().strip()
        settings["USE_GPU_OCR"] = self.gpu_ocr_checkbox.isChecked()
        settings["USE_GPU_MODEL"] = self.gpu_model_checkbox.isChecked()
        # Новые настройки
        settings["ASYMMETRIC_TRANSLATION"] = self.async_check.isChecked()
        settings["MAX_CONCURRENT_REQUESTS"] = self.max_concurrent_spin.value()

        engines = []
        if self.check_easyocr.isChecked():
            engines.append("easyocr")
        if self.check_tesseract.isChecked():
            engines.append("tesseract")
        if self.check_windows_ocr.isChecked():
            engines.append("windows_ocr")
        settings["OCR_ENGINES"] = engines

        if settings["USE_GPU_OCR"] and (not TORCH_AVAILABLE or not torch.cuda.is_available()):
            QMessageBox.warning(self, "GPU недоступен", 
                "Вы включили использование GPU для OCR, но CUDA не обнаружена.\n"
                "Убедитесь, что установлен PyTorch с поддержкой CUDA и драйверы NVIDIA.\n"
                "Будет использован CPU.")
            settings["USE_GPU_OCR"] = False

        need_unload = False
        if old_gpu_ocr != settings["USE_GPU_OCR"]:
            need_unload = True
            msg_gpu = f"GPU для OCR {'включён' if settings['USE_GPU_OCR'] else 'выключен'}. Модели будут перезагружены."
        else:
            msg_gpu = ""

        if (old_manga_ocr != settings["MANGA_OCR_BACKEND"] or
            old_text_ocr != settings["TEXT_OCR_BACKEND"] or
            old_ocr_lang != settings["OCR_LANG"]):
            need_unload = True
            msg_ocr = "Настройки OCR изменены. Модели перезагрузятся при следующем переводе."
        else:
            msg_ocr = ""

        if need_unload:
            unload_all_ocr()
            unload_mode_models(settings["MODE"])

        if old_mode != settings["MODE"]:
            unload_mode_models(settings["MODE"])

        save_settings_to_file()
        full_msg = "\n".join([msg_gpu, msg_ocr]).strip()
        QMessageBox.information(self, "Сохранено", full_msg or "Настройки сохранены.")
        self.hide()

active_overlays = []

def clear_overlays():
    global active_overlays
    for ov in active_overlays:
        ov.close()
    active_overlays.clear()
    print("🧹 Оверлеи скрыты")

class TranslatorApp(QApplication):
    def __init__(self, argv):
        super().__init__(argv)
        self.setQuitOnLastWindowClosed(False)

        self.capture_frame = None
        self.selector = None
        self.translation_thread = None
        self.auto_mode = False
        self.auto_timer = None
        self.last_hash = None
        self.current_region = None
        self.last_translation_time = None

        self.diagnose_ocr()

        # Иконка трея
        tray_icon_pixmap = QPixmap(64, 64)
        tray_icon_pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(tray_icon_pixmap)
        painter.setBrush(QBrush(QColor(255, 0, 127)))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(0, 0, 64, 64, 10, 10)
        painter.setPen(QPen(Qt.GlobalColor.white, 3))
        painter.setFont(QFont("Arial", 32, QFont.Weight.Bold))
        painter.drawText(0, 0, 64, 64, Qt.AlignmentFlag.AlignCenter, "M")
        painter.end()
        tray_icon = QIcon(tray_icon_pixmap)

        self.tray_icon = QSystemTrayIcon(self)
        self.tray_icon.setIcon(tray_icon)
        self.tray_icon.setToolTip("Переводчик манги / текста (Multi-OCR)")
        tray_menu = QMenu()
        show_action = QAction("Показать панель", self)
        settings_action = QAction("Настройки", self)
        exit_action = QAction("Выход", self)
        show_action.triggered.connect(self.show_main_panel)
        settings_action.triggered.connect(self.show_settings)
        exit_action.triggered.connect(self.exit_app)
        tray_menu.addAction(show_action)
        tray_menu.addAction(settings_action)
        tray_menu.addSeparator()
        tray_menu.addAction(exit_action)
        self.tray_icon.setContextMenu(tray_menu)
        self.tray_icon.show()

        # Главная панель
        self.main_panel = QWidget()
        self.main_panel.setWindowFlags(Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.FramelessWindowHint)
        self.main_panel.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.main_panel.setGeometry(50, 50, 280, 200)
        self.main_panel.setStyleSheet("background-color: rgba(20,20,20,200); border-radius: 10px; border: 1px solid #ff007f;")

        layout = QVBoxLayout(self.main_panel)
        layout.setSpacing(5)
        btn_select = QPushButton("🎯 Выбрать область")
        btn_settings = QPushButton("⚙ Настройки")
        btn_tray = QPushButton("▼ Трей")
        btn_exit = QPushButton("🚪 Выход")
        for btn in (btn_select, btn_settings, btn_tray, btn_exit):
            btn.setStyleSheet("""
                QPushButton {
                    background: #2d2d2d;
                    color: white;
                    border: 1px solid #555;
                    border-radius: 5px;
                    padding: 5px;
                }
                QPushButton:hover {
                    background: #ff007f;
                    border-color: #ff007f;
                }
            """)
        btn_select.clicked.connect(self.start_region_selection)
        btn_settings.clicked.connect(self.show_settings)
        btn_tray.clicked.connect(self.hide_main_panel)
        btn_exit.clicked.connect(self.exit_app)
        layout.addWidget(btn_select)
        layout.addWidget(btn_settings)
        layout.addWidget(btn_tray)
        layout.addWidget(btn_exit)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #ff007f;
                border-radius: 5px;
                text-align: center;
                color: white;
                background-color: #2d2d2d;
                height: 20px;
            }
            QProgressBar::chunk {
                background-color: #ff007f;
                border-radius: 4px;
            }
        """)
        layout.addWidget(self.progress_bar)

        self.time_label = QLabel("⏱️ Время последнего перевода: --")
        self.time_label.setStyleSheet("color: #ccc; font-size: 10px; background: rgba(0,0,0,0.5); padding: 2px; border-radius: 3px;")
        self.time_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.time_label)

        self.main_panel.show()
        self.settings_window = None

        print("\n=== 🌀 ПЕРЕВОДЧИК ЗАПУЩЕН (Multi-OCR только для manga, все движки поддерживают блоки) 🌀 ===")
        print("Значок в трее: правый клик для меню")
        print("Нажмите 'Выбрать область' и настройте рамку")
        print("Асимметричная обработка: параллельные запросы в LM Studio (отключает контекст)")
        print("========================================\n")

    def diagnose_ocr(self):
        print("\n🔍 ДИАГНОСТИКА OCR:")
        print(f"   TEXT_OCR_BACKEND = {settings['TEXT_OCR_BACKEND']}")
        print(f"   MANGA_OCR_BACKEND = {settings['MANGA_OCR_BACKEND']}")
        print(f"   OCR_LANG = {settings['OCR_LANG']}")
        print(f"   OCR_STRATEGY = {settings['OCR_STRATEGY']}")
        print(f"   OCR_ENGINES = {settings['OCR_ENGINES']}")
        print(f"   Порог EasyOCR: {settings.get('EASYOCR_CONFIDENCE', 0.2)}")
        print(f"   Порог Tesseract: {settings.get('TESSERACT_CONFIDENCE', 0.3)}")
        print(f"   Порог Windows OCR: {settings.get('WINDOWS_OCR_CONFIDENCE', 0.3)}")
        print(f"   Tesseract путь: {settings.get('TESSERACT_PATH', 'не задан')}")
        print(f"   USE_GPU_OCR: {settings.get('USE_GPU_OCR', False)}")
        print(f"   USE_GPU_MODEL: {settings.get('USE_GPU_MODEL', False)}")
        print(f"   ASYMMETRIC_TRANSLATION: {settings.get('ASYMMETRIC_TRANSLATION', False)}")
        print(f"   MAX_CONCURRENT_REQUESTS: {settings.get('MAX_CONCURRENT_REQUESTS', 5)}")
        if TORCH_AVAILABLE:
            print(f"   CUDA доступна: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"   GPU: {torch.cuda.get_device_name(0)}")

        if TESSERACT_AVAILABLE:
            if set_tesseract_path():
                try:
                    ver = pytesseract.get_tesseract_version()
                    print(f"   ✅ Tesseract доступен, версия {ver}")
                except:
                    print("   ⚠️ Tesseract установлен, но не удалось получить версию")
            else:
                print("   ❌ Tesseract не найден. Укажите путь в настройках или установите tesseract.exe в PATH")
        else:
            print("   ❌ pytesseract не установлен (pip install pytesseract)")

        if WINDOWS_OCR_AVAILABLE and sys.platform == "win32":
            print("   ✅ Windows OCR (winocr) доступен, поддерживает координаты блоков")
        else:
            print("   ❌ Windows OCR недоступен (требуется winocr и Windows 10/11)")

        print("----------------------------------------\n")

    def show_main_panel(self):
        self.main_panel.show()
        self.main_panel.raise_()

    def hide_main_panel(self):
        self.main_panel.hide()

    def show_settings(self):
        if self.settings_window is None:
            self.settings_window = SettingsWindow(self)
        self.settings_window.show()
        self.settings_window.raise_()

    def start_region_selection(self):
        if self.capture_frame:
            self.capture_frame.close()
            self.capture_frame = None
        if self.selector:
            self.selector.close()
        self.selector = RegionSelector(self)
        self.selector.show()

    def create_capture_frame(self, x, y, w, h):
        self.current_region = (x, y, w, h)
        if self.capture_frame:
            self.capture_frame.close()
        self.capture_frame = CaptureFrame(x, y, w, h, self)
        self.capture_frame.show()
        if self.auto_mode:
            self.toggle_auto_mode()
        print(f"📐 Рамка создана: {x},{y},{w},{h}")

    def update_region(self, x, y, w, h):
        self.current_region = (x, y, w, h)

    def reset_frame(self):
        self.capture_frame = None
        self.current_region = None
        if self.auto_mode:
            self.toggle_auto_mode()

    def manual_translate(self):
        if not self.current_region:
            print("❌ Сначала выберите область через кнопку на панели")
            return
        if self.translation_thread and self.translation_thread.isRunning():
            print("⚠️ Перевод уже выполняется, подождите...")
            return
        clear_overlays()
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        print("🚀 Перевод...")
        self.translation_thread = TranslationThread(self.current_region, settings["MODE"])
        self.translation_thread.progress.connect(self.update_progress)
        self.translation_thread.finished.connect(self.on_translation_finished)
        self.translation_thread.start()

    def update_progress(self, current, total):
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)

    def on_translation_finished(self, overlays_data, elapsed_seconds):
        if self.translation_thread:
            self.translation_thread.quit()
            self.translation_thread.wait()
            self.translation_thread = None

        self.last_translation_time = elapsed_seconds
        self.time_label.setText(f"⏱️ Время последнего перевода: {elapsed_seconds:.2f} сек")
        self.time_label.setStyleSheet("color: #ffcc00; font-size: 10px; background: rgba(0,0,0,0.5); padding: 2px; border-radius: 3px;")

        if not overlays_data:
            self.progress_bar.setVisible(False)
            return

        current_mode = settings["MODE"]

        if current_mode == "manga":
            self.progress_bar.setMaximum(0)
            self.progress_bar.setFormat("Группировка баблов...")
            print("[OVL] Режим манги: группировка пересекающихся баблов с возможностью расширения")

            groups = group_overlays(overlays_data)
            print(f"[OVL] Получено {len(groups)} групп баблов")

            if not groups:
                self.progress_bar.setVisible(False)
                return

            self.progress_bar.setMaximum(len(groups))
            self.progress_bar.setValue(0)
            self.progress_bar.setFormat("Создание оверлеев: %v/%m")

            for i, group in enumerate(groups):
                comp = CompositeOverlayMangaGroup(group, max_expand_w=150, max_expand_h=80)
                active_overlays.append(comp)
                self.progress_bar.setValue(i + 1)
                QApplication.processEvents()

            if self.capture_frame:
                self.capture_frame.raise_()
                for btn in (self.capture_frame.btn_translate, self.capture_frame.btn_auto,
                            self.capture_frame.btn_clear, self.capture_frame.btn_close):
                    btn.raise_()

            self.progress_bar.setVisible(False)
            print(f"[OVL] Создано {len(groups)} групп-оверлеев для манги\n")
            return

        # Текстовый режим
        self.progress_bar.setMaximum(0)
        self.progress_bar.setFormat("Отрисовка оверлеев...")
        print("[OVL] Текстовый режим: постобработка переводов...")

        temp_font = QFont(settings.get("FONT_FAMILY", "Arial"), settings.get("FONT_SIZE", 12))

        if settings.get("ENABLE_OVERLAP_RELOCATION", True):
            overlays_data = resolve_overlaps(overlays_data, temp_font)
        else:
            expanded = []
            for (x, y, w, h, text) in overlays_data:
                final_w, final_h = get_text_block_size(text, temp_font)
                center_x = x + w // 2
                center_y = y + h // 2
                new_x = center_x - final_w // 2
                new_y = center_y - final_h // 2
                expanded.append((new_x, new_y, final_w, final_h, text))
            overlays_data = expanded

        groups = group_overlays(overlays_data)
        print(f"[OVL] Получено {len(groups)} групп")

        if not groups:
            self.progress_bar.setVisible(False)
            return

        self.progress_bar.setMaximum(len(groups))
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Создание оверлеев: %v/%m")

        for i, group in enumerate(groups):
            comp = CompositeOverlay(group)
            active_overlays.append(comp)
            self.progress_bar.setValue(i + 1)
            QApplication.processEvents()

        if self.capture_frame:
            self.capture_frame.raise_()
            for btn in (self.capture_frame.btn_translate, self.capture_frame.btn_auto,
                        self.capture_frame.btn_clear, self.capture_frame.btn_close):
                btn.raise_()

        self.progress_bar.setVisible(False)
        print(f"[OVL] Создано {len(groups)} окон-оверлеев\n")

    def toggle_auto_mode(self):
        if not self.current_region:
            print("❌ Сначала выберите область")
            return
        self.auto_mode = not self.auto_mode
        if self.auto_mode:
            print("🔁 АВТОРЕЖИМ ВКЛЮЧЁН")
            self.last_hash = None
            if not self.auto_timer:
                self.auto_timer = QTimer()
                self.auto_timer.timeout.connect(self.auto_check)
            self.auto_timer.start(int(settings["AUTO_CHECK_INTERVAL"] * 1000))
            if self.capture_frame:
                self.capture_frame.btn_auto.setStyleSheet("background-color: #ff007f; color: black; border:1px solid #ff007f;")
        else:
            print("🔁 АВТОРЕЖИМ ВЫКЛЮЧЕН")
            if self.auto_timer:
                self.auto_timer.stop()
            if self.capture_frame:
                self.capture_frame.btn_auto.setStyleSheet("background-color: rgba(30,30,30,200); color: white; border:1px solid #ff007f;")

    def auto_check(self):
        if not self.current_region or not self.auto_mode:
            return
        x, y, w, h = self.current_region
        img = pyautogui.screenshot(region=(x, y, w, h))
        hsh = hash(img.tobytes())
        if self.last_hash is None:
            self.last_hash = hsh
            return
        if hsh != self.last_hash:
            print("🔄 Изменение кадра -> перевод")
            self.last_hash = hsh
            if self.translation_thread and self.translation_thread.isRunning():
                return
            clear_overlays()
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            self.translation_thread = TranslationThread(self.current_region, settings["MODE"])
            self.translation_thread.progress.connect(self.update_progress)
            self.translation_thread.finished.connect(self.on_translation_finished)
            self.translation_thread.start()

    def exit_app(self):
        clear_overlays()
        if self.capture_frame:
            self.capture_frame.close()
        if self.selector:
            self.selector.close()
        if self.settings_window:
            self.settings_window.close()
        self.main_panel.close()
        self.tray_icon.hide()
        self.quit()

class CompositeOverlayManga(QWidget):
    def __init__(self, x, y, w, h, text):
        super().__init__()
        self.text = text
        font_family = settings.get("FONT_FAMILY", "Arial")
        font_size = settings.get("FONT_SIZE", 12)
        self.font = QFont(font_family, font_size)
        self.setGeometry(x, y, w, h)
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.show()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QBrush(QColor(0, 0, 0, 200)))
        painter.setPen(QPen(QColor(255, 0, 127), 2))
        painter.drawRect(self.rect().adjusted(2, 2, -2, -2))
        painter.setFont(self.font)
        painter.setPen(QPen(QColor(255, 255, 255)))
        margin = 8
        text_rect = self.rect().adjusted(margin, margin, -margin, -margin)
        painter.drawText(text_rect, Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap, self.text)

class CompositeOverlayMangaGroup(QWidget):
    def __init__(self, group_items, max_expand_w=150, max_expand_h=80):
        super().__init__()
        self.group_items = group_items
        font_family = settings.get("FONT_FAMILY", "Arial")
        font_size = settings.get("FONT_SIZE", 12)
        self.font = QFont(font_family, font_size)
        metrics = QFontMetrics(self.font)
        
        width_factor = settings.get("TEXT_BOX_WIDTH_FACTOR", 1.5)
        height_factor = settings.get("TEXT_BOX_HEIGHT_FACTOR", 1.5)
        text_pad_h = int(10 * width_factor)
        text_pad_v = int(10 * height_factor)
        
        expanded_rects = []
        for (x, y, w, h, text) in group_items:
            if not text:
                expanded_rects.append((x, y, w, h, text))
                continue
            
            words = text.split()
            max_word_width = max(metrics.horizontalAdvance(word) for word in words) if words else 0
            required_width = max_word_width + 2 * text_pad_h
            
            new_w = max(w, min(required_width, w + max_expand_w))
            delta_w = new_w - w
            new_x = x - delta_w // 2
            
            lines = text.split('\n')
            if not lines:
                lines = [text]
            char_width = metrics.averageCharWidth()
            max_chars_per_line = max(1, int((new_w - 2*text_pad_h) / char_width))
            wrapped_lines = []
            for line in lines:
                words_line = line.split()
                current_line = []
                for word in words_line:
                    if metrics.horizontalAdvance(' '.join(current_line + [word])) <= new_w - 2*text_pad_h:
                        current_line.append(word)
                    else:
                        if current_line:
                            wrapped_lines.append(' '.join(current_line))
                        current_line = [word]
                if current_line:
                    wrapped_lines.append(' '.join(current_line))
            line_count = max(1, len(wrapped_lines))
            required_height = int(metrics.height() * line_count * height_factor) + 2 * text_pad_v
            new_h = max(h, min(required_height, h + max_expand_h))
            delta_h = new_h - h
            new_y = y - delta_h // 2
            
            expanded_rects.append((new_x, new_y, new_w, new_h, text))
        
        min_x = min(x for x, y, w, h, text in expanded_rects)
        min_y = min(y for x, y, w, h, text in expanded_rects)
        max_x = max(x + w for x, y, w, h, text in expanded_rects)
        max_y = max(y + h for x, y, w, h, text in expanded_rects)
        self.setGeometry(min_x, min_y, max_x - min_x, max_y - min_y)
        
        self.local_rects = [(x - min_x, y - min_y, w, h, text) for (x, y, w, h, text) in expanded_rects]
        
        self.merged_path = QPainterPath()
        for rx, ry, rw, rh, _ in self.local_rects:
            rect_path = QPainterPath()
            rect_path.addRect(rx, ry, rw, rh)
            self.merged_path = self.merged_path.united(rect_path)
        
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.show()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillPath(self.merged_path, QBrush(QColor(0, 0, 0, 200)))
        painter.setPen(QPen(QColor(255, 0, 127), 2))
        painter.drawPath(self.merged_path)
        painter.setFont(self.font)
        painter.setPen(QPen(QColor(255, 255, 255)))
        
        width_factor = settings.get("TEXT_BOX_WIDTH_FACTOR", 1.5)
        height_factor = settings.get("TEXT_BOX_HEIGHT_FACTOR", 1.5)
        text_pad_h = int(10 * width_factor)
        text_pad_v = int(10 * height_factor)
        
        for rx, ry, rw, rh, text in self.local_rects:
            if not text:
                continue
            text_rect = QRect(rx + text_pad_h//2, ry + text_pad_v//2, rw - text_pad_h, rh - text_pad_v)
            painter.drawText(text_rect, Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap, text)

if __name__ == "__main__":
    try:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        EASYOCR_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        load_settings()
        app = TranslatorApp(sys.argv)
        sys.exit(app.exec())
    except Exception as e:
        print(f"\n❌ ОШИБКА: {e}")
        input("Нажмите Enter для выхода...")
        sys.exit(1)