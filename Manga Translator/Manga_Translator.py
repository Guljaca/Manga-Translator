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
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QLineEdit, QSpinBox, QDoubleSpinBox, QProgressBar, QSystemTrayIcon, QMenu,
    QMainWindow, QGroupBox, QFormLayout, QMessageBox, QComboBox, QCheckBox,
    QFontComboBox, QFileDialog, QScrollArea, QColorDialog, QSlider, QTextEdit,
    QSplitter, QListWidget, QListWidgetItem
)
from PyQt6.QtCore import Qt, QSize, QTimer, QThread, pyqtSignal, QPoint, QRect
from PyQt6.QtGui import (
    QPainter, QPen, QColor, QBrush, QIcon, QAction, QPixmap, QFont, QPainterPath,
    QFontMetrics, QImage
)

print("[DEBUG] Загрузка модулей...")

# ----------------------- Импорт torch -----------------------
try:
    import torch
    TORCH_AVAILABLE = True
    print("Torch available, CUDA:", torch.cuda.is_available())
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️ PyTorch не установлен")

# ----------------------- OCR движки -----------------------
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    print("⚠️ pytesseract не установлен")

try:
    import winocr
    WINDOWS_OCR_AVAILABLE = True
except ImportError:
    WINDOWS_OCR_AVAILABLE = False
    print("⚠️ winocr не установлен")

MANGA_OCR_AVAILABLE = False
try:
    from manga_ocr import MangaOcr
    MANGA_OCR_AVAILABLE = True
    print("✅ Manga OCR доступен")
except ImportError:
    print("⚠️ manga-ocr не установлен")

# ----------------------- Пути -----------------------
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
# Списки языков
# ----------------------------------------------------------------------
SOURCE_LANGUAGES = [
    "Japanese", "Chinese (Simplified)", "Chinese (Traditional)",
    "English", "Korean", "Russian", "French", "German", "Spanish"
]

SOURCE_TO_EASYOCR = {
    "Japanese": "ja", "Chinese (Simplified)": "ch_sim", "Chinese (Traditional)": "ch_tra",
    "English": "en", "Korean": "ko", "Russian": "ru", "French": "fr", "German": "de", "Spanish": "es"
}

TESSERACT_LANG_MAP = {
    "ja": "jpn", "ch_sim": "chi_sim", "ch_tra": "chi_tra", "en": "eng", "ko": "kor",
    "ru": "rus", "fr": "fra", "de": "deu", "es": "spa", "it": "ita", "ar": "ara",
    "el": "ell", "nl": "nld", "pl": "pol", "pt": "por", "tr": "tur", "vi": "vie"
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
    "MANGA_OCR_ENGINES": ["easyocr"],
    "MANGA_OCR_STRATEGY": "best_confidence",
    "NOVEL_OCR_ENGINES": ["easyocr"],
    "NOVEL_OCR_STRATEGY": "best_confidence",
    "EASYOCR_CONFIDENCE": 0.2,
    "TESSERACT_CONFIDENCE": 0.3,
    "WINDOWS_OCR_CONFIDENCE": 0.3,
    "MANGA_OCR_CONFIDENCE": 0.4,
    "TESSERACT_PATH": "",
    "TEXT_BOX_WIDTH_FACTOR": 1.5,
    "TEXT_BOX_HEIGHT_FACTOR": 1.5,
    "ENABLE_OVERLAP_RELOCATION": True,
    "FONT_FAMILY": "Arial",
    "FONT_SIZE": 12,
    "FONT_COLOR": "#ffffff",
    "FONT_BOLD": False,
    "FONT_OUTLINE": 0,
    "BACKGROUND_COLOR": "#000000",
    "BACKGROUND_OPACITY": 200,
    "OUTLINE_COLOR": "#ff007f",
    "USE_GPU_OCR": True,
    "USE_GPU_MODEL": False,
    "ASYMMETRIC_TRANSLATION": False,
    "MAX_CONCURRENT_REQUESTS": 5,
    "SCREENSHOT_SIMILARITY_THRESHOLD": 0.95,
    "NOVEL_V2_OUTPUT_X": 100,
    "NOVEL_V2_OUTPUT_Y": 100,
    "NOVEL_V2_OUTPUT_W": 400,
    "NOVEL_V2_OUTPUT_H": 300,
    "STREAMING_OUTPUT": False,
    "MODE1": "novel_v2",
    "MODE2": "text",
    "REGION1": [100, 100, 300, 200],
    "REGION2": [150, 150, 400, 300],
    "ENABLE_BINARIZATION": True,
    "ENABLE_INVERSION": True,
    "BINARIZATION_THRESHOLD": 127,
    "BINARIZATION_BLOCK_SIZE": 0,
    "BINARIZATION_C": 2,
}

settings = default_settings.copy()
translation_history = []
history_lock = threading.Lock()

mocr = None
yolo = None
easyocr_reader = None
current_ocr_lang = None

ocr_log = []  # (text, pixmap, timestamp)
ocr_log_lock = threading.Lock()
last_screenshot_pixmap = None
last_screenshot_lock = threading.Lock()

def add_ocr_log_entry(entry, pixmap=None):
    with ocr_log_lock:
        if pixmap is not None and not pixmap.isNull():
            scaled = pixmap.scaled(150, 150, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        else:
            scaled = None
        ocr_log.append((entry, scaled, time.time()))
        if len(ocr_log) > 100:
            ocr_log.pop(0)

def get_ocr_log():
    with ocr_log_lock:
        return ocr_log.copy()

def set_last_screenshot(pixmap):
    global last_screenshot_pixmap
    with last_screenshot_lock:
        last_screenshot_pixmap = pixmap

def get_last_screenshot():
    with last_screenshot_lock:
        return last_screenshot_pixmap

def get_ocr_lang_from_source():
    return SOURCE_TO_EASYOCR.get(settings["SOURCE_LANG"], "en")

def sync_all_ocr_langs():
    lang_code = get_ocr_lang_from_source()
    settings["MANGA_OCR_LANG"] = lang_code
    settings["TEXT_OCR_LANG"] = lang_code
    settings["NOVEL_OCR_LANG"] = lang_code

def load_settings():
    global settings
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                saved = json.load(f)
            for key in default_settings:
                if key in saved:
                    settings[key] = saved[key]
            for key in default_settings:
                if key not in settings:
                    settings[key] = default_settings[key]
        except Exception as e:
            print(f"⚠️ Ошибка загрузки настроек: {e}")
    if "USE_GPU_OCR" not in settings:
        settings["USE_GPU_OCR"] = TORCH_AVAILABLE and torch.cuda.is_available()
    sync_all_ocr_langs()

def save_settings_to_file():
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"⚠️ Ошибка сохранения настроек: {e}")

# ----------------------------------------------------------------------
# Предобработка изображений
# ----------------------------------------------------------------------
def preprocess_image(img_cv, method):
    if method == 'original':
        return img_cv.copy(), "оригинал"
    elif method == 'binarized':
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        thresh = settings.get("BINARIZATION_THRESHOLD", 127)
        block_size = settings.get("BINARIZATION_BLOCK_SIZE", 0)
        if block_size > 0 and block_size % 2 == 1:
            C = settings.get("BINARIZATION_C", 2)
            binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY, block_size, C)
        else:
            _, binary = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
        binary_bgr = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        return binary_bgr, "бинаризация"
    elif method == 'inverted':
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        thresh = settings.get("BINARIZATION_THRESHOLD", 127)
        block_size = settings.get("BINARIZATION_BLOCK_SIZE", 0)
        if block_size > 0 and block_size % 2 == 1:
            C = settings.get("BINARIZATION_C", 2)
            binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY, block_size, C)
        else:
            _, binary = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
        inverted = cv2.bitwise_not(binary)
        inverted_bgr = cv2.cvtColor(inverted, cv2.COLOR_GRAY2BGR)
        return inverted_bgr, "инверсия"
    else:
        return img_cv.copy(), "исходное"

def cv2_to_pixmap(cv_img):
    if cv_img is None:
        return QPixmap()
    rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    bytes_per_line = ch * w
    qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg)

# ----------------------------------------------------------------------
# Управление моделями
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

def unload_mode_models(mode):
    global yolo
    if mode != "manga":
        if yolo is not None:
            del yolo
            yolo = None
            gc.collect()

def is_gpu_available_for_ocr():
    return settings.get("USE_GPU_OCR", False) and TORCH_AVAILABLE and torch.cuda.is_available()

def ensure_yolo():
    global yolo
    if yolo is None:
        device = 'cuda' if is_gpu_available_for_ocr() else 'cpu'
        from ultralytics import YOLO
        yolo = YOLO(str(YOLO_MODEL_PATH))
        if device == 'cuda':
            yolo.to('cuda')
    return yolo

def ensure_manga_ocr():
    global mocr
    if mocr is None and MANGA_OCR_AVAILABLE:
        device = 'cuda' if is_gpu_available_for_ocr() else 'cpu'
        try:
            mocr = MangaOcr(pretrained_model_name_or_path=str(MOCA_MODEL_DIR), device=device)
        except TypeError:
            mocr = MangaOcr(pretrained_model_name_or_path=str(MOCA_MODEL_DIR))
            if device == 'cuda' and hasattr(mocr, 'model') and hasattr(mocr.model, 'to'):
                mocr.model.to('cuda')
    return mocr

def ensure_easyocr():
    global easyocr_reader, current_ocr_lang
    mode = settings["MODE"]
    if mode == "manga":
        lang = settings["MANGA_OCR_LANG"]
    else:
        lang = settings["TEXT_OCR_LANG"]
    if easyocr_reader is None or current_ocr_lang != lang:
        if easyocr_reader is not None:
            del easyocr_reader
            gc.collect()
        use_gpu = is_gpu_available_for_ocr()
        import easyocr
        EASYOCR_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        easyocr_reader = easyocr.Reader([lang], gpu=use_gpu, model_storage_directory=str(EASYOCR_MODEL_DIR))
        current_ocr_lang = lang
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

# ---------- Базовые OCR-функции ----------
def ocr_with_easyocr(img_cv, lang_code):
    reader = ensure_easyocr()
    try:
        results = reader.readtext(img_cv, paragraph=True)
        if not results:
            return "", 0.0
        text = results[0][1]
        word_results = reader.readtext(img_cv, paragraph=False)
        confs = [conf for (_, _, conf) in word_results if conf > 0]
        avg_conf = sum(confs)/len(confs) if confs else 0.5
        return text.strip(), avg_conf
    except Exception as e:
        add_ocr_log_entry(f"   [EasyOCR] ошибка: {e}")
        return "", 0.0

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
            add_ocr_log_entry(f"   [Tesseract] язык {tesseract_lang} не найден")
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
        add_ocr_log_entry(f"   [Tesseract] ошибка: {e}")
        return "", 0.0

def ocr_with_windows_ocr(img_cv, lang_code):
    if not WINDOWS_OCR_AVAILABLE or sys.platform != "win32":
        return "", 0.0
    winocr_lang_map = {
        "ch_sim": "zh-CN", "ch_tra": "zh-TW", "ja": "ja-JP", "en": "en-US", "ko": "ko-KR", "ru": "ru-RU",
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
        add_ocr_log_entry(f"   [Windows OCR] ошибка: {e}")
        return "", 0.0

def ocr_with_manga_ocr(img_cv, lang_code):
    if not MANGA_OCR_AVAILABLE:
        return "", 0.0
    if lang_code not in ("ja", "jpn", "Japanese"):
        return "", 0.0
    model = ensure_manga_ocr()
    if model is None:
        return "", 0.0
    try:
        pil_img = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
        text = model(pil_img)
        if text and len(text.strip()) >= 2:
            return text.strip(), 0.9
        return "", 0.0
    except Exception as e:
        add_ocr_log_entry(f"   [Manga OCR] ошибка: {e}")
        return "", 0.0

# ----------------------------------------------------------------------
# Функции детекции и распознавания на основе предобработанного изображения
# ----------------------------------------------------------------------
def detect_and_recognize_on_image(img_cv, mode, preproc_name, preproc_pixmap):
    """
    Применяет детекцию и OCR к уже предобработанному изображению.
    Возвращает список (текст, уверенность, блок_инфо) для manga/text,
    либо (текст, уверенность) для novel.
    Также логирует миниатюру полного обработанного изображения (preproc_pixmap).
    """
    if mode == "manga":
        yolo_model = ensure_yolo()
        boxes = detect_bubbles(yolo_model, img_cv)
        results = []
        for (bx, by, bw, bh) in boxes:
            crop = img_cv[by:by+bh, bx:bx+bw]
            if crop.size == 0:
                continue
            # Распознавание текста в облачке (здесь используется либо manga_ocr, либо easyocr, либо multi)
            text = ocr_on_image_manga(crop)   # уже включает свой выбор методов
            if text and len(text.strip()) >= 2:
                results.append((text, 0.9, (bx, by, bw, bh)))  # уверенность приблизительная
        add_ocr_log_entry(f"   {preproc_name}: распознано {len(results)} облачков")
        return results
    elif mode == "text":
        blocks = ocr_on_image_text_blocks(img_cv)   # возвращает (x,y,w,h,text)
        results = [(text, 0.9, (x,y,w,h)) for (x,y,w,h,text) in blocks]  # уверенность из блока не сохраняется, ставим 0.9
        add_ocr_log_entry(f"   {preproc_name}: распознано {len(results)} текстовых блоков")
        return results
    else:  # novel
        # Для novel просто распознаём весь кадр
        text, conf = ocr_on_image_novel(img_cv)   # возвращает (текст, уверенность)
        add_ocr_log_entry(f"   {preproc_name}: '{text}' (conf={conf:.2f})")
        return [(text, conf, None)]

def ocr_on_image_manga(img_cv):
    # Упрощённо: просто вызов manga_ocr или easyocr без мульти-движков внутри (можно оставить как было)
    # Здесь можно использовать отдельную логику, но для единообразия лучше вызвать multi_ocr_recognize с одним движком
    backend = settings["MANGA_OCR_BACKEND"]
    if backend == "manga_ocr" and MANGA_OCR_AVAILABLE:
        model = ensure_manga_ocr()
        if model:
            pil_img = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
            text = model(pil_img)
            return text if len(text.strip())>=2 else ""
    if backend == "easyocr" or not MANGA_OCR_AVAILABLE:
        text, _ = ocr_with_easyocr(img_cv, settings["MANGA_OCR_LANG"])
        return text
    # multi
    lang = settings["MANGA_OCR_LANG"]
    engines = settings["MANGA_OCR_ENGINES"]
    strat = settings["MANGA_OCR_STRATEGY"]
    text, _ = multi_ocr_recognize(img_cv, lang, engines=engines, strategy=strat, log_prefix="")
    return text

def ocr_on_image_text_blocks(img_cv):
    # Используем EasyOCR для детекции блоков, затем multi_ocr_recognize для каждого
    reader = ensure_easyocr()
    confidence_threshold = settings.get("EASYOCR_CONFIDENCE", 0.2)
    results = reader.readtext(img_cv, paragraph=False)
    blocks = []
    for (bbox, easy_text, conf) in results:
        if conf < confidence_threshold or len(easy_text.strip()) < 2:
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
        crop = img_cv[y:y+h, x:x+w]
        if crop.size == 0:
            continue
        best_text, best_conf = multi_ocr_recognize(crop, settings["TEXT_OCR_LANG"],
                                                   engines=settings["NOVEL_OCR_ENGINES"],
                                                   strategy=settings["NOVEL_OCR_STRATEGY"],
                                                   log_prefix="")
        if best_text and len(best_text.strip()) >= 2 and best_conf >= confidence_threshold:
            blocks.append((x, y, w, h, best_text))
    return blocks

def ocr_on_image_novel(img_cv):
    # Просто multi_ocr_recognize на всём изображении
    lang = settings["NOVEL_OCR_LANG"]
    engines = settings.get("NOVEL_OCR_ENGINES", ["easyocr"])
    strat = settings.get("NOVEL_OCR_STRATEGY", "best_confidence")
    text, conf = multi_ocr_recognize(img_cv, lang, engines=engines, strategy=strat, log_prefix="")
    return text, conf

def multi_ocr_recognize(img_cv, lang_code, engines=None, strategy="best_confidence", return_all=False, log_prefix=""):
    # Упрощённо: перебирает движки, но без предобработки (предобработка уже сделана на уровне выше)
    if engines is None:
        engines = ["easyocr"]
    all_results = []
    for eng in engines:
        if eng == "easyocr" and ensure_easyocr():
            text, conf = ocr_with_easyocr(img_cv, lang_code)
        elif eng == "tesseract" and TESSERACT_AVAILABLE:
            text, conf = ocr_with_tesseract(img_cv, lang_code)
        elif eng == "windows_ocr" and WINDOWS_OCR_AVAILABLE and sys.platform=="win32":
            text, conf = ocr_with_windows_ocr(img_cv, lang_code)
        elif eng == "manga_ocr" and MANGA_OCR_AVAILABLE:
            text, conf = ocr_with_manga_ocr(img_cv, lang_code)
        else:
            continue
        if text and len(text.strip()) >= 2:
            all_results.append((eng, text, conf))
    if not all_results:
        return "", 0.0
    if strategy == "best_confidence":
        best = max(all_results, key=lambda x: x[2])
    else:
        best = all_results[0]
    if return_all:
        return best[1], all_results
    else:
        return best[1], best[2]

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
    add_ocr_log_entry(f"🔍 YOLO найдено {len(boxes)} облачков")
    return boxes

# ----------------------------------------------------------------------
# Перевод
# ----------------------------------------------------------------------
def get_translation(text, use_context=True, callback=None):
    if not text or len(text.strip()) < 2:
        return ""
    src = settings["SOURCE_LANG"]
    tgt = settings["TARGET_LANG"]
    messages = [{"role": "system", "content": f"You are a professional {src}-to-{tgt} translator. Respond only with the {tgt} translation."}]
    if use_context:
        with history_lock:
            recent = translation_history[-settings["CONTEXT_SIZE"]:] if translation_history else []
            if recent:
                ctx = "\n".join([f"Original ({src}): {o}\n{tgt}: {r}" for o, r in recent])
                messages.append({"role": "user", "content": f"Context:\n{ctx}"})
    messages.append({"role": "user", "content": f"Translate from {src} to {tgt}:\n{text}"})
    try:
        if settings.get("STREAMING_OUTPUT", False) and callback is not None:
            resp = requests.post(f"{settings['LM_STUDIO_URL']}/chat/completions",
                json={"model": settings["MODEL_NAME"], "messages": messages, "temperature": settings["TEMPERATURE"], "stream": True},
                timeout=settings["TIMEOUT"], stream=True)
            if resp.status_code != 200:
                return "[Ошибка перевода]"
            full_text = ""
            for line in resp.iter_lines(decode_unicode=False):
                if not line:
                    continue
                if isinstance(line, bytes):
                    line = line.decode('utf-8', errors='replace')
                line = line.strip()
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                        delta = data.get("choices", [{}])[0].get("delta", {})
                        if "content" in delta:
                            token = delta["content"]
                            full_text += token
                            callback(full_text)
                    except:
                        continue
            if full_text and full_text != "[Ошибка перевода]":
                if use_context:
                    with history_lock:
                        translation_history.append((text, full_text))
                        if len(translation_history) > 100:
                            translation_history.pop(0)
            return full_text
        else:
            resp = requests.post(f"{settings['LM_STUDIO_URL']}/chat/completions",
                json={"model": settings["MODEL_NAME"], "messages": messages, "temperature": settings["TEMPERATURE"]},
                timeout=settings["TIMEOUT"])
            if resp.status_code != 200:
                return "[Ошибка перевода]"
            data = resp.json()
            trans = data['choices'][0]['message']['content'].strip()
            if trans and trans != "[Ошибка перевода]" and use_context:
                with history_lock:
                    translation_history.append((text, trans))
                    if len(translation_history) > 100:
                        translation_history.pop(0)
            return trans
    except Exception as e:
        print(f"Ошибка перевода: {e}")
        return "[Ошибка перевода]"

# ----------------------------------------------------------------------
# Вспомогательные для оверлеев
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

def resolve_overlaps(overlays, font, step=5, max_shift=150):
    if not overlays:
        return overlays
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
        found = False
        for shift in range(step, max_shift+1, step):
            for dy in (shift, -shift, 0):
                for dx in (0, shift, -shift):
                    if dx==0 and dy==0:
                        continue
                    nx, ny = x+dx, y+dy
                    overlap = False
                    for (ox,oy,ow,oh,_) in result:
                        if not (nx+w <= ox or ox+ow <= nx or ny+h <= oy or oy+oh <= ny):
                            overlap = True
                            break
                    if not overlap:
                        result.append([nx, ny, w, h, text])
                        found = True
                        break
                if found:
                    break
            if found:
                break
        if not found:
            result.append([x, y, w, h, text])
    return [tuple(b) for b in result]

def group_overlays(overlays):
    if not overlays:
        return []
    rects = [(x, y, x+w, y+h, text) for (x,y,w,h,text) in overlays]
    n = len(rects)
    parent = list(range(n))
    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u
    def union(u,v):
        ru, rv = find(u), find(v)
        if ru != rv:
            parent[rv] = ru
    for i in range(n):
        x1i,y1i,x2i,y2i,_ = rects[i]
        for j in range(i+1,n):
            x1j,y1j,x2j,y2j,_ = rects[j]
            if not (x2i <= x1j or x2j <= x1i or y2i <= y1j or y2j <= y1i):
                union(i,j)
    groups = {}
    for i in range(n):
        root = find(i)
        groups.setdefault(root, []).append(rects[i])
    result = []
    for grp in groups.values():
        group_items = [(x1, y1, x2-x1, y2-y1, text) for (x1,y1,x2,y2,text) in grp]
        result.append(group_items)
    return result

# ----------------------------------------------------------------------
# CaptureFrame, ControlPanel, NovelV2OutputWindow, CompositeOverlay, CompositeOverlayMangaGroup, RegionSelector, TranslationThread, SettingsWindow, TranslatorApp
# ----------------------------------------------------------------------
class CaptureFrame(QWidget):
    def __init__(self, x, y, w, h, parent_app, region_id=0):
        super().__init__()
        self.parent_app = parent_app
        self.region_id = region_id
        self.translated_text = ""
        self.output_window = None
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setGeometry(x, y, w, h)
        self.setMinimumSize(100, 100)
        self.is_novel_v1 = (settings["MODE"] == "novel_v1")
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
        self.drag_pos = None
        self.resizing = False
        self.resize_edge = None
        self.resize_margin = 20
        self.show()
        self.control_panel = ControlPanel(self, parent_app)
        self.control_panel.show()
        self.control_panel.update_position()
    def apply_mode_behavior(self):
        self.is_novel_v1 = (settings["MODE"] == "novel_v1")
        self.update()
    def update_appearance_from_settings(self):
        self.update()
        if hasattr(self, 'control_panel'):
            self.control_panel.update_style()
    def get_background_brush(self):
        if self.is_novel_v1:
            bg = QColor(settings.get("BACKGROUND_COLOR","#000000"))
            bg.setAlpha(settings.get("BACKGROUND_OPACITY",200))
            return QBrush(bg)
        return QBrush(QColor(0,0,0,0))
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        if self.is_novel_v1:
            painter.fillRect(self.rect(), self.get_background_brush())
        painter.setPen(QPen(QColor(settings.get("OUTLINE_COLOR","#ff007f")),3))
        painter.drawRect(2,2,self.width()-4,self.height()-4)
        if self.is_novel_v1 and self.translated_text:
            font = QFont(settings.get("FONT_FAMILY","Arial"), settings.get("FONT_SIZE",12))
            if settings.get("FONT_BOLD",False):
                font.setBold(True)
            painter.setFont(font)
            fc = QColor(settings.get("FONT_COLOR","#ffffff"))
            outline = settings.get("FONT_OUTLINE",0)
            margin=8
            tr = QRect(margin,margin,self.width()-2*margin,self.height()-2*margin)
            if outline>0:
                for dx,dy in [(-outline,0),(outline,0),(0,-outline),(0,outline)]:
                    painter.setPen(QPen(QColor(0,0,0),0))
                    painter.drawText(tr.adjusted(dx,dy,dx,dy), Qt.AlignmentFlag.AlignLeft|Qt.TextFlag.TextWordWrap, self.translated_text)
                painter.setPen(QPen(fc))
            else:
                painter.setPen(QPen(fc))
            painter.drawText(tr, Qt.AlignmentFlag.AlignLeft|Qt.TextFlag.TextWordWrap, self.translated_text)
    def _is_on_edge(self, pos):
        w,h=self.width(),self.height()
        return pos.x()<=self.resize_margin or pos.x()>=w-self.resize_margin or pos.y()<=self.resize_margin or pos.y()>=h-self.resize_margin
    def _get_resize_edge(self, pos):
        w,h=self.width(),self.height()
        on_left = pos.x()<=self.resize_margin
        on_right = pos.x()>=w-self.resize_margin
        on_top = pos.y()<=self.resize_margin
        on_bottom = pos.y()>=h-self.resize_margin
        if on_top and on_left: return 'top-left'
        if on_top and on_right: return 'top-right'
        if on_bottom and on_left: return 'bottom-left'
        if on_bottom and on_right: return 'bottom-right'
        if on_left: return 'left'
        if on_right: return 'right'
        if on_top: return 'top'
        if on_bottom: return 'bottom'
        return None
    def mousePressEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton:
            event.ignore()
            return
        pos = event.position().toPoint()
        if self.is_novel_v1:
            if self._is_on_edge(pos):
                self.resizing=True
                self.resize_edge=self._get_resize_edge(pos)
                event.accept()
            else:
                self.drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
                event.accept()
        else:
            if self._is_on_edge(pos):
                self.resizing=True
                self.resize_edge=self._get_resize_edge(pos)
                event.accept()
            else:
                event.ignore()
    def mouseMoveEvent(self, event):
        if self.resizing:
            gpos = event.globalPosition().toPoint()
            rect = self.geometry()
            min_w,min_h=100,100
            e=self.resize_edge
            if e=='right': rect.setWidth(max(min_w, gpos.x()-rect.x()))
            elif e=='left':
                delta=rect.x()-gpos.x()
                if delta<rect.width()-min_w:
                    new_x=gpos.x()
                    new_w=rect.width()+delta
                    if new_w>=min_w:
                        rect.setX(new_x)
                        rect.setWidth(new_w)
            elif e=='bottom': rect.setHeight(max(min_h, gpos.y()-rect.y()))
            elif e=='top':
                delta=rect.y()-gpos.y()
                if delta<rect.height()-min_h:
                    new_y=gpos.y()
                    new_h=rect.height()+delta
                    if new_h>=min_h:
                        rect.setY(new_y)
                        rect.setHeight(new_h)
            elif e=='top-left':
                dx=rect.x()-gpos.x()
                dy=rect.y()-gpos.y()
                if dx<rect.width()-min_w:
                    new_x=gpos.x()
                    new_w=rect.width()+dx
                    if new_w>=min_w:
                        rect.setX(new_x)
                        rect.setWidth(new_w)
                if dy<rect.height()-min_h:
                    new_y=gpos.y()
                    new_h=rect.height()+dy
                    if new_h>=min_h:
                        rect.setY(new_y)
                        rect.setHeight(new_h)
            elif e=='top-right':
                dy=rect.y()-gpos.y()
                new_w=max(min_w, gpos.x()-rect.x())
                if dy<rect.height()-min_h:
                    new_y=gpos.y()
                    new_h=rect.height()+dy
                    if new_h>=min_h:
                        rect.setY(new_y)
                        rect.setHeight(new_h)
                rect.setWidth(new_w)
            elif e=='bottom-left':
                dx=rect.x()-gpos.x()
                new_h=max(min_h, gpos.y()-rect.y())
                if dx<rect.width()-min_w:
                    new_x=gpos.x()
                    new_w=rect.width()+dx
                    if new_w>=min_w:
                        rect.setX(new_x)
                        rect.setWidth(new_w)
                rect.setHeight(new_h)
            elif e=='bottom-right':
                new_w=max(min_w, gpos.x()-rect.x())
                new_h=max(min_h, gpos.y()-rect.y())
                rect.setSize(QSize(new_w,new_h))
            self.setGeometry(rect)
            if self.region_id>0:
                if self.region_id==1:
                    settings["REGION1"]=[rect.x(),rect.y(),rect.width(),rect.height()]
                else:
                    settings["REGION2"]=[rect.x(),rect.y(),rect.width(),rect.height()]
                save_settings_to_file()
                self.parent_app.update_dual_regions()
            else:
                self.parent_app.update_region(rect.x(),rect.y(),rect.width(),rect.height())
            if self.control_panel:
                self.control_panel.update_position()
            event.accept()
        elif self.drag_pos and self.is_novel_v1:
            self.move(event.globalPosition().toPoint() - self.drag_pos)
            g=self.geometry()
            if self.region_id>0:
                if self.region_id==1:
                    settings["REGION1"]=[g.x(),g.y(),g.width(),g.height()]
                else:
                    settings["REGION2"]=[g.x(),g.y(),g.width(),g.height()]
                save_settings_to_file()
                self.parent_app.update_dual_regions()
            else:
                self.parent_app.update_region(g.x(),g.y(),g.width(),g.height())
            if self.control_panel:
                self.control_panel.update_position()
            event.accept()
        else:
            event.ignore()
    def mouseReleaseEvent(self, event):
        self.resizing=False
        self.drag_pos=None
        event.accept()
    def close_frame(self):
        self.parent_app.clear_overlays()
        if self.output_window:
            self.output_window.close()
            self.output_window = None
        if self.parent_app.novel_v2_output_window and self.parent_app.novel_v2_output_window == self.output_window:
            self.parent_app.novel_v2_output_window = None
        if self.control_panel:
            self.control_panel.close()
        if self.region_id == 0:
            self.parent_app.capture_frame = None
        else:
            self.parent_app.dual_frames[self.region_id] = None
        self.close()
    def set_translated_text(self, text):
        self.translated_text = text
        self.update()
    def clear_text(self):
        self.translated_text = ""
        self.update()

class ControlPanel(QWidget):
    def __init__(self, capture_frame, parent_app):
        super().__init__()
        self.capture_frame = capture_frame
        self.parent_app = parent_app
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.update_style()
        layout = QHBoxLayout()
        layout.setContentsMargins(5,5,5,5)
        self.btn_translate = QPushButton("OCR")
        self.btn_auto = QPushButton("Авто")
        self.btn_clear = QPushButton("Очистить")
        self.btn_close = QPushButton("✕")
        for btn in (self.btn_translate, self.btn_auto, self.btn_clear, self.btn_close):
            btn.setStyleSheet("QPushButton { background-color: #2d2d2d; color: white; border: 1px solid #ff007f; border-radius: 4px; padding: 4px 8px; font-size: 11px; font-weight: bold; } QPushButton:hover { background-color: #ff007f; color: black; }")
        layout.addWidget(self.btn_translate)
        layout.addWidget(self.btn_auto)
        layout.addWidget(self.btn_clear)
        layout.addWidget(self.btn_close)
        self.setLayout(layout)
        self.btn_translate.clicked.connect(self.translate)
        self.btn_auto.clicked.connect(self.toggle_auto)
        self.btn_clear.clicked.connect(self.clear)
        self.btn_close.clicked.connect(self.close_frame)
        self.update_position()
        self.show()
        self.auto_mode = False
        self.auto_timer = None
        self.prev_text_hash = None
        self.prev_screenshot = None
    def update_style(self):
        self.setStyleSheet("background-color: rgba(20,20,20,200); border-radius: 5px; border: 1px solid #ff007f;")
    def update_position(self):
        if self.capture_frame:
            geo = self.capture_frame.geometry()
            self.move(geo.x(), geo.y() - self.height() - 5)
    def translate(self):
        if self.capture_frame.region_id == 0:
            mode = settings["MODE"]
        else:
            if self.capture_frame.region_id == 1:
                mode = settings.get("MODE1", "novel_v2")
            else:
                mode = settings.get("MODE2", "text")
        self.parent_app.manual_translate_for_region(self.capture_frame.region_id, self.capture_frame.geometry(), mode, self.capture_frame)
    def toggle_auto(self):
        self.auto_mode = not self.auto_mode
        if self.auto_mode:
            self.btn_auto.setStyleSheet("background-color: #ff007f; color: black; border:1px solid #ff007f;")
            self.prev_text_hash = None
            self.prev_screenshot = None
            if not self.auto_timer:
                self.auto_timer = QTimer()
                self.auto_timer.timeout.connect(self.auto_check)
            interval = int(settings["AUTO_CHECK_INTERVAL"] * 1000)
            self.auto_timer.start(interval)
        else:
            self.btn_auto.setStyleSheet("background-color: #2d2d2d; color: white; border:1px solid #ff007f;")
            if self.auto_timer:
                self.auto_timer.stop()
    def auto_check(self):
        if not self.auto_mode:
            return
        if self.parent_app.translation_thread and self.parent_app.translation_thread.isRunning():
            return
        rect = self.capture_frame.geometry()
        x,y,w,h = rect.x(), rect.y(), rect.width(), rect.height()
        if self.capture_frame.region_id == 0:
            mode = settings["MODE"]
        else:
            if self.capture_frame.region_id == 1:
                mode = settings.get("MODE1", "novel_v2")
            else:
                mode = settings.get("MODE2", "text")
        if mode == "novel_v1" and self.capture_frame.isVisible():
            old_op = self.capture_frame.windowOpacity()
            self.capture_frame.setWindowOpacity(0.0)
            QApplication.processEvents()
            time.sleep(0.02)
            screenshot = pyautogui.screenshot(region=(x,y,w,h))
            self.capture_frame.setWindowOpacity(old_op)
        else:
            screenshot = pyautogui.screenshot(region=(x,y,w,h))
        np_img = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
        if self.prev_screenshot is not None:
            if are_screenshots_similar(np_img, self.prev_screenshot, settings.get("SCREENSHOT_SIMILARITY_THRESHOLD",0.95)):
                return
        self.prev_screenshot = np_img.copy()
        # Здесь должен быть полный цикл предобработки всего скриншота
        # Но в auto_check мы используем ту же логику, что и в ручном режиме
        # Для простоты переиспользуем manual_translate
        self.translate()
    def clear(self):
        self.capture_frame.clear_text()
        if self.capture_frame.region_id != 1:
            self.parent_app.clear_overlays()
        if self.capture_frame.output_window:
            self.capture_frame.output_window.clear_text()
    def close_frame(self):
        self.capture_frame.close_frame()
    def resizeEvent(self, event):
        self.update_position()
        super().resizeEvent(event)

def are_screenshots_similar(img1, img2, threshold):
    if img1 is None or img2 is None:
        return False
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    if h1 != h2 or w1 != w2:
        small1 = cv2.resize(img1, (100, 100), interpolation=cv2.INTER_AREA)
        small2 = cv2.resize(img2, (100, 100), interpolation=cv2.INTER_AREA)
    else:
        small1 = cv2.resize(img1, (100, 100), interpolation=cv2.INTER_AREA)
        small2 = cv2.resize(img2, (100, 100), interpolation=cv2.INTER_AREA)
    gray1 = cv2.cvtColor(small1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(small2, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray1, gray2)
    mean_diff = np.mean(diff) / 255.0
    similarity = 1.0 - mean_diff
    return similarity >= threshold

class NovelV2OutputWindow(QWidget):
    def __init__(self, parent_app, capture_frame):
        super().__init__()
        self.parent_app = parent_app
        self.capture_frame = capture_frame
        self.setWindowFlags(Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setStyleSheet("background-color: rgba(0,0,0,180); border: 2px solid #ff007f; border-radius: 8px;")
        layout = QVBoxLayout()
        layout.setContentsMargins(8,8,8,8)
        self.text_label = QLabel("")
        self.text_label.setWordWrap(True)
        self.text_label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.update_font_and_color()
        layout.addWidget(self.text_label)
        self.setLayout(layout)
        x = settings.get("NOVEL_V2_OUTPUT_X", 100)
        y = settings.get("NOVEL_V2_OUTPUT_Y", 100)
        w = settings.get("NOVEL_V2_OUTPUT_W", 400)
        h = settings.get("NOVEL_V2_OUTPUT_H", 300)
        self.setGeometry(x, y, w, h)
        self.drag_pos = None
        self.resizing = False
        self.resize_edge = None
        self.resize_margin = 20
        self.show()
        self.save_timer = QTimer()
        self.save_timer.setSingleShot(True)
        self.save_timer.timeout.connect(self.save_geometry)
    def update_font_and_color(self):
        font = QFont(settings.get("FONT_FAMILY","Arial"), settings.get("FONT_SIZE",12))
        if settings.get("FONT_BOLD",False):
            font.setBold(True)
        self.text_label.setFont(font)
        self.text_label.setStyleSheet(f"color: {settings.get('FONT_COLOR','#ffffff')};")
    def set_text(self, text):
        self.text_label.setText(text)
    def clear_text(self):
        self.text_label.setText("")
    def save_geometry(self):
        g = self.geometry()
        settings["NOVEL_V2_OUTPUT_X"] = g.x()
        settings["NOVEL_V2_OUTPUT_Y"] = g.y()
        settings["NOVEL_V2_OUTPUT_W"] = g.width()
        settings["NOVEL_V2_OUTPUT_H"] = g.height()
        save_settings_to_file()
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setPen(QPen(QColor(settings.get("OUTLINE_COLOR","#ff007f")),2))
        painter.drawRect(0,0,self.width()-1,self.height()-1)
        super().paintEvent(event)
    def _is_on_edge(self, pos):
        w,h=self.width(),self.height()
        return pos.x()<=self.resize_margin or pos.x()>=w-self.resize_margin or pos.y()<=self.resize_margin or pos.y()>=h-self.resize_margin
    def _get_resize_edge(self, pos):
        w,h=self.width(),self.height()
        on_left = pos.x()<=self.resize_margin
        on_right = pos.x()>=w-self.resize_margin
        on_top = pos.y()<=self.resize_margin
        on_bottom = pos.y()>=h-self.resize_margin
        if on_top and on_left: return 'top-left'
        if on_top and on_right: return 'top-right'
        if on_bottom and on_left: return 'bottom-left'
        if on_bottom and on_right: return 'bottom-right'
        if on_left: return 'left'
        if on_right: return 'right'
        if on_top: return 'top'
        if on_bottom: return 'bottom'
        return None
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            pos = event.position().toPoint()
            if self._is_on_edge(pos):
                self.resizing=True
                self.resize_edge=self._get_resize_edge(pos)
                event.accept()
            else:
                self.drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
                event.accept()
        else:
            event.ignore()
    def mouseMoveEvent(self, event):
        if self.resizing:
            gpos = event.globalPosition().toPoint()
            rect = self.geometry()
            min_w,min_h=100,100
            e=self.resize_edge
            if e=='right': rect.setWidth(max(min_w, gpos.x()-rect.x()))
            elif e=='left':
                delta=rect.x()-gpos.x()
                if delta<rect.width()-min_w:
                    new_x=gpos.x()
                    new_w=rect.width()+delta
                    if new_w>=min_w:
                        rect.setX(new_x)
                        rect.setWidth(new_w)
            elif e=='bottom': rect.setHeight(max(min_h, gpos.y()-rect.y()))
            elif e=='top':
                delta=rect.y()-gpos.y()
                if delta<rect.height()-min_h:
                    new_y=gpos.y()
                    new_h=rect.height()+delta
                    if new_h>=min_h:
                        rect.setY(new_y)
                        rect.setHeight(new_h)
            elif e=='top-left':
                dx=rect.x()-gpos.x()
                dy=rect.y()-gpos.y()
                if dx<rect.width()-min_w:
                    new_x=gpos.x()
                    new_w=rect.width()+dx
                    if new_w>=min_w:
                        rect.setX(new_x)
                        rect.setWidth(new_w)
                if dy<rect.height()-min_h:
                    new_y=gpos.y()
                    new_h=rect.height()+dy
                    if new_h>=min_h:
                        rect.setY(new_y)
                        rect.setHeight(new_h)
            elif e=='top-right':
                dy=rect.y()-gpos.y()
                new_w=max(min_w, gpos.x()-rect.x())
                if dy<rect.height()-min_h:
                    new_y=gpos.y()
                    new_h=rect.height()+dy
                    if new_h>=min_h:
                        rect.setY(new_y)
                        rect.setHeight(new_h)
                rect.setWidth(new_w)
            elif e=='bottom-left':
                dx=rect.x()-gpos.x()
                new_h=max(min_h, gpos.y()-rect.y())
                if dx<rect.width()-min_w:
                    new_x=gpos.x()
                    new_w=rect.width()+dx
                    if new_w>=min_w:
                        rect.setX(new_x)
                        rect.setWidth(new_w)
                rect.setHeight(new_h)
            elif e=='bottom-right':
                new_w=max(min_w, gpos.x()-rect.x())
                new_h=max(min_h, gpos.y()-rect.y())
                rect.setSize(QSize(new_w,new_h))
            self.setGeometry(rect)
            self.save_timer.start(300)
            event.accept()
        elif self.drag_pos:
            self.move(event.globalPosition().toPoint() - self.drag_pos)
            self.save_timer.start(300)
            event.accept()
        else:
            event.ignore()
    def mouseReleaseEvent(self, event):
        self.resizing=False
        self.drag_pos=None
        self.save_geometry()
        event.accept()

class CompositeOverlay(QWidget):
    def __init__(self, group_items):
        super().__init__()
        self.group_items = group_items
        self.update_style_from_settings()
        base_x = min(item[0] for item in group_items)
        base_y = min(item[1] for item in group_items)
        self.local_rects = [(x-base_x, y-base_y, w, h, text) for (x,y,w,h,text) in group_items]
        self.merged_path = QPainterPath()
        for rx,ry,rw,rh,_ in self.local_rects:
            self.merged_path.addRect(rx,ry,rw,rh)
        max_x = max(rx+rw for rx,ry,rw,rh,_ in self.local_rects)
        max_y = max(ry+rh for rx,ry,rw,rh,_ in self.local_rects)
        self.setGeometry(base_x, base_y, max_x, max_y)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.Tool)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.show()
        self.block_texts = {}
        self.global_to_local = {}
    def set_block_texts(self, block_texts, global_to_local=None):
        self.block_texts = block_texts
        if global_to_local:
            self.global_to_local = global_to_local
        for gidx, text in block_texts.items():
            lidx = self.global_to_local.get(gidx, gidx)
            if lidx < len(self.local_rects):
                x,y,w,h,_ = self.local_rects[lidx]
                self.local_rects[lidx] = (x,y,w,h,text)
        self.update()
    def update_block_text(self, idx, new_text):
        if idx in self.block_texts:
            self.block_texts[idx] = new_text
            lidx = self.global_to_local.get(idx, idx)
            if lidx < len(self.local_rects):
                x,y,w,h,_ = self.local_rects[lidx]
                self.local_rects[lidx] = (x,y,w,h,new_text)
            self.update()
    def update_style_from_settings(self):
        self.font = QFont(settings.get("FONT_FAMILY","Arial"), settings.get("FONT_SIZE",12))
        if settings.get("FONT_BOLD",False):
            self.font.setBold(True)
        self.font_color = QColor(settings.get("FONT_COLOR","#ffffff"))
        self.outline_width = settings.get("FONT_OUTLINE",0)
        self.bg_color = QColor(settings.get("BACKGROUND_COLOR","#000000"))
        self.bg_color.setAlpha(settings.get("BACKGROUND_OPACITY",200))
        self.outline_color = QColor(settings.get("OUTLINE_COLOR","#ff007f"))
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillPath(self.merged_path, QBrush(self.bg_color))
        painter.setPen(QPen(self.outline_color,2))
        painter.drawPath(self.merged_path)
        painter.setFont(self.font)
        wf = settings.get("TEXT_BOX_WIDTH_FACTOR",1.5)
        hf = settings.get("TEXT_BOX_HEIGHT_FACTOR",1.5)
        pad_h = int(10*wf)
        pad_v = int(10*hf)
        for rx,ry,rw,rh,text in self.local_rects:
            if not text:
                continue
            tr = QRect(rx+pad_h//2, ry+pad_v//2, rw-pad_h, rh-pad_v)
            if self.outline_width>0:
                for dx,dy in [(-self.outline_width,0),(self.outline_width,0),(0,-self.outline_width),(0,self.outline_width)]:
                    painter.setPen(QPen(QColor(0,0,0),0))
                    painter.drawText(tr.adjusted(dx,dy,dx,dy), Qt.AlignmentFlag.AlignCenter|Qt.TextFlag.TextWordWrap, text)
                painter.setPen(QPen(self.font_color))
            else:
                painter.setPen(QPen(self.font_color))
            painter.drawText(tr, Qt.AlignmentFlag.AlignCenter|Qt.TextFlag.TextWordWrap, text)

class CompositeOverlayMangaGroup(QWidget):
    def __init__(self, group_items, max_expand_w=150, max_expand_h=80):
        super().__init__()
        self.group_items = group_items
        self.update_style_from_settings()
        metrics = QFontMetrics(self.font)
        wf = settings.get("TEXT_BOX_WIDTH_FACTOR",1.5)
        hf = settings.get("TEXT_BOX_HEIGHT_FACTOR",1.5)
        pad_h = int(10*wf)
        pad_v = int(10*hf)
        expanded = []
        for (x,y,w,h,text) in group_items:
            if not text:
                expanded.append((x,y,w,h,text))
                continue
            words = text.split()
            max_word_w = max(metrics.horizontalAdvance(w) for w in words) if words else 0
            need_w = max_word_w + 2*pad_h
            new_w = max(w, min(need_w, w+max_expand_w))
            dx = new_w - w
            nx = x - dx//2
            lines = text.split('\n') or [text]
            char_w = metrics.averageCharWidth()
            max_chars = max(1, int((new_w-2*pad_h)/char_w))
            wrapped=[]
            for line in lines:
                wl = line.split()
                cur=[]
                for word in wl:
                    if metrics.horizontalAdvance(' '.join(cur+[word])) <= new_w-2*pad_h:
                        cur.append(word)
                    else:
                        if cur:
                            wrapped.append(' '.join(cur))
                        cur=[word]
                if cur:
                    wrapped.append(' '.join(cur))
            line_cnt = max(1, len(wrapped))
            need_h = int(metrics.height()*line_cnt*hf) + 2*pad_v
            new_h = max(h, min(need_h, h+max_expand_h))
            dy = new_h - h
            ny = y - dy//2
            expanded.append((nx, ny, new_w, new_h, text))
        min_x = min(x for x,y,w,h,text in expanded)
        min_y = min(y for x,y,w,h,text in expanded)
        max_x = max(x+w for x,y,w,h,text in expanded)
        max_y = max(y+h for x,y,w,h,text in expanded)
        self.setGeometry(min_x, min_y, max_x-min_x, max_y-min_y)
        self.local_rects = [(x-min_x, y-min_y, w, h, text) for (x,y,w,h,text) in expanded]
        self.merged_path = QPainterPath()
        for rx,ry,rw,rh,_ in self.local_rects:
            self.merged_path.addRect(rx,ry,rw,rh)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.Tool)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.show()
        self.block_texts = {}
        self.global_to_local = {}
    def set_block_texts(self, block_texts, global_to_local=None):
        self.block_texts = block_texts
        if global_to_local:
            self.global_to_local = global_to_local
        for gidx, text in block_texts.items():
            lidx = self.global_to_local.get(gidx, gidx)
            if lidx < len(self.local_rects):
                x,y,w,h,_ = self.local_rects[lidx]
                self.local_rects[lidx] = (x,y,w,h,text)
        self.update()
    def update_block_text(self, idx, new_text):
        if idx in self.block_texts:
            self.block_texts[idx] = new_text
            lidx = self.global_to_local.get(idx, idx)
            if lidx < len(self.local_rects):
                x,y,w,h,_ = self.local_rects[lidx]
                self.local_rects[lidx] = (x,y,w,h,new_text)
            self.update()
    def update_style_from_settings(self):
        self.font = QFont(settings.get("FONT_FAMILY","Arial"), settings.get("FONT_SIZE",12))
        if settings.get("FONT_BOLD",False):
            self.font.setBold(True)
        self.font_color = QColor(settings.get("FONT_COLOR","#ffffff"))
        self.outline_width = settings.get("FONT_OUTLINE",0)
        self.bg_color = QColor(settings.get("BACKGROUND_COLOR","#000000"))
        self.bg_color.setAlpha(settings.get("BACKGROUND_OPACITY",200))
        self.outline_color = QColor(settings.get("OUTLINE_COLOR","#ff007f"))
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillPath(self.merged_path, QBrush(self.bg_color))
        painter.setPen(QPen(self.outline_color,2))
        painter.drawPath(self.merged_path)
        painter.setFont(self.font)
        wf = settings.get("TEXT_BOX_WIDTH_FACTOR",1.5)
        hf = settings.get("TEXT_BOX_HEIGHT_FACTOR",1.5)
        pad_h = int(10*wf)
        pad_v = int(10*hf)
        for rx,ry,rw,rh,text in self.local_rects:
            if not text:
                continue
            tr = QRect(rx+pad_h//2, ry+pad_v//2, rw-pad_h, rh-pad_v)
            if self.outline_width>0:
                for dx,dy in [(-self.outline_width,0),(self.outline_width,0),(0,-self.outline_width),(0,self.outline_width)]:
                    painter.setPen(QPen(QColor(0,0,0),0))
                    painter.drawText(tr.adjusted(dx,dy,dx,dy), Qt.AlignmentFlag.AlignCenter|Qt.TextFlag.TextWordWrap, text)
                painter.setPen(QPen(self.font_color))
            else:
                painter.setPen(QPen(self.font_color))
            painter.drawText(tr, Qt.AlignmentFlag.AlignCenter|Qt.TextFlag.TextWordWrap, text)

class RegionSelector(QWidget):
    def __init__(self, parent_app, region_id=0):
        super().__init__()
        self.parent_app = parent_app
        self.region_id = region_id
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setGeometry(100,100,400,300)
        self.setMinimumSize(100,100)
        self.drag_pos = None
        self.resizing = False
        self.resize_edge = None
        self.resize_margin = 10
        self.btn = QPushButton(f"✓ Зафиксировать область" + (f" {region_id}" if region_id>0 else ""), self)
        self.btn.setGeometry(10, self.height()-40, 150, 30)
        self.btn.setStyleSheet("background:#2d2d2d; color:white; border:1px solid #ff007f; border-radius:4px;")
        self.btn.clicked.connect(self.accept_region)
        label_text = "Перетащите и измените размер рамки\nЗатем нажмите кнопку"
        if region_id>0:
            label_text = f"Перетащите и измените размер рамки для области {region_id}\nЗатем нажмите кнопку"
        self.label = QLabel(label_text, self)
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
        painter.setPen(QPen(QColor(255,0,127),3))
        painter.drawRect(0,0,self.width()-1,self.height()-1)
    def mousePressEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton:
            event.ignore()
            return
        pos = event.position().toPoint()
        w,h=self.width(),self.height()
        on_left = pos.x()<=self.resize_margin
        on_right = pos.x()>=w-self.resize_margin
        on_top = pos.y()<=self.resize_margin
        on_bottom = pos.y()>=h-self.resize_margin
        if on_top and on_left:
            self.resizing,self.resize_edge=True,'top-left'
        elif on_top and on_right:
            self.resizing,self.resize_edge=True,'top-right'
        elif on_bottom and on_left:
            self.resizing,self.resize_edge=True,'bottom-left'
        elif on_bottom and on_right:
            self.resizing,self.resize_edge=True,'bottom-right'
        elif on_left:
            self.resizing,self.resize_edge=True,'left'
        elif on_right:
            self.resizing,self.resize_edge=True,'right'
        elif on_top:
            self.resizing,self.resize_edge=True,'top'
        elif on_bottom:
            self.resizing,self.resize_edge=True,'bottom'
        else:
            self.drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
        event.accept()
    def mouseMoveEvent(self, event):
        if self.resizing:
            gpos = event.globalPosition().toPoint()
            rect = self.geometry()
            min_w,min_h=100,100
            e=self.resize_edge
            if e=='right': rect.setWidth(max(min_w, gpos.x()-rect.x()))
            elif e=='left':
                delta=rect.x()-gpos.x()
                if delta<rect.width()-min_w:
                    new_x=gpos.x()
                    new_w=rect.width()+delta
                    if new_w>=min_w:
                        rect.setX(new_x)
                        rect.setWidth(new_w)
            elif e=='bottom': rect.setHeight(max(min_h, gpos.y()-rect.y()))
            elif e=='top':
                delta=rect.y()-gpos.y()
                if delta<rect.height()-min_h:
                    new_y=gpos.y()
                    new_h=rect.height()+delta
                    if new_h>=min_h:
                        rect.setY(new_y)
                        rect.setHeight(new_h)
            elif e=='top-left':
                dx=rect.x()-gpos.x()
                dy=rect.y()-gpos.y()
                if dx<rect.width()-min_w:
                    new_x=gpos.x()
                    new_w=rect.width()+dx
                    if new_w>=min_w:
                        rect.setX(new_x)
                        rect.setWidth(new_w)
                if dy<rect.height()-min_h:
                    new_y=gpos.y()
                    new_h=rect.height()+dy
                    if new_h>=min_h:
                        rect.setY(new_y)
                        rect.setHeight(new_h)
            elif e=='top-right':
                dy=rect.y()-gpos.y()
                new_w=max(min_w, gpos.x()-rect.x())
                if dy<rect.height()-min_h:
                    new_y=gpos.y()
                    new_h=rect.height()+dy
                    if new_h>=min_h:
                        rect.setY(new_y)
                        rect.setHeight(new_h)
                rect.setWidth(new_w)
            elif e=='bottom-left':
                dx=rect.x()-gpos.x()
                new_h=max(min_h, gpos.y()-rect.y())
                if dx<rect.width()-min_w:
                    new_x=gpos.x()
                    new_w=rect.width()+dx
                    if new_w>=min_w:
                        rect.setX(new_x)
                        rect.setWidth(new_w)
                rect.setHeight(new_h)
            elif e=='bottom-right':
                new_w=max(min_w, gpos.x()-rect.x())
                new_h=max(min_h, gpos.y()-rect.y())
                rect.setSize(QSize(new_w,new_h))
            self.setGeometry(rect)
        elif self.drag_pos:
            self.move(event.globalPosition().toPoint() - self.drag_pos)
        event.accept()
    def mouseReleaseEvent(self, event):
        self.resizing=False
        self.drag_pos=None
        event.accept()
    def accept_region(self):
        rect = self.geometry()
        if self.region_id == 0:
            self.parent_app.accept_single_region(rect)
        else:
            self.parent_app.accept_dual_region(rect, self.region_id)
        self.close()

class TranslationThread(QThread):
    finished = pyqtSignal(list, float, object)
    progress = pyqtSignal(int, int)
    prepare_overlays = pyqtSignal(list, list, str)
    update_overlay_text = pyqtSignal(int, str)
    update_novel_output = pyqtSignal(str)
    translation_error = pyqtSignal(str)

    def __init__(self, region, mode, capture_frame=None, region_id=0):
        super().__init__()
        self.region = region
        self.mode = mode
        self.capture_frame = capture_frame
        self.region_id = region_id
        self.start_time = None

    def run(self):
        self.start_time = time.time()
        x,y,w,h = self.region
        add_ocr_log_entry(f"▶️ НАЧАЛО ОБРАБОТКИ: режим={self.mode}, область={x},{y},{w},{h}")
        # Делаем скриншот
        if self.mode == "novel_v1" and self.capture_frame and self.capture_frame.isVisible():
            old_op = self.capture_frame.windowOpacity()
            self.capture_frame.setWindowOpacity(0.0)
            QApplication.processEvents()
            time.sleep(0.02)
            screenshot = pyautogui.screenshot(region=(x,y,w,h))
            self.capture_frame.setWindowOpacity(old_op)
        else:
            screenshot = pyautogui.screenshot(region=(x,y,w,h))
        qimage = screenshot.toqimage()
        pixmap = QPixmap.fromImage(qimage)
        if pixmap.width()>400:
            pixmap = pixmap.scaledToWidth(400, Qt.TransformationMode.SmoothTransformation)
        set_last_screenshot(pixmap)
        add_ocr_log_entry(f"📸 Скриншот ({w}x{h}) режим {self.mode}")
        np_img = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

        # ------------------------------------------------------------------
        # НОВАЯ ЛОГИКА: предобработка всего скриншота тремя способами
        # ------------------------------------------------------------------
        preprocess_methods = ["original"]
        if settings.get("ENABLE_BINARIZATION", True):
            preprocess_methods.append("binarized")
        if settings.get("ENABLE_INVERSION", True):
            preprocess_methods.append("inverted")
        
        # Собираем все результаты со всех вариантов
        all_recognitions = []  # (текст, уверенность, блок_инфо, название_метода, пиксмап_всего_скрина)
        for method in preprocess_methods:
            proc_img, method_name = preprocess_image(np_img, method)
            # Логируем миниатюру обработанного полного скриншота
            full_pixmap = cv2_to_pixmap(proc_img)
            add_ocr_log_entry(f"🖼️ Предобработка: {method_name}", full_pixmap)
            
            if self.mode == "manga":
                # Детекция облачков и распознавание
                yolo_model = ensure_yolo()
                boxes = detect_bubbles(yolo_model, proc_img)
                for (bx, by, bw, bh) in boxes:
                    crop = proc_img[by:by+bh, bx:bx+bw]
                    if crop.size == 0:
                        continue
                    text = ocr_on_image_manga(crop)
                    if text and len(text.strip()) >= 2:
                        all_recognitions.append((text, 0.9, (x+bx, y+by, bw, bh), method_name, full_pixmap))
            elif self.mode == "text":
                blocks = ocr_on_image_text_blocks(proc_img)
                for (bx, by, bw, bh, text) in blocks:
                    if text and len(text.strip()) >= 2:
                        all_recognitions.append((text, 0.9, (x+bx, y+by, bw, bh), method_name, full_pixmap))
            else:  # novel
                text, conf = ocr_on_image_novel(proc_img)
                if text and len(text.strip()) >= 2:
                    all_recognitions.append((text, conf, None, method_name, full_pixmap))
        
        # Дедупликация по тексту: оставляем вариант с максимальной уверенностью
        unique = {}
        for text, conf, block, method, full_pix in all_recognitions:
            if text not in unique or conf > unique[text][1]:
                unique[text] = (text, conf, block, method, full_pix)
        best_results = list(unique.values())
        
        if not best_results:
            add_ocr_log_entry("⚠️ Текст не распознан ни в одном варианте предобработки")
            self.finished.emit([], time.time()-self.start_time, None)
            return
        
        # Логируем итоговые выбранные тексты
        for text, conf, block, method, _ in best_results:
            add_ocr_log_entry(f"✅ Итоговый распознанный текст ({method}, увер.{conf:.2f}): '{text}'")
        
        # Для manga/text: формируем блоки данных для оверлеев
        if self.mode in ("manga","text"):
            blocks_data = []
            for text, conf, block, method, _ in best_results:
                if block is not None:
                    bx, by, bw, bh = block
                    blocks_data.append((bx, by, bw, bh, text))
            if not blocks_data:
                self.finished.emit([], time.time()-self.start_time, None)
                return
            indices = list(range(len(blocks_data)))
            # Подготавливаем оверлеи (пока с заглушками)
            self.prepare_overlays.emit(blocks_data, indices, self.mode)
            # Переводим
            overlays = []
            total = len(blocks_data)
            for idx, (bx, by, bw, bh, orig) in enumerate(blocks_data):
                trans = get_translation(orig, use_context=True)
                if trans and trans != "[Ошибка перевода]":
                    overlays.append((bx, by, bw, bh, trans))
                    self.update_overlay_text.emit(idx, trans)
                self.progress.emit(idx+1, total)
            elapsed = time.time() - self.start_time
            self.finished.emit(overlays, elapsed, None)
        else:  # novel
            # Берём лучший текст (с максимальной уверенностью)
            best = max(best_results, key=lambda x: x[1])
            text, conf, _, method, _ = best
            add_ocr_log_entry(f"📤 Отправлено на перевод (novel): '{text}' (увер.{conf:.2f})")
            translated = get_translation(text, use_context=True)
            if translated and translated != "[Ошибка перевода]":
                self.update_novel_output.emit(translated)
                overlays = [(x, y, w, h, translated)]
                add_ocr_log_entry(f"   Перевод novel: '{translated}'")
            else:
                overlays = []
            elapsed = time.time() - self.start_time
            self.finished.emit(overlays, elapsed, None)

# Функции ocr_on_image_manga, ocr_on_image_text_blocks, ocr_on_image_novel, multi_ocr_recognize определены выше (они уже не используют внутреннюю предобработку)
# Они должны быть определены до TranslationThread. Они есть.

# ----------------------------------------------------------------------
# SettingsWindow (сокращён для экономии, но полный)
# ----------------------------------------------------------------------
class SettingsWindow(QMainWindow):
    def __init__(self, parent_app):
        super().__init__()
        self.parent_app = parent_app
        self.setWindowTitle("Настройки переводчика")
        self.setWindowFlags(Qt.WindowType.WindowStaysOnTopHint)
        self.setMinimumSize(720,600)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setCentralWidget(scroll)
        central = QWidget()
        scroll.setWidget(central)
        layout = QVBoxLayout(central)
        layout.setSpacing(15)

        g1 = QGroupBox("LM Studio")
        f1 = QFormLayout()
        self.url_edit = QLineEdit(settings["LM_STUDIO_URL"])
        self.model_edit = QLineEdit(settings["MODEL_NAME"])
        f1.addRow("URL сервера:", self.url_edit)
        f1.addRow("Название модели:", self.model_edit)
        g1.setLayout(f1)
        layout.addWidget(g1)

        g_lang = QGroupBox("Языки перевода")
        f_lang = QFormLayout()
        self.source_lang_combo = QComboBox()
        self.source_lang_combo.addItems(SOURCE_LANGUAGES)
        self.source_lang_combo.setCurrentText(settings["SOURCE_LANG"])
        self.target_lang_combo = QComboBox()
        self.target_lang_combo.addItems(["Russian","English","Ukrainian","German","French","Spanish","Chinese (Simplified)","Chinese (Traditional)"])
        self.target_lang_combo.setCurrentText(settings["TARGET_LANG"])
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["manga","text","novel_v1","novel_v2","dual"])
        self.mode_combo.setCurrentText(settings["MODE"])
        f_lang.addRow("С какого языка:", self.source_lang_combo)
        f_lang.addRow("На какой язык:", self.target_lang_combo)
        f_lang.addRow("Режим:", self.mode_combo)
        g_lang.setLayout(f_lang)
        layout.addWidget(g_lang)

        g_dual = QGroupBox("Dual режим (настройки режимов областей)")
        f_dual = QFormLayout()
        self.mode1_combo = QComboBox()
        self.mode1_combo.addItems(["novel_v2","novel_v1"])
        self.mode1_combo.setCurrentText(settings.get("MODE1","novel_v2"))
        self.mode2_combo = QComboBox()
        self.mode2_combo.addItems(["text","manga"])
        self.mode2_combo.setCurrentText(settings.get("MODE2","text"))
        f_dual.addRow("Режим области1 (диалоги):", self.mode1_combo)
        f_dual.addRow("Режим области2 (интерфейс):", self.mode2_combo)
        g_dual.setLayout(f_dual)
        layout.addWidget(g_dual)

        g_ocr = QGroupBox("OCR настройки")
        f_ocr = QFormLayout()
        self.manga_ocr_combo = QComboBox()
        self.manga_ocr_combo.addItems(["manga_ocr","easyocr","multi"])
        self.manga_ocr_combo.setCurrentText(settings["MANGA_OCR_BACKEND"])
        f_ocr.addRow("OCR для manga (отдельно):", self.manga_ocr_combo)
        engines_layout = QHBoxLayout()
        self.check_easyocr = QCheckBox("EasyOCR")
        self.check_tesseract = QCheckBox("Tesseract")
        self.check_windows_ocr = QCheckBox("Windows OCR")
        self.check_manga_ocr = QCheckBox("Manga OCR")
        engines = settings.get("NOVEL_OCR_ENGINES", ["easyocr"])
        self.check_easyocr.setChecked("easyocr" in engines)
        self.check_tesseract.setChecked("tesseract" in engines)
        self.check_windows_ocr.setChecked("windows_ocr" in engines)
        self.check_manga_ocr.setChecked("manga_ocr" in engines)
        if not TESSERACT_AVAILABLE: self.check_tesseract.setEnabled(False)
        if not WINDOWS_OCR_AVAILABLE or sys.platform!="win32": self.check_windows_ocr.setEnabled(False)
        if not MANGA_OCR_AVAILABLE: self.check_manga_ocr.setEnabled(False)
        engines_layout.addWidget(self.check_easyocr)
        engines_layout.addWidget(self.check_tesseract)
        engines_layout.addWidget(self.check_windows_ocr)
        engines_layout.addWidget(self.check_manga_ocr)
        self.strategy_combo = QComboBox()
        self.strategy_combo.addItems(["best_confidence","voting"])
        self.strategy_combo.setCurrentText(settings.get("NOVEL_OCR_STRATEGY","best_confidence"))
        f_ocr.addRow("Активные движки (multi):", engines_layout)
        f_ocr.addRow("Стратегия (multi):", self.strategy_combo)
        self.easyocr_conf = QDoubleSpinBox(); self.easyocr_conf.setRange(0,0.9); self.easyocr_conf.setValue(settings.get("EASYOCR_CONFIDENCE",0.2))
        self.tesseract_conf = QDoubleSpinBox(); self.tesseract_conf.setRange(0,0.9); self.tesseract_conf.setValue(settings.get("TESSERACT_CONFIDENCE",0.3))
        self.windows_ocr_conf = QDoubleSpinBox(); self.windows_ocr_conf.setRange(0,0.9); self.windows_ocr_conf.setValue(settings.get("WINDOWS_OCR_CONFIDENCE",0.3))
        self.manga_ocr_conf = QDoubleSpinBox(); self.manga_ocr_conf.setRange(0,0.9); self.manga_ocr_conf.setValue(settings.get("MANGA_OCR_CONFIDENCE",0.4))
        f_ocr.addRow("Порог EasyOCR:", self.easyocr_conf)
        f_ocr.addRow("Порог Tesseract:", self.tesseract_conf)
        f_ocr.addRow("Порог Windows OCR:", self.windows_ocr_conf)
        f_ocr.addRow("Порог Manga OCR:", self.manga_ocr_conf)
        g_ocr.setLayout(f_ocr)
        layout.addWidget(g_ocr)

        g_preproc = QGroupBox("Предобработка изображений для OCR")
        f_preproc = QFormLayout()
        self.check_binarization = QCheckBox("Использовать бинаризацию (ч/б)")
        self.check_binarization.setChecked(settings.get("ENABLE_BINARIZATION", True))
        self.check_inversion = QCheckBox("Использовать инверсию (после бинаризации)")
        self.check_inversion.setChecked(settings.get("ENABLE_INVERSION", True))
        self.bin_thresh = QSpinBox()
        self.bin_thresh.setRange(0,255)
        self.bin_thresh.setValue(settings.get("BINARIZATION_THRESHOLD", 127))
        self.bin_block = QSpinBox()
        self.bin_block.setRange(0,201)
        self.bin_block.setSingleStep(2)
        self.bin_block.setValue(settings.get("BINARIZATION_BLOCK_SIZE", 0))
        self.bin_block.setToolTip("0 = обычная пороговая, нечётное число >0 = адаптивная")
        self.bin_c = QDoubleSpinBox()
        self.bin_c.setRange(0,20)
        self.bin_c.setValue(settings.get("BINARIZATION_C", 2))
        f_preproc.addRow("", self.check_binarization)
        f_preproc.addRow("", self.check_inversion)
        f_preproc.addRow("Порог бинаризации (0-255):", self.bin_thresh)
        f_preproc.addRow("Размер блока адаптивной (0=выкл):", self.bin_block)
        f_preproc.addRow("Константа C (адаптивная):", self.bin_c)
        g_preproc.setLayout(f_preproc)
        layout.addWidget(g_preproc)

        g_auto = QGroupBox("Автоматический режим")
        f_auto = QFormLayout()
        self.auto_interval = QDoubleSpinBox(); self.auto_interval.setRange(0.5,10); self.auto_interval.setValue(settings.get("AUTO_CHECK_INTERVAL",2.0))
        self.sim_thresh = QDoubleSpinBox(); self.sim_thresh.setRange(0.8,1.0); self.sim_thresh.setValue(settings.get("SCREENSHOT_SIMILARITY_THRESHOLD",0.95))
        f_auto.addRow("Интервал проверки (сек):", self.auto_interval)
        f_auto.addRow("Порог схожести:", self.sim_thresh)
        g_auto.setLayout(f_auto)
        layout.addWidget(g_auto)

        g2 = QGroupBox("Параметры перевода и оверлеев")
        f2 = QFormLayout()
        self.context = QSpinBox(); self.context.setRange(1,20); self.context.setValue(settings["CONTEXT_SIZE"])
        self.temp = QDoubleSpinBox(); self.temp.setRange(0,2); self.temp.setValue(settings["TEMPERATURE"])
        self.timeout = QSpinBox(); self.timeout.setRange(10,120); self.timeout.setValue(settings["TIMEOUT"])
        self.yolo_conf = QDoubleSpinBox(); self.yolo_conf.setRange(0.1,0.9); self.yolo_conf.setValue(settings["CONFIDENCE_THRESHOLD"])
        self.width_factor = QDoubleSpinBox(); self.width_factor.setRange(0.5,3); self.width_factor.setValue(settings.get("TEXT_BOX_WIDTH_FACTOR",1.5))
        self.height_factor = QDoubleSpinBox(); self.height_factor.setRange(0.5,3); self.height_factor.setValue(settings.get("TEXT_BOX_HEIGHT_FACTOR",1.5))
        f2.addRow("Размер контекста:", self.context)
        f2.addRow("Температура:", self.temp)
        f2.addRow("Таймаут (сек):", self.timeout)
        f2.addRow("Порог YOLO:", self.yolo_conf)
        f2.addRow("Коэф. ширины текста:", self.width_factor)
        f2.addRow("Коэф. высоты текста:", self.height_factor)
        g2.setLayout(f2)
        layout.addWidget(g2)

        g_async = QGroupBox("Асимметричная обработка")
        f_async = QFormLayout()
        self.async_check = QCheckBox("Включить асимметричную обработку")
        self.async_check.setChecked(settings.get("ASYMMETRIC_TRANSLATION",False))
        self.max_concurrent = QSpinBox(); self.max_concurrent.setRange(1,20); self.max_concurrent.setValue(settings.get("MAX_CONCURRENT_REQUESTS",5))
        f_async.addRow("", self.async_check)
        f_async.addRow("Макс. параллельных:", self.max_concurrent)
        g_async.setLayout(f_async)
        layout.addWidget(g_async)

        g_stream = QGroupBox("Потоковый вывод")
        f_stream = QFormLayout()
        self.stream_check = QCheckBox("Включить потоковый вывод")
        self.stream_check.setChecked(settings.get("STREAMING_OUTPUT",False))
        f_stream.addRow("", self.stream_check)
        g_stream.setLayout(f_stream)
        layout.addWidget(g_stream)

        g_disp = QGroupBox("Отображение")
        f_disp = QFormLayout()
        self.font_combo = QFontComboBox()
        self.font_combo.setCurrentFont(QFont(settings.get("FONT_FAMILY","Arial")))
        self.font_size = QSpinBox(); self.font_size.setRange(8,48); self.font_size.setValue(settings.get("FONT_SIZE",12))
        self.bold_check = QCheckBox("Жирный"); self.bold_check.setChecked(settings.get("FONT_BOLD",False))
        self.outline = QSpinBox(); self.outline.setRange(0,5); self.outline.setValue(settings.get("FONT_OUTLINE",0))
        self.font_color_btn = QPushButton("Цвет шрифта"); self.font_color_btn.setStyleSheet(f"background-color:{settings.get('FONT_COLOR','#ffffff')}")
        self.bg_color_btn = QPushButton("Цвет фона"); self.bg_color_btn.setStyleSheet(f"background-color:{settings.get('BACKGROUND_COLOR','#000000')}")
        self.outline_color_btn = QPushButton("Цвет рамки"); self.outline_color_btn.setStyleSheet(f"background-color:{settings.get('OUTLINE_COLOR','#ff007f')}")
        self.opacity_slider = QSlider(Qt.Orientation.Horizontal); self.opacity_slider.setRange(0,255); self.opacity_slider.setValue(settings.get("BACKGROUND_OPACITY",200))
        self.opacity_label = QLabel(f"{settings.get('BACKGROUND_OPACITY',200)}/255")
        self.opacity_slider.valueChanged.connect(lambda v: self.opacity_label.setText(f"{v}/255"))
        self.relocation_check = QCheckBox("Смещать пересекающиеся блоки")
        self.relocation_check.setChecked(settings.get("ENABLE_OVERLAP_RELOCATION",True))
        f_disp.addRow("Шрифт:", self.font_combo)
        f_disp.addRow("Размер:", self.font_size)
        f_disp.addRow("", self.bold_check)
        f_disp.addRow("Контур:", self.outline)
        f_disp.addRow("Цвет шрифта:", self.font_color_btn)
        f_disp.addRow("Цвет фона:", self.bg_color_btn)
        f_disp.addRow("Прозрачность:", self.opacity_slider)
        f_disp.addRow("", self.opacity_label)
        f_disp.addRow("Цвет рамки:", self.outline_color_btn)
        f_disp.addRow("", self.relocation_check)
        g_disp.setLayout(f_disp)
        layout.addWidget(g_disp)

        g_gpu = QGroupBox("Ускорение")
        f_gpu = QFormLayout()
        self.gpu_ocr = QCheckBox("Использовать GPU для OCR")
        self.gpu_ocr.setChecked(settings.get("USE_GPU_OCR",False))
        self.gpu_model = QCheckBox("Использовать GPU для модели (инфо)")
        self.gpu_model.setChecked(settings.get("USE_GPU_MODEL",False))
        f_gpu.addRow("", self.gpu_ocr)
        f_gpu.addRow("", self.gpu_model)
        g_gpu.setLayout(f_gpu)
        layout.addWidget(g_gpu)

        btn_layout = QHBoxLayout()
        save_btn = QPushButton("Сохранить")
        close_btn = QPushButton("Закрыть")
        save_btn.clicked.connect(self.save_settings)
        close_btn.clicked.connect(self.hide)
        btn_layout.addWidget(save_btn)
        btn_layout.addWidget(close_btn)
        layout.addLayout(btn_layout)

        info = QLabel("📌 Предобработка (бинаризация/инверсия) применяется ко ВСЕМУ скриншоту области целиком.\n"
                      "   Затем каждый вариант обрабатывается OCR, результаты собираются и дедуплицируются.\n"
                      "   В логе отображаются миниатюры обработанных полных скриншотов.")
        info.setStyleSheet("color:#aaa; font-size:10px;")
        layout.addWidget(info)

        self.source_lang_combo.currentTextChanged.connect(self.on_source_lang_changed)
        self.mode_combo.currentTextChanged.connect(self.on_mode_changed)
        self.font_color_btn.clicked.connect(lambda: self.choose_color("font"))
        self.bg_color_btn.clicked.connect(lambda: self.choose_color("bg"))
        self.outline_color_btn.clicked.connect(lambda: self.choose_color("outline"))

    def choose_color(self, target):
        cur = {"font":settings.get("FONT_COLOR","#ffffff"), "bg":settings.get("BACKGROUND_COLOR","#000000"), "outline":settings.get("OUTLINE_COLOR","#ff007f")}[target]
        col = QColorDialog.getColor(QColor(cur), self)
        if col.isValid():
            if target=="font":
                settings["FONT_COLOR"]=col.name()
                self.font_color_btn.setStyleSheet(f"background-color:{col.name()}")
            elif target=="bg":
                settings["BACKGROUND_COLOR"]=col.name()
                self.bg_color_btn.setStyleSheet(f"background-color:{col.name()}")
            else:
                settings["OUTLINE_COLOR"]=col.name()
                self.outline_color_btn.setStyleSheet(f"background-color:{col.name()}")
            self.parent_app.update_style_all()
    def on_source_lang_changed(self, new_source):
        reply = QMessageBox.question(self, "Обновить язык OCR?", "Обновить язык OCR для всех режимов?", QMessageBox.StandardButton.Yes|QMessageBox.StandardButton.No)
        if reply==QMessageBox.StandardButton.Yes:
            new_lang = SOURCE_TO_EASYOCR.get(new_source,"en")
            settings["MANGA_OCR_LANG"]=new_lang
            settings["TEXT_OCR_LANG"]=new_lang
            settings["NOVEL_OCR_LANG"]=new_lang
    def on_mode_changed(self, new_mode):
        self.parent_app.mode_changed_in_settings(new_mode)
    def save_settings(self):
        global settings
        old_mode = settings["MODE"]
        old_gpu = settings.get("USE_GPU_OCR",False)
        old_auto = settings.get("AUTO_CHECK_INTERVAL",2.0)
        settings["LM_STUDIO_URL"]=self.url_edit.text().strip()
        settings["MODEL_NAME"]=self.model_edit.text().strip()
        settings["CONTEXT_SIZE"]=self.context.value()
        settings["TEMPERATURE"]=self.temp.value()
        settings["TIMEOUT"]=self.timeout.value()
        settings["CONFIDENCE_THRESHOLD"]=self.yolo_conf.value()
        settings["AUTO_CHECK_INTERVAL"]=self.auto_interval.value()
        settings["SCREENSHOT_SIMILARITY_THRESHOLD"]=self.sim_thresh.value()
        settings["SOURCE_LANG"]=self.source_lang_combo.currentText()
        settings["TARGET_LANG"]=self.target_lang_combo.currentText()
        settings["MODE"]=self.mode_combo.currentText()
        settings["MANGA_OCR_BACKEND"]=self.manga_ocr_combo.currentText()
        engines=[]
        if self.check_easyocr.isChecked(): engines.append("easyocr")
        if self.check_tesseract.isChecked(): engines.append("tesseract")
        if self.check_windows_ocr.isChecked(): engines.append("windows_ocr")
        if self.check_manga_ocr.isChecked(): engines.append("manga_ocr")
        settings["NOVEL_OCR_ENGINES"]=engines
        settings["MANGA_OCR_ENGINES"]=engines[:]
        settings["NOVEL_OCR_STRATEGY"]=self.strategy_combo.currentText()
        settings["MANGA_OCR_STRATEGY"]=self.strategy_combo.currentText()
        settings["EASYOCR_CONFIDENCE"]=self.easyocr_conf.value()
        settings["TESSERACT_CONFIDENCE"]=self.tesseract_conf.value()
        settings["WINDOWS_OCR_CONFIDENCE"]=self.windows_ocr_conf.value()
        settings["MANGA_OCR_CONFIDENCE"]=self.manga_ocr_conf.value()
        settings["TEXT_BOX_WIDTH_FACTOR"]=self.width_factor.value()
        settings["TEXT_BOX_HEIGHT_FACTOR"]=self.height_factor.value()
        settings["ENABLE_OVERLAP_RELOCATION"]=self.relocation_check.isChecked()
        settings["FONT_FAMILY"]=self.font_combo.currentFont().family()
        settings["FONT_SIZE"]=self.font_size.value()
        settings["FONT_BOLD"]=self.bold_check.isChecked()
        settings["FONT_OUTLINE"]=self.outline.value()
        settings["BACKGROUND_OPACITY"]=self.opacity_slider.value()
        settings["USE_GPU_OCR"]=self.gpu_ocr.isChecked()
        settings["USE_GPU_MODEL"]=self.gpu_model.isChecked()
        settings["ASYMMETRIC_TRANSLATION"]=self.async_check.isChecked()
        settings["MAX_CONCURRENT_REQUESTS"]=self.max_concurrent.value()
        settings["STREAMING_OUTPUT"]=self.stream_check.isChecked()
        settings["MODE1"]=self.mode1_combo.currentText()
        settings["MODE2"]=self.mode2_combo.currentText()
        settings["ENABLE_BINARIZATION"]=self.check_binarization.isChecked()
        settings["ENABLE_INVERSION"]=self.check_inversion.isChecked()
        settings["BINARIZATION_THRESHOLD"]=self.bin_thresh.value()
        settings["BINARIZATION_BLOCK_SIZE"]=self.bin_block.value()
        settings["BINARIZATION_C"]=self.bin_c.value()
        sync_all_ocr_langs()
        if settings["USE_GPU_OCR"] and (not TORCH_AVAILABLE or not torch.cuda.is_available()):
            QMessageBox.warning(self,"GPU недоступен","GPU для OCR недоступен, будет использован CPU.")
            settings["USE_GPU_OCR"]=False
        need_unload = (old_gpu != settings["USE_GPU_OCR"])
        if need_unload:
            unload_all_ocr()
            unload_mode_models(settings["MODE"])
        if old_mode != settings["MODE"]:
            unload_mode_models(settings["MODE"])
            self.parent_app.recreate_interface_for_mode()
        if abs(old_auto - settings["AUTO_CHECK_INTERVAL"])>0.01 and self.parent_app.auto_mode:
            self.parent_app.auto_timer.stop()
            self.parent_app.auto_timer.start(int(settings["AUTO_CHECK_INTERVAL"]*1000))
        save_settings_to_file()
        self.parent_app.update_style_all()
        QMessageBox.information(self,"Сохранено","Настройки сохранены.")

# ----------------------------------------------------------------------
# TranslatorApp (главный)
# ----------------------------------------------------------------------
active_overlays = []

class TranslatorApp(QApplication):
    def __init__(self, argv):
        super().__init__(argv)
        self.setQuitOnLastWindowClosed(False)
        self.capture_frame = None
        self.control_panel = None
        self.novel_v2_output_window = None
        self.selector = None
        self.translation_thread = None
        self.auto_mode = False
        self.auto_timer = None
        self.current_region = None
        self.last_translation_time = None
        self.ocr_log_window = None
        self.log_text = None
        self.dual_frames = {1: None, 2: None}
        self.dual_regions = {
            1: tuple(settings.get("REGION1", [100,100,300,200])),
            2: tuple(settings.get("REGION2", [150,150,400,300]))
        }
        self.active_overlay_widgets = {}
        self.current_overlay_blocks = []
        self.current_mode = None

        # Трей и панель (как ранее, опущено для краткости, но в финале должно быть)
        # Создание иконки трея
        tray_pixmap = QPixmap(64,64)
        tray_pixmap.fill(Qt.GlobalColor.transparent)
        p = QPainter(tray_pixmap)
        p.setBrush(QBrush(QColor(255,0,127)))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawRoundedRect(0,0,64,64,10,10)
        p.setPen(QPen(Qt.GlobalColor.white,3))
        p.setFont(QFont("Arial",32,QFont.Weight.Bold))
        p.drawText(0,0,64,64,Qt.AlignmentFlag.AlignCenter,"M")
        p.end()
        tray_icon = QIcon(tray_pixmap)
        self.tray_icon = QSystemTrayIcon(self)
        self.tray_icon.setIcon(tray_icon)
        self.tray_icon.setToolTip("Переводчик манги / текста (Multi-OCR)")
        tray_menu = QMenu()
        show_action = QAction("Показать панель", self)
        settings_action = QAction("Настройки", self)
        show_log_action = QAction("Показать лог OCR", self)
        exit_action = QAction("Выход", self)
        show_action.triggered.connect(self.show_main_panel)
        settings_action.triggered.connect(self.show_settings)
        show_log_action.triggered.connect(self.show_ocr_log)
        exit_action.triggered.connect(self.exit_app)
        tray_menu.addAction(show_action)
        tray_menu.addAction(settings_action)
        tray_menu.addAction(show_log_action)
        tray_menu.addSeparator()
        tray_menu.addAction(exit_action)
        self.tray_icon.setContextMenu(tray_menu)
        self.tray_icon.show()

        # Главная панель
        self.main_panel = QWidget()
        self.main_panel.setWindowFlags(Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.FramelessWindowHint)
        self.main_panel.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.main_panel.setGeometry(50,50,280,250)
        self.main_panel.setStyleSheet("background-color: rgba(20,20,20,200); border-radius: 10px; border: 1px solid #ff007f;")
        layout = QVBoxLayout(self.main_panel)
        layout.setSpacing(5)
        self.btn_select = QPushButton("🎯 Выбрать область")
        self.btn_select.clicked.connect(lambda: self.start_region_selection(0))
        self.btn_dual_region1 = QPushButton("🔹 Выбрать область 1 (диалоги)")
        self.btn_dual_region1.clicked.connect(lambda: self.start_region_selection(1))
        self.btn_dual_region2 = QPushButton("🔸 Выбрать область 2 (интерфейс)")
        self.btn_dual_region2.clicked.connect(lambda: self.start_region_selection(2))
        btn_settings = QPushButton("⚙ Настройки")
        btn_show_log = QPushButton("📋 Лог OCR")
        btn_tray = QPushButton("▼ Трей")
        btn_exit = QPushButton("🚪 Выход")
        for btn in (self.btn_select, self.btn_dual_region1, self.btn_dual_region2, btn_settings, btn_show_log, btn_tray, btn_exit):
            btn.setStyleSheet("QPushButton { background: #2d2d2d; color: white; border: 1px solid #555; border-radius: 5px; padding: 5px; } QPushButton:hover { background: #ff007f; border-color: #ff007f; }")
        btn_settings.clicked.connect(self.show_settings)
        btn_show_log.clicked.connect(self.show_ocr_log)
        btn_tray.clicked.connect(self.hide_main_panel)
        btn_exit.clicked.connect(self.exit_app)
        layout.addWidget(self.btn_select)
        layout.addWidget(self.btn_dual_region1)
        layout.addWidget(self.btn_dual_region2)
        layout.addWidget(btn_settings)
        layout.addWidget(btn_show_log)
        layout.addWidget(btn_tray)
        layout.addWidget(btn_exit)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet("QProgressBar { border:1px solid #ff007f; border-radius:5px; text-align:center; color:white; background:#2d2d2d; height:20px; } QProgressBar::chunk { background:#ff007f; border-radius:4px; }")
        layout.addWidget(self.progress_bar)
        self.time_label = QLabel("⏱️ Время последнего перевода: --")
        self.time_label.setStyleSheet("color:#ccc; font-size:10px; background:rgba(0,0,0,0.5); padding:2px; border-radius:3px;")
        self.time_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.time_label)
        self.main_panel.show()
        self.settings_window = None
        self.update_dual_buttons_state()
        self.diagnose_ocr()
        print("\n=== 🌀 ПЕРЕВОДЧИК ЗАПУЩЕН (режимы: manga, text, novel_v1, novel_v2, dual) 🌀 ===")
        print("Предобработка (бинаризация/инверсия) применяется ко ВСЕМУ скриншоту области.")
        print("========================================\n")

    def update_dual_buttons_state(self):
        is_dual = (settings["MODE"] == "dual")
        self.btn_dual_region1.setEnabled(is_dual)
        self.btn_dual_region2.setEnabled(is_dual)
        if not is_dual:
            self.close_dual_frames()

    def close_dual_frames(self):
        for i in (1,2):
            if self.dual_frames[i]:
                self.dual_frames[i].close_frame()
                self.dual_frames[i] = None

    def update_dual_regions(self):
        self.dual_regions[1] = tuple(settings["REGION1"])
        self.dual_regions[2] = tuple(settings["REGION2"])

    def mode_changed_in_settings(self, new_mode):
        self.update_dual_buttons_state()
        if new_mode != "dual":
            self.close_dual_frames()
            self.close_current_mode()

    def update_style_all(self):
        for ov in active_overlays:
            if hasattr(ov,'update_style_from_settings'):
                ov.update_style_from_settings()
            ov.update()
        for i in (1,2):
            if self.dual_frames[i]:
                self.dual_frames[i].update_appearance_from_settings()
        if self.capture_frame:
            self.capture_frame.update_appearance_from_settings()
        if hasattr(self,'log_text') and self.log_text:
            font = QFont("Consolas",10)
            if settings.get("FONT_BOLD",False):
                font.setBold(True)
            self.log_text.setFont(font)

    def start_region_selection(self, region_id=0):
        if region_id != 0 and settings["MODE"] != "dual":
            QMessageBox.warning(None, "Недоступно", "Выбор областей доступен только в dual режиме.\nВыберите режим 'dual' в настройках.")
            return
        if self.selector:
            self.selector.close()
        self.selector = RegionSelector(self, region_id=region_id)
        self.selector.show()

    def accept_single_region(self, rect):
        self.current_region = (rect.x(), rect.y(), rect.width(), rect.height())
        self.create_capture_frame(*self.current_region)
        if self.selector:
            self.selector.close()
            self.selector = None

    def accept_dual_region(self, rect, region_id):
        x,y,w,h = rect.x(), rect.y(), rect.width(), rect.height()
        self.dual_regions[region_id] = (x,y,w,h)
        if region_id == 1:
            settings["REGION1"] = [x,y,w,h]
        else:
            settings["REGION2"] = [x,y,w,h]
        save_settings_to_file()
        if self.dual_frames[region_id] is None:
            self.dual_frames[region_id] = CaptureFrame(x,y,w,h, self, region_id=region_id)
        else:
            self.dual_frames[region_id].setGeometry(x,y,w,h)
            self.dual_frames[region_id].show()
            if self.dual_frames[region_id].control_panel:
                self.dual_frames[region_id].control_panel.update_position()
        if self.selector:
            self.selector.close()
            self.selector = None

    def create_capture_frame(self, x,y,w,h):
        self.current_region = (x,y,w,h)
        self.capture_frame = CaptureFrame(x,y,w,h, self, region_id=0)
        if self.auto_mode:
            self.toggle_auto_mode()
        print(f"📐 Область создана: {x},{y},{w},{h}")

    def update_region(self, x,y,w,h):
        self.current_region = (x,y,w,h)

    def reset_frame(self):
        if self.capture_frame:
            self.capture_frame.close_frame()
            self.capture_frame = None
        self.current_region = None

    def clear_overlays(self):
        global active_overlays
        for ov in active_overlays:
            ov.close()
            ov.deleteLater()
        active_overlays.clear()
        self.active_overlay_widgets.clear()
        self.current_overlay_blocks = []
        QApplication.processEvents()

    def close_current_mode(self):
        if self.capture_frame:
            self.capture_frame.close_frame()
            self.capture_frame = None
        self.clear_overlays()

    def manual_translate_for_region(self, region_id, rect, mode, capture_frame):
        x,y,w,h = rect.x(), rect.y(), rect.width(), rect.height()
        if self.translation_thread and self.translation_thread.isRunning():
            return
        if mode in ("novel_v1","novel_v2") and capture_frame:
            capture_frame.clear_text()
            if capture_frame.output_window:
                capture_frame.output_window.clear_text()
        else:
            self.clear_overlays()
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.translation_thread = TranslationThread((x,y,w,h), mode, capture_frame, region_id)
        self.translation_thread.prepare_overlays.connect(self.on_prepare_overlays)
        self.translation_thread.update_overlay_text.connect(self.on_update_overlay_text)
        self.translation_thread.update_novel_output.connect(lambda text: self.on_update_novel_output(text, capture_frame))
        self.translation_thread.progress.connect(self.update_progress)
        self.translation_thread.finished.connect(lambda ov,el,ex: self.on_translation_finished(ov,el,ex, capture_frame, mode))
        self.translation_thread.start()

    def update_progress(self, current, total):
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)

    def on_prepare_overlays(self, blocks_data, indices, mode):
        self.current_overlay_blocks = blocks_data
        self.current_mode = mode
        self.clear_overlays()
        temp_texts = {i: "..." for i in indices}
        temp_overlays = [(x,y,w,h, temp_texts[i]) for i, (x,y,w,h,_) in enumerate(blocks_data)]
        if mode in ("manga", "text"):
            groups = group_overlays(temp_overlays)
            for group in groups:
                comp = CompositeOverlayMangaGroup(group)
                global_to_local = {}
                for local_pos, (gx,gy,gw,gh,_) in enumerate(group):
                    for gidx, (bx,by,bw,bh,_) in enumerate(blocks_data):
                        if gx==bx and gy==by and gw==bw and gh==bh:
                            global_to_local[gidx]=local_pos
                            break
                for gidx in global_to_local:
                    self.active_overlay_widgets[gidx] = comp
                comp.set_block_texts({gidx:"..." for gidx in global_to_local}, global_to_local)
                active_overlays.append(comp)
        else:
            font = QFont(settings.get("FONT_FAMILY","Arial"), settings.get("FONT_SIZE",12))
            if settings.get("ENABLE_OVERLAP_RELOCATION",True):
                temp_overlays = resolve_overlaps(temp_overlays, font)
            else:
                expanded = []
                for (x,y,w,h,text) in temp_overlays:
                    fw,fh = get_text_block_size(text, font)
                    cx,cy = x+w//2, y+h//2
                    expanded.append((cx-fw//2, cy-fh//2, fw, fh, text))
                temp_overlays = expanded
            groups = group_overlays(temp_overlays)
            for group in groups:
                comp = CompositeOverlay(group)
                global_to_local = {}
                for local_pos, (gx,gy,gw,gh,_) in enumerate(group):
                    for gidx, (bx,by,bw,bh,_) in enumerate(blocks_data):
                        if gx==bx and gy==by and gw==bw and gh==bh:
                            global_to_local[gidx]=local_pos
                            break
                for gidx in global_to_local:
                    self.active_overlay_widgets[gidx] = comp
                comp.set_block_texts({gidx:"..." for gidx in global_to_local}, global_to_local)
                active_overlays.append(comp)
        QApplication.processEvents()

    def on_update_overlay_text(self, idx, text):
        w = self.active_overlay_widgets.get(idx)
        if w:
            w.update_block_text(idx, text)
        else:
            if not hasattr(self,'_partial_texts'):
                self._partial_texts = {}
            self._partial_texts[idx] = text
            self.rebuild_overlays()

    def rebuild_overlays(self):
        if not hasattr(self,'current_overlay_blocks') or not self.current_overlay_blocks:
            return
        if not hasattr(self,'_partial_texts'):
            self._partial_texts = {}
        full = []
        for i, (x,y,w,h,orig) in enumerate(self.current_overlay_blocks):
            txt = self._partial_texts.get(i, "...")
            full.append((x,y,w,h,txt))
        self.clear_overlays()
        if self.current_mode in ("manga", "text"):
            groups = group_overlays(full)
            for group in groups:
                comp = CompositeOverlayMangaGroup(group)
                global_to_local = {}
                for local_pos, (gx,gy,gw,gh,_) in enumerate(group):
                    for gidx, (bx,by,bw,bh,_) in enumerate(self.current_overlay_blocks):
                        if gx==bx and gy==by and gw==bw and gh==bh:
                            global_to_local[gidx]=local_pos
                            break
                for gidx in global_to_local:
                    self.active_overlay_widgets[gidx] = comp
                comp.set_block_texts({gidx:self._partial_texts.get(gidx,"...") for gidx in global_to_local}, global_to_local)
                active_overlays.append(comp)
        else:
            font = QFont(settings.get("FONT_FAMILY","Arial"), settings.get("FONT_SIZE",12))
            if settings.get("ENABLE_OVERLAP_RELOCATION",True):
                full = resolve_overlaps(full, font)
            else:
                expanded = []
                for (x,y,w,h,text) in full:
                    fw,fh = get_text_block_size(text, font)
                    cx,cy = x+w//2, y+h//2
                    expanded.append((cx-fw//2, cy-fh//2, fw, fh, text))
                full = expanded
            groups = group_overlays(full)
            for group in groups:
                comp = CompositeOverlay(group)
                global_to_local = {}
                for local_pos, (gx,gy,gw,gh,_) in enumerate(group):
                    for gidx, (bx,by,bw,bh,_) in enumerate(self.current_overlay_blocks):
                        if gx==bx and gy==by and gw==bw and gh==bh:
                            global_to_local[gidx]=local_pos
                            break
                for gidx in global_to_local:
                    self.active_overlay_widgets[gidx] = comp
                comp.set_block_texts({gidx:self._partial_texts.get(gidx,"...") for gidx in global_to_local}, global_to_local)
                active_overlays.append(comp)
        QApplication.processEvents()

    def on_update_novel_output(self, text, capture_frame):
        if capture_frame:
            if capture_frame.output_window is None:
                capture_frame.output_window = NovelV2OutputWindow(self, capture_frame)
            capture_frame.output_window.set_text(text)
        else:
            if self.novel_v2_output_window is None:
                self.novel_v2_output_window = NovelV2OutputWindow(self, None)
            self.novel_v2_output_window.set_text(text)

    def on_translation_finished(self, overlays, elapsed, extra, capture_frame, mode):
        if self.translation_thread:
            self.translation_thread.quit()
            self.translation_thread.wait()
            self.translation_thread = None
        self.last_translation_time = elapsed
        self.time_label.setText(f"⏱️ Время: {elapsed:.2f} сек")
        if mode in ("manga","text") and not settings.get("STREAMING_OUTPUT",False) and overlays:
            self.clear_overlays()
            self.current_mode = mode
            if mode in ("manga","text"):
                groups = group_overlays(overlays)
                for group in groups:
                    active_overlays.append(CompositeOverlayMangaGroup(group))
            else:
                font = QFont(settings.get("FONT_FAMILY","Arial"), settings.get("FONT_SIZE",12))
                if settings.get("ENABLE_OVERLAP_RELOCATION",True):
                    overlays = resolve_overlaps(overlays, font)
                else:
                    expanded = []
                    for (x,y,w,h,text) in overlays:
                        fw,fh = get_text_block_size(text, font)
                        cx,cy = x+w//2, y+h//2
                        expanded.append((cx-fw//2, cy-fh//2, fw, fh, text))
                    overlays = expanded
                groups = group_overlays(overlays)
                for group in groups:
                    active_overlays.append(CompositeOverlay(group))
        elif mode in ("novel_v1","novel_v2") and not settings.get("STREAMING_OUTPUT",False) and overlays:
            _,_,_,_,txt = overlays[0]
            if mode == "novel_v2":
                if capture_frame:
                    if capture_frame.output_window is None:
                        capture_frame.output_window = NovelV2OutputWindow(self, capture_frame)
                    capture_frame.output_window.set_text(txt)
                else:
                    if self.novel_v2_output_window is None:
                        self.novel_v2_output_window = NovelV2OutputWindow(self, None)
                    self.novel_v2_output_window.set_text(txt)
            elif mode == "novel_v1" and capture_frame:
                capture_frame.set_translated_text(txt)
        self.progress_bar.setVisible(False)
        if hasattr(self,'_partial_texts'):
            del self._partial_texts

    def show_ocr_log(self):
        if self.ocr_log_window is None or not self.ocr_log_window.isVisible():
            self.ocr_log_window = QWidget()
            self.ocr_log_window.setWindowTitle("Лог OCR (с миниатюрами)")
            self.ocr_log_window.setWindowFlags(Qt.WindowType.WindowStaysOnTopHint)
            self.ocr_log_window.resize(900, 700)
            layout = QVBoxLayout(self.ocr_log_window)
            splitter = QSplitter(Qt.Orientation.Vertical)
            self.log_text = QTextEdit()
            self.log_text.setReadOnly(True)
            self.log_text.setFont(QFont("Consolas", 10))
            splitter.addWidget(self.log_text)
            self.image_list = QListWidget()
            self.image_list.setIconSize(QSize(150, 150))
            self.image_list.setResizeMode(QListWidget.ResizeMode.Adjust)
            self.image_list.setViewMode(QListWidget.ViewMode.IconMode)
            self.image_list.setMovement(QListWidget.Movement.Static)
            splitter.addWidget(self.image_list)
            layout.addWidget(splitter)
            btn_refresh = QPushButton("Обновить")
            btn_refresh.clicked.connect(self.refresh_ocr_log_with_images)
            layout.addWidget(btn_refresh)
            self.refresh_ocr_log_with_images()
        self.ocr_log_window.show()
        self.ocr_log_window.raise_()

    def refresh_ocr_log_with_images(self):
        if hasattr(self, 'log_text'):
            self.log_text.clear()
            self.image_list.clear()
            log_entries = get_ocr_log()
            for entry, pixmap, ts in log_entries:
                time_str = time.strftime('%H:%M:%S', time.localtime(ts))
                self.log_text.append(f"[{time_str}] {entry}")
                if pixmap is not None and not pixmap.isNull():
                    item = QListWidgetItem()
                    item.setIcon(QIcon(pixmap))
                    item.setToolTip(entry[:100])
                    self.image_list.addItem(item)

    def diagnose_ocr(self):
        print("\n🔍 ДИАГНОСТИКА:")
        print(f"   Режим: {settings['MODE']}")
        print(f"   Исходный язык: {settings['SOURCE_LANG']} -> OCR: {get_ocr_lang_from_source()}")
        print(f"   Dual MODE1: {settings.get('MODE1')}, MODE2: {settings.get('MODE2')}")
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
    def exit_app(self):
        self.clear_overlays()
        self.close_current_mode()
        self.close_dual_frames()
        if self.selector:
            self.selector.close()
        if self.settings_window:
            self.settings_window.close()
        if self.ocr_log_window:
            self.ocr_log_window.close()
        self.main_panel.close()
        self.tray_icon.hide()
        self.quit()
    def recreate_interface_for_mode(self):
        self.close_current_mode()
        self.close_dual_frames()
        self.current_region = None
        self.update_dual_buttons_state()

# ----------------------------------------------------------------------
# Запуск
# ----------------------------------------------------------------------
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