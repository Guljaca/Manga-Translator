@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo ============================================================
echo   Manga Translator - Setup and Launch
echo ============================================================
echo.

:: Step 1: Check Python
echo [1/5] Checking Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found.
    echo Please install Python 3.8+ from python.org
    echo Make sure to check "Add Python to PATH"
    exit /b 1
)
python --version
echo.

:: Step 2: Upgrade pip
echo [2/5] Upgrading pip...
python -m pip install --upgrade pip
echo.

:: Step 3: Optional PyTorch with CUDA
echo [3/5] PyTorch setup (optional for OCR acceleration)
echo ------------------------------------------------------------
echo NOTE: If you already have PyTorch installed (CPU or CUDA),
echo you can skip this step by answering N.
echo.
echo PyTorch is used for OCR acceleration (YOLO, EasyOCR, MangaOCR).
echo Without CUDA, OCR works but slower.
echo Translation via LM Studio does NOT depend on this.
echo.
echo Installing CUDA version requires NVIDIA GPU and ~2.5 GB space.
echo.
set /p cuda_choice="Install / reinstall PyTorch with CUDA? (Y/N, default N): "
if /i "!cuda_choice!"=="Y" (
    echo Installing PyTorch with CUDA 11.8...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    if errorlevel 1 (
        echo WARNING: CUDA installation failed. Installing CPU version instead.
        pip install torch
    )
) else (
    echo Skipping PyTorch installation. Keeping existing installation.
)
echo.

:: Step 4: Install other dependencies
echo [4/5] Installing other libraries...
echo This may take several minutes.
echo.
pip install requests pyautogui opencv-python numpy Pillow PyQt6 pytesseract easyocr ultralytics transformers sentencepiece manga-ocr
echo.

:: Step 5: Run the program
echo [5/5] Launching Manga_Translator.py
echo ============================================================
if not exist "Manga_Translator.py" (
    echo ERROR: Manga_Translator.py not found in current folder.
    echo Current folder: %cd%
    exit /b 1
)
python Manga_Translator.py

echo.
echo Program finished.