@echo off
echo Setting up Python Virtual Environment...

:: Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python is not found in your PATH. 
    echo If you use Anaconda, please open Anaconda Prompt, navigate to this folder, and run:
    echo   conda create -n ct_env python=3.10
    echo   conda activate ct_env
    echo   pip install -r requirements.txt
    echo   python main.py
    pause
    exit /b
)

:: Create virtual environment if it doesn't exist
if not exist venv (
    echo Creating new virtual environment 'venv'...
    python -m venv venv
)

:: Activate and install requirements
echo Activating virtual environment...
call venv\Scripts\activate

echo Installing requirements...
pip install -r requirements.txt

echo Running application...
python main.py

pause
