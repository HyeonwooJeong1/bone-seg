@echo off
echo ============================================================
echo  3D CT Bone Visualizer + AI 뼈 분할  설치/실행
echo ============================================================
echo.

:: Python 확인
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [오류] Python이 PATH에 없습니다. Python 3.11 설치 후 다시 실행하세요.
    echo   https://www.python.org/downloads/  (설치 시 "Add to PATH" 체크)
    pause
    exit /b
)

:: 가상환경 생성
if not exist venv (
    echo 가상환경 'venv' 생성 중...
    python -m venv venv
)

echo 가상환경 활성화...
call venv\Scripts\activate

:: AI 포함 의존성 설치 (torch 등 최초 1회 다운로드 ~2.5GB, 시간 소요)
echo 의존성 설치 중... (최초 1회 torch 다운로드로 수 분 걸릴 수 있음)
pip install -r requirements_ai.txt

echo.
echo 애플리케이션 실행...
python main.py

pause
