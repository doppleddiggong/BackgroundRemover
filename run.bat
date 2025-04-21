@echo off
chcp 65001 > nul
title 배경 제거 프로그램

echo 배경 제거 프로그램을 시작합니다...
echo.

:: Python이 설치되어 있는지 확인
python --version > nul 2>&1
if errorlevel 1 (
    echo Python이 설치되어 있지 않습니다.
    echo https://www.python.org/downloads/ 에서 Python을 다운로드하여 설치해주세요.
    echo.
    pause
    exit
)

:: 필요한 패키지 설치 확인 및 설치
echo 필요한 패키지를 확인하고 설치합니다...
python -m pip install --upgrade pip
pip install PyQt6 Pillow torch transparent-background

:: 프로그램 실행
echo.
echo 프로그램을 실행합니다...
python main.py

pause 