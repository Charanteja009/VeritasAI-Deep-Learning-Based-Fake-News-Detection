@echo off
title VeritasAI - Fake News Detection Server
echo ==============================================================
echo       Starting VeritasAI Fake News Detection Web App
echo ==============================================================

if exist "C:\Users\chara\miniconda3\python.exe" (
    "C:\Users\chara\miniconda3\python.exe" run.py
) else (
    python run.py
)

pause
