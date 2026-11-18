@echo off
title DermAI Chrome Dashboard
cd /d "C:\Users\Admin\Documents\DermAI_Milestone2_Dataset"
echo Starting DermAI dashboard...
py -3.15 -m uvicorn web_app.app:app --host 127.0.0.1 --port 8000
pause