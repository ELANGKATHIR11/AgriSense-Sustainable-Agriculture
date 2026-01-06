@echo off
cd /d "F:\AGRISENSEFULL-STACK\AGRISENSEFULL-STACK\src\backend"
.\.venv\Scripts\python.exe -m uvicorn main:app --host 0.0.0.0 --port 8004
pause
