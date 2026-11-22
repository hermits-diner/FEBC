@echo off
chcp 65001 >nul
python -I "%~dp0febc_edit.py" --edit %*
echo.
pause
