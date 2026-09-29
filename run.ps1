# MOTION DX PowerShell Launcher
Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host " MOTION DX: Objective movement measurement using computer vision" -ForegroundColor Green
Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Launching clinical prototype interface..." -ForegroundColor Yellow
py -3.13 -m streamlit run app.py
