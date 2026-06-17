@echo off

setlocal

cd /d "%~dp0"



for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8020.*LISTENING"') do taskkill /PID %%a /F >nul 2>&1



where python >nul 2>&1 || (

  echo ERROR: Python not found on PATH.

  echo Install Python 3 and retry, or run: py -3 manual_register_server.py config/GP04.json --session pre-droplet

  pause

  exit /b 1

)



echo Using Python:

python --version

where python



python -c "import flask" 2>nul || (

  echo.

  echo Installing dependencies ^(flask, tifffile, Pillow^)...

  python -m pip install flask tifffile Pillow

  if errorlevel 1 (

    echo pip install failed.

    pause

    exit /b 1

  )

)



echo.

echo Starting TIFF manual registration server ...

echo Config: config\GP04.json  session: pre-droplet

echo.

echo IMPORTANT: Keep this window open while using the tool.

echo Opening browser now (server loads scene in background)...
start "" "http://127.0.0.1:8020/endpoints"
echo.
echo If the page stays on Loading, wait 1-3 min — watch this window for progress.
echo Manual URL: http://127.0.0.1:8020/endpoints
echo.



python manual_register_server.py config/GP04.json --session pre-droplet



pause

