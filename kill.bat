@echo off
:: Ports to kill processes on
set ports=3000 8000 11434

:: Loop through each port and kill the associated processes
for %%P in (%ports%) do (
    echo Checking port %%P...
    for /f "tokens=5" %%A in ('netstat -ano ^| findstr :%%P') do (
        echo Killing process with PID %%A on port %%P...
        taskkill /PID %%A /F >nul 2>&1
        if %errorlevel%==0 (
            echo Successfully killed process %%A.
        ) else (
            echo Failed to kill process %%A or no process found on port %%P.
        )
    )
)

echo Done!
pause