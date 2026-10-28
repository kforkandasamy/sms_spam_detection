@echo off
echo ============================================================
echo   Push BAMAT207 SMS Spam Detection Project to GitHub
echo ============================================================
echo.
set /p USERNAME="Enter your GitHub Username: "
set /p REPONAME="Enter your Repository Name (default: sms-spam-detection): "
if "%REPONAME%"=="" set REPONAME=sms-spam-detection

echo.
echo [*] Setting remote origin to https://github.com/%USERNAME%/%REPONAME%.git ...
git remote remove origin 2>nul
git remote add origin https://github.com/%USERNAME%/%REPONAME%.git
git branch -M main

echo [*] Pushing code to GitHub...
git push -u origin main

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ============================================================
    echo   SUCCESS! Project pushed to:
    echo   https://github.com/%USERNAME%/%REPONAME%
    echo.
    echo   To enable GitHub Pages for free live deployment:
    echo   1. Go to https://github.com/%USERNAME%/%REPONAME%/settings/pages
    echo   2. Under 'Build and deployment', set Branch to 'main'
    echo   3. Click Save!
    echo   Your live web app will be at:
    echo   https://%USERNAME%.github.io/%REPONAME%/
    echo ============================================================
) else (
    echo.
    echo [!] Push failed. Please make sure you created the repo on https://github.com/new first and signed in.
)
pause
