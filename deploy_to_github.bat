@echo off
REM CPUWARP-ML GitHub Deployment Script
REM =====================================

echo ============================================================
echo      CPUWARP-ML GitHub Deployment Setup
echo ============================================================
echo.

REM Check if git is available
where git >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Git is not found in PATH!
    echo.
    echo Please ensure Git is installed and added to PATH.
    echo You can download Git from: https://git-scm.com/download/win
    echo.
    echo After installing, restart this script.
    pause
    exit /b 1
)

echo [OK] Git found!
git --version
echo.

REM Initialize git repository if not already initialized
if not exist ".git" (
    echo Initializing Git repository...
    git init
    echo.
)

REM Configure git (you'll need to update these)
echo Setting up Git configuration...
echo Please enter your GitHub username:
set /p github_username=
echo.
echo Please enter your GitHub email:
set /p github_email=

git config user.name "%github_username%"
git config user.email "%github_email%"

echo.
echo Git configured for user: %github_username%
echo.

REM Add all files
echo Adding files to Git...
git add .
echo.

REM Create initial commit
echo Creating initial commit...
git commit -m "Initial commit: CPUWARP-ML - High-Performance CPU-Optimized ML Training Framework"
echo.

echo ============================================================
echo     NEXT STEPS TO DEPLOY TO GITHUB:
echo ============================================================
echo.
echo 1. Go to GitHub.com and create a new repository:
echo    https://github.com/new
echo.
echo 2. Name it: cpuwarp-ml
echo    Description: High-Performance CPU-Optimized ML Training Framework
echo    Make it PUBLIC so others can use it!
echo.
echo 3. After creating the repository, GitHub will show you commands.
echo    Run these commands in this folder:
echo.
echo    git remote add origin https://github.com/%github_username%/cpuwarp-ml.git
echo    git branch -M main
echo    git push -u origin main
echo.
echo ============================================================
echo     Your repository will be available at:
echo     https://github.com/%github_username%/cpuwarp-ml
echo ============================================================
echo.
pause