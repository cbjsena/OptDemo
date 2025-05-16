@echo off
REM Create a Python virtual environment and activate it

REM Define the Python executable path
set PYTHON_EXE="C:\Users\2007513\AppData\Local\Programs\Python\Python311\python.exe"
REM Current working directory (project root)
set PROJECT_ROOT=%CD%
REM Full path to the virtual environment
set VENV_DIR=.venv
set VENV_PATH=%PROJECT_ROOT%\%VENV_DIR%
REM Path to the VS Code settings directory
set VSCODE_SETTINGS_DIR=%PROJECT_ROOT%\.vscode
REM Path to the VS Code settings file
set SETTINGS_JSON=%VSCODE_SETTINGS_DIR%\settings.json


echo Python execute file:     %PYTHON_EXE%
echo Project Root:            %PROJECT_ROOT%
echo Virtual environment path:%VENV_PATH%
echo VS Code settings file:   %SETTINGS_JSON%
echo.


REM Check if Python executable exists
if not exist "%PYTHON_EXE%" (
   echo ERROR: Python executable not found at "%PYTHON_EXE%".
   echo Please set the PYTHON_EXE variable correctly at the top of the script.
   pause
   exit /b 1
)


REM Check if the virtual environment folder already exists
if exist "%VENV_PATH%" (
   echo.
   echo Existing virtual environment folder "%VENV_PATH%" found.
   set /p choice="Do you want to delete it and create a new one? (Y/N): "
   if /i "%choice%"=="Y" (
      echo Deleting existing virtual environment folder...
      rmdir /s /q "%VENV_PATH%"
      echo Existing virtual environment folder deleted.
   )else (
      echo Keeping the existing virtual environment folder.
      echo Ensure "%SETTINGS_JSON%" is configured correctly if you want VS Code to use this environment.
      pause
      exit /b 0
   )
)

echo Creating virtual environment in %VENV_PATH%...
%PYTHON_EXE% -m venv %VENV_PATH%

echo.
echo Virtual environment created successfully!

REM Create .vscode directory if it doesn't exist
if not exist "%VSCODE_SETTINGS_DIR%" (
   echo Creating ".vscode" directory...
   mkdir "%VSCODE_SETTINGS_DIR%"
   if errorlevel 1 (
      echo ERROR: Failed to create ".vscode" directory.
      pause
      exit /b 1
   )
)

echo.
echo Creating/Updating VS Code settings file: "%SETTINGS_JSON%"
REM This will overwrite the settings.json file.
REM VS Code handles relative paths from the workspace root and prefers forward slashes.
(
   echo {
   echo  "terminal.integrated.defaultProfile.windows": "Command Prompt",
   echo  "python.defaultInterpreterPath": "%VENV_DIR%/Scripts/python.exe"
   echo }
) > "%SETTINGS_JSON%"
    
echo --- IMPORTANT ---
echo 1. Ensure the Microsoft Python extension is installed in VS Code.
echo 2. If VS Code was already open, you might need to Reload Window (Ctrl+Shift+P, type "Reload Window") or restart VS Code for settings to take full effect.
echo 3. Check Ctrl+Shift+P -> "Python: Select Interpreter" to ensure the correct interpreter is selected(.venv\Scripts\python.exe).
echo   If you don't see the virtual environment in the list, select "Enter interpreter path" and browse to:
echo 4. When you open a NEW terminal (Ctrl+Shift+`) in VS Code,
echo   it should now automatically use this virtual environment.
echo.
echo For manual activation in an external command prompt:
echo   cd /d "%PROJECT_ROOT%"
echo   REM CMD:
echo   call "%VENV_DIR%\Scripts\activate.bat"
echo   REM PowerShell (Run the following if you get an execution policy error):
echo   REM Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
echo   REM PowerShell:
echo   .\%VENV_DIR%\Scripts\Activate.ps1
echo.
pause
exitit /b 0