@echo off

REM 激活虚拟环境
call %~dp0venv\scripts\activate.bat
REM 执行 Python 脚本
python prediction.py
set "BLENDER_PATH=%~dp0blender-2.92\blender.exe"
set "BLEND_FILE=%~dp0Visualization.blend"
"%BLENDER_PATH%" "%BLEND_FILE%"