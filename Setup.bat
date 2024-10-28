@echo off
REM 设置虚拟环境的目录
set "VENV_DIR=venv"

REM 创建虚拟环境
python -m venv "%VENV_DIR%"

REM 激活虚拟环境
call "%VENV_DIR%\Scripts\activate.bat"

REM 升级 pip
python -m pip install --upgrade pip

REM 安装依赖库
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install  dgl -f https://data.dgl.ai/wheels/cu121/repo.html
pip install -r requirements.txt
REM 提示完成
echo The virtual environment has been created, and the libraries from requirements.txt have been installed.

pause