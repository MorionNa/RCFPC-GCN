@echo off

REM 升级 pip
.\blender-2.92.0-windows64\2.92\python\bin\python.exe -m pip install --upgrade pip

.\blender-2.92.0-windows64\2.92\python\bin\python.exe -m pip install openpyxl==3.1.2 --target=.\blender-2.92.0-windows64\2.92\python\lib\site-packages
.\blender-2.92.0-windows64\2.92\python\bin\python.exe -m pip install numpy==1.17.5 --target=.\blender-2.92.0-windows64\2.92\python\lib\site-packages
.\blender-2.92.0-windows64\2.92\python\bin\python.exe -m pip install pandas==1.3.5 --target=.\blender-2.92.0-windows64\2.92\python\lib\site-packages

