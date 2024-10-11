@echo off
for /F "delims=" %%i in (_clustering_/requirements.txt) do (
    echo Installing %%i
    python -m pip install %%i --ignore-installed || echo Failed to install %%i
)
