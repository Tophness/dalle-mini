@echo off
reg query HKEY_CURRENT_USER\SOFTWARE\Python\ContinuumAnalytics > tmpFile.txt
for /F "delims=" %%A in (tmpFile.txt) DO (set B=%%A)
del tmpFile.txt
SET "C=\InstallPath"
reg query %B%%C% /ve > tmpFile.txt
for /F "tokens=3" %%D in (tmpFile.txt) DO (SET "adir=%%D")
del tmpFile.txt
SET "abat=\Scripts\activate.bat"
SET "conda=%adir%%abat%"
%windir%\System32\cmd.exe /k "%conda% && conda create --name dallemini python=3.7 && conda activate dallemini && pip install https://whls.blob.core.windows.net/unstable/cuda111/jaxlib-0.3.14+cuda11.cudnn82-cp37-none-win_amd64.whl && pip install dalle-mini && pip install git+https://github.com/patil-suraj/vqgan-jax.git"