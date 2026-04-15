@echo off
:: 設定編碼為 UTF-8，確保繁體中文正常顯示
chcp 65001 > nul

echo =========================================
echo       GitHub 自動推送小幫手 (Git Push)
echo =========================================
echo.

:: 1. 顯示目前狀態 (可選，讓你知道改了哪些東西)
echo [目前變更狀態]
git status -s
echo.

:: 2. 將所有變更加入暫存區
echo [1/3] 正在將所有變更加入暫存區 (git add .)...
git add .
echo.

:: 3. 提示輸入 Commit 訊息
set /p commit_msg="請輸入 Commit 訊息 (直接按 Enter 預設為 'Auto update'): "

:: 如果沒有輸入，給予預設值
if "%commit_msg%"=="" set commit_msg=Auto update

echo.
echo [2/3] 正在提交變更 (git commit)...
git commit -m "%commit_msg%"
echo.

:: 4. 推送到遠端
echo [3/3] 正在推送到 GitHub (git push)...
git push

echo.
echo =========================================
echo   ✅ 執行完畢！
echo =========================================
:: 暫停視窗，讓你確認有沒有報錯，按任意鍵才會關閉
pause