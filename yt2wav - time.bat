@echo off
:: 強迫命令提示字元使用 UTF-8 編碼，解決中文亂碼問題
chcp 65001 >nul

:: 將提示文字與輸入放在同一行，畫面更乾淨
set /p url="Please paste the URL : "

:: 自動移除使用者可能不小心貼上的雙引號
set url=%url:"=%

:: 讓您輸入想截取的開始與結束時間
echo.
echo [Time Format Example] 35 (35秒), 1:15 (1分15秒), 0 (開頭)
set /p start_time="Enter START time : "
set /p end_time="Enter END time   : "

:: 執行 yt-dlp 並加入 --download-sections 參數
yt-dlp ^
-x ^
--audio-format wav ^
--download-sections "*%start_time%-%end_time%" ^
-o "%%(title)s.%%(ext)s" ^
"%url%"

echo.
echo Done! Please press any key to exit.
pause >nul