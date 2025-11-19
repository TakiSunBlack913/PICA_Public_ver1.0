@echo off
REM スクリプトが置かれているディレクトリに移動
REM %~dp0 はバッチファイル自身のドライブ名とパスを意味します
cd /d "%~dp0"

REM Pythonスクリプトを実行
REM python コマンドがうまく動かない場合は python3 に変更してください
python main_hub.py

REM 実行後にウィンドウが閉じないように一時停止
pause