import tkinter as tk
from tkinter import filedialog
import os

def open_finder_dialog():
    """
    ボタンが押されたときに実行される関数。
    ファイル選択ダイアログを開き、選択されたファイルのパスを取得する。
    """
    print("ファイル選択ダイアログを開きます...")
    
    # 選択できるファイルの種類と拡張子を定義
    # (例: テキストファイルとすべてのファイル)
    file_types = [
        ("テキストファイル", "*.txt"), 
        ("Pythonファイル", "*.py"), 
        ("すべてのファイル", "*.*")
    ]
    
    # ダイアログが表示されたときの初期ディレクトリを指定
    # 例として、デスクトップを初期ディレクトリに設定します。
    # macOS/Linuxの場合のホームディレクトリ (~/Desktop)
    # Windowsの場合のデスクトップ (C:\Users\UserName\Desktop)
    initial_dir = os.path.expanduser("~/Desktop") 

    # ファイル選択ダイアログを表示
    # 選択されたファイルの絶対パス（文字列）が返されます
    file_path = filedialog.askopenfilename(
        title="ファイルを選択してください", # ダイアログのタイトル
        filetypes=file_types,            # フィルター
        initialdir=initial_dir           # 初期ディレクトリ
    )

    # パスが取得できたかどうか（ユーザーがキャンセルしなかったか）をチェック
    if file_path:
        print("\n--- 選択されたファイル情報 ---")
        print(f"**ファイルパス:** {file_path}")
        print(f"**ファイル名:** {os.path.basename(file_path)}")
        
        # 実際にファイルを読み込む場合は、この下に処理を記述します。
        # 例: with open(file_path, 'r', encoding='utf-8') as f: ...
    else:
        print("\nファイル選択がキャンセルされました。")

# 1. Tkinterのルートウィンドウを作成
root = tk.Tk()
root.title("ファイル選択")
root.geometry("400x150") # ウィンドウサイズ

# 2. ファイル選択ボタンを作成
open_button = tk.Button(
    root, 
    text="📁 ファイルを選択 (Finderが開きます)", 
    command=open_finder_dialog, # ボタンが押されたらopen_finder_dialog関数を実行
    font=("Helvetica", 12),
    bg="#f0f0f0", # ボタンの背景色（macOSっぽく）
    padx=10,
    pady=5
)

# 3. ボタンをウィンドウに配置
open_button.pack(pady=40)

# 4. メインループの開始
root.mainloop()