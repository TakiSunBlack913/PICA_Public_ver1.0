# main_hub.py

import tkinter as tk
from tkinter import messagebox
import subprocess
import sys
import os

# --- 1. 実行するモジュール名の設定 ---
# これらのファイルは main_hub.py と同じディレクトリにある必要があります
MODULE_CROP = "face_crop_tool_gui.py"
MODULE_TRAIN = "train_model_2.py"
MODULE_SORT = "sort_faces_gui.py"
MODULE_APP = "face_app_tk.py"

class MainHubApp:
    def __init__(self, master):
        self.master = master
        master.title("🤖 顔識別システム - メインハブ")
        master.geometry("500x380")

        # 案内ラベル
        tk.Label(
            master,
            text="実行したいシステムの機能を選択してください",
            font=('Helvetica', 16, 'bold'),
            pady=20
        ).pack()

        # --- 2. 実行ボタンの配置 ---

        # 2.1. 学習データ作成ツール
        self.create_button(
            "1️⃣ 学習データ作成 (face_crop_tool_gui.py) を起動",
            MODULE_CROP,
            'lightblue'
        ).pack(pady=5)
        
        # 2.2. モデル学習ツール
        self.create_button(
            "2️⃣ モデル学習 (train_model_2.py) を実行",
            MODULE_TRAIN,
            'lightcoral'
        ).pack(pady=5)

        # 2.3. ファイル振り分けツール
        self.create_button(
            "3️⃣ ファイル振り分け (sort_faces_gui.py) を起動",
            MODULE_SORT,
            'lightgreen'
        ).pack(pady=5)
        
        # 2.4. 人物識別(旧名称：リアルタイム)識別アプリ
        self.create_button(
            "4️⃣ 人物識別 (face_app_tk.py) を起動",
            MODULE_APP,
            'lightgoldenrod'
        ).pack(pady=5)
        
        # 起動確認用のステータスラベル
        self.status_label = tk.Label(master, text="", fg='blue')
        self.status_label.pack(pady=10)

    def create_button(self, text, module_name, color):
        """共通のボタンウィジェットを作成するヘルパー関数"""
        return tk.Button(
            self.master,
            text=text,
            command=lambda: self.run_module(module_name),
            font=('Helvetica', 12),
            bg=color,
            padx=10,
            pady=8,
            width=40
        )

    # --- 3. モジュール実行ロジック ---
    def run_module(self, module_name):
        """指定されたPythonスクリプトを別プロセスで起動する"""
        
        # 相対パスでのファイル存在チェック
        if not os.path.exists(module_name):
            messagebox.showerror("エラー", f"ファイル '{module_name}' が見つかりません。\nファイル名を確認してください。")
            return
            
        self.log(f"'{module_name}' を起動中です...")
        
        try:
            # sys.executableは現在実行中のPythonインタープリタのパス
            # subprocess.Popen で別プロセスとして起動
            # main_hub.py とサブモジュールが同じディレクトリにあることが前提
            subprocess.Popen([sys.executable, module_name])
            
            self.log(f"'{module_name}' が起動しました。")
            
        except Exception as e:
            messagebox.showerror("実行エラー", f"'{module_name}' の起動中にエラーが発生しました: {e}")
            self.log("起動エラーが発生しました。")
            
    def log(self, message):
        """ステータスラベルを更新する"""
        self.status_label.config(text=message)
        self.master.update()

if __name__ == "__main__":
    # Windows/Macで起動時にPythonの黒いコンソールを出さないようにするための処理 (Macの.shで起動する場合不要)
    if sys.platform.startswith('win') and sys.executable.endswith("pythonw.exe"):
        # WindowsのGUI環境から実行された場合
        pass 
    
    root = tk.Tk()
    app = MainHubApp(root)
    root.mainloop()