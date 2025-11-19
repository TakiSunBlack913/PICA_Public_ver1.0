# face_crop_tool_gui.py

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import face_recognition
from PIL import Image
import os
import shutil

# --- 設定 ---
PADDING = 40  # 切り取る顔の周囲に加える余白（ピクセル）

class FaceCropToolApp:
    def __init__(self, master):
        self.master = master
        master.title("✂️ 顔切り取りツール (GUI)")
        master.geometry("550x450")

        # --- 変数 ---
        self.input_dir_var = tk.StringVar()
        self.output_dir_var = tk.StringVar()
        
        # --- レイアウト ---
        
        main_frame = tk.Frame(master, padx=20, pady=20)
        main_frame.pack(fill="both", expand=True)

        tk.Label(main_frame, text="ステップ 1: 入力フォルダの選択", font=('Helvetica', 12, 'bold')).pack(pady=(5, 5))
        
        # 入力フォルダ選択
        input_frame = tk.Frame(main_frame)
        input_frame.pack(fill='x', pady=5)
        
        tk.Button(input_frame, text="元画像フォルダを選択", command=self.select_input_dir).pack(side='left', padx=10)
        tk.Label(input_frame, text="入力パス:").pack(side='left')
        tk.Entry(input_frame, textvariable=self.input_dir_var, width=40).pack(side='left', fill='x', expand=True)

        tk.Label(main_frame, text="ステップ 2: 出力フォルダの選択", font=('Helvetica', 12, 'bold')).pack(pady=(15, 5))
        
        # 出力フォルダ選択
        output_frame = tk.Frame(main_frame)
        output_frame.pack(fill='x', pady=5)
        
        tk.Button(output_frame, text="保存先フォルダを選択", command=self.select_output_dir).pack(side='left', padx=10)
        tk.Label(output_frame, text="出力パス:").pack(side='left')
        tk.Entry(output_frame, textvariable=self.output_dir_var, width=40).pack(side='left', fill='x', expand=True)

        # --- 処理実行ボタン ---
        tk.Button(main_frame, text="🔴 顔切り取り処理を実行", command=self.start_processing, 
                  font=('Helvetica', 12, 'bold'), bg='#FFCCCC', padx=20, pady=10).pack(pady=20)
        
        # --- ステータスと進捗 ---
        tk.Label(main_frame, text="--- ステータス ---", font=('Helvetica', 10, 'italic')).pack(pady=(5, 0))
        
        self.status_label = tk.Label(main_frame, text="準備完了。フォルダを選択してください。", wraplength=500)
        self.status_label.pack(pady=5)
        
        self.progress_bar = ttk.Progressbar(main_frame, orient="horizontal", length=400, mode="determinate")
        self.progress_bar.pack(pady=10)
        
        # 処理中にエラーメッセージや警告を保持するリスト（コンソールとGUIで確認用）
        self.process_logs = []

    # --- コマンド ---

    def select_input_dir(self):
        """元画像フォルダを選択"""
        directory = filedialog.askdirectory(title="元画像が入っているフォルダを選択")
        if directory:
            self.input_dir_var.set(directory)

    def select_output_dir(self):
        """保存先フォルダを選択"""
        directory = filedialog.askdirectory(title="切り取り画像を保存するフォルダを選択")
        if directory:
            self.output_dir_var.set(directory)

    def start_processing(self):
        """処理を開始する前のチェックと実行"""
        input_dir = self.input_dir_var.get()
        output_dir = self.output_dir_var.get()

        if not os.path.isdir(input_dir):
            messagebox.showerror("エラー", "入力フォルダが無効です。再度選択してください。")
            return
        if not output_dir:
            messagebox.showerror("エラー", "出力フォルダを選択してください。")
            return
        
        # 処理実行
        self.process_directory(input_dir, output_dir)
        
    def process_directory(self, input_dir, output_dir):
        """顔切り取りのメインロジックを実行"""
        
        self.process_logs = []
        self.status_label.config(text="処理開始中...")
        self.master.update()

        # 出力フォルダの準備と既存データ削除の確認
        if os.path.exists(output_dir):
            if not messagebox.askyesno("確認", "出力フォルダは既に存在します。内容を削除して続行しますか？"):
                self.status_label.config(text="処理中断。")
                return
            shutil.rmtree(output_dir)
            
        os.makedirs(output_dir)

        # 全体のファイル数をカウント（進捗バーのため）
        all_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        total_files = len(all_files)
        
        if total_files == 0:
            messagebox.showinfo("情報", "入力フォルダ内に画像ファイルが見つかりませんでした。")
            self.status_label.config(text="処理完了（画像なし）。")
            return

        # --- メイン処理 ---
        total_faces = 0
        
        for index, filename in enumerate(all_files):
            input_path = os.path.join(input_dir, filename)
            
            # 進捗バーの更新
            progress_val = int(((index + 1) / total_files) * 100)
            self.progress_bar['value'] = progress_val
            self.status_label.config(text=f"処理中: {index + 1}/{total_files} 枚 ({progress_val}%)")
            self.master.update()
            
            try:
                # 1. 画像の読み込みと顔検出
                image = face_recognition.load_image_file(input_path)
                face_locations = face_recognition.face_locations(image, model="cnn")
                
                if not face_locations:
                    self.process_logs.append(f"[⚠️ 警告] {filename}: 顔が検出されませんでした。スキップ。")
                    continue
                
                # 2. 各顔を切り取り、保存
                for i, (top, right, bottom, left) in enumerate(face_locations):
                    pil_image = Image.fromarray(image)
                    
                    # 座標にパディングを適用
                    cropped_face = pil_image.crop((
                        max(0, left - PADDING), 
                        max(0, top - PADDING), 
                        min(pil_image.width, right + PADDING), 
                        min(pil_image.height, bottom + PADDING)
                    ))
                    
                    # ファイル名の生成
                    base_name, ext = os.path.splitext(filename)
                    output_filename = f"{base_name}_face_{i+1}{ext}"
                    output_path = os.path.join(output_dir, output_filename)
                    
                    cropped_face.save(output_path)
                    total_faces += 1
                
            except Exception as e:
                self.process_logs.append(f"[❌ エラー] {filename} の処理中にエラーが発生: {e}")


        # --- 処理結果表示 ---
        self.progress_bar['value'] = 100
        
        result_message = f"✅ 処理が完了しました！\n"
        result_message += f"処理ファイル数: {total_files} 枚\n"
        result_message += f"保存された顔画像数: {total_faces} 枚"
        
        self.status_label.config(text=result_message)
        
        if self.process_logs:
            log_text = "\n".join(self.process_logs)
            messagebox.showwarning("警告/エラーログ", f"処理中に警告またはエラーが発生しました。\nログをコンソールに出力します。\n\n{log_text}")
            print("\n--- 処理ログ ---")
            print(log_text)
            print("----------------")
        
        # 実行完了後、進捗バーをリセット
        self.progress_bar['value'] = 0


if __name__ == "__main__":
    root = tk.Tk()
    app = FaceCropToolApp(root)
    root.mainloop()