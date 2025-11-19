# sort_faces_gui.py

import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
import os
import pickle
import numpy as np
import face_recognition
import shutil
from collections import defaultdict
import threading # GUIをフリーズさせないために、処理を別スレッドで実行

# --- 1. 定数設定 ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__)) 
MODEL_FILE = os.path.join(PROJECT_ROOT, "face_classifier_model.pkl")
DEFAULT_THRESHOLD = 0.77

# --- 2. モデルのロード ---
def load_model():
    """モデル、エンコーダーを読み込む"""
    try:
        with open(MODEL_FILE, 'rb') as f:
            (clf, le) = pickle.load(f)
        return clf, le
    except FileNotFoundError:
        messagebox.showerror("エラー", f"モデルファイルが見つかりません: {MODEL_FILE}\n先に学習を実行してください。")
        return None, None
    except Exception as e:
        messagebox.showerror("エラー", f"モデルロード中に予期せぬエラーが発生しました: {e}")
        return None, None

# アプリ起動時にモデルをロード
clf, le = load_model()

# --- 3. メインアプリの定義 ---

class FaceSorterApp:
    def __init__(self, master):
        self.master = master
        master.title("📁 顔画像ファイル振り分けツール")
        master.geometry("650x700")

        if clf is None:
            tk.Label(master, text="🚨 モデルがロードされていません。アプリを終了します。", fg="red").pack(pady=20)
            master.protocol("WM_DELETE_WINDOW", master.quit)
            return

        self.setup_ui()
        
    def setup_ui(self):
        """UI要素の配置"""
        
        main_frame = tk.Frame(self.master, padx=10, pady=10)
        main_frame.pack(fill='x')

        # --- 3.1. 入力ディレクトリ設定 ---
        tk.Label(main_frame, text="1. 入力フォルダ (振り分け対象画像)", anchor="w").pack(fill='x', pady=(10, 0))
        input_frame = tk.Frame(main_frame)
        input_frame.pack(fill='x')
        self.input_dir_var = tk.StringVar(value=os.path.join(PROJECT_ROOT, "test_data"))
        tk.Entry(input_frame, textvariable=self.input_dir_var, width=50).pack(side='left', fill='x', expand=True)
        tk.Button(input_frame, text="参照", command=lambda: self.select_directory(self.input_dir_var)).pack(side='left')

        # --- 3.2. 出力ディレクトリ設定 ---
        tk.Label(main_frame, text="2. 出力フォルダ (振り分け先)", anchor="w").pack(fill='x', pady=(10, 0))
        output_frame = tk.Frame(main_frame)
        output_frame.pack(fill='x')
        self.output_dir_var = tk.StringVar(value=os.path.join(PROJECT_ROOT, "sorted_output"))
        tk.Entry(output_frame, textvariable=self.output_dir_var, width=50).pack(side='left', fill='x', expand=True)
        tk.Button(output_frame, text="参照", command=lambda: self.select_directory(self.output_dir_var)).pack(side='left')

        # --- 3.3. しきい値設定 ---
        tk.Label(main_frame, text="3. 確信度しきい値 (例: 0.77)", anchor="w").pack(fill='x', pady=(10, 0))
        self.threshold_var = tk.StringVar(value=str(DEFAULT_THRESHOLD))
        tk.Entry(main_frame, textvariable=self.threshold_var, width=10).pack(fill='x')

        # --- 3.4. 実行ボタン ---
        self.sort_button = tk.Button(
            main_frame,
            text="🚀 振り分け実行 (ファイルをコピーします)",
            command=self.start_sorting_thread,
            font=('Helvetica', 12, 'bold'),
            bg='orange',
            padx=10,
            pady=10
        )
        self.sort_button.pack(pady=20, fill='x')

        #プログレスバーの追加
        self.progress_bar = ttk.Progressbar(
             main_frame,
             orient='horizontal',
             mode='determinate'
         )
        self.progress_bar.pack(fill='x', padx=5, pady=(5, 10))
        self.progress_bar.config(value=0) # 初期値は0
        
        # --- 3.5. 結果表示エリア ---
        tk.Label(self.master, text="結果とログ:", anchor="w").pack(fill='x', padx=10)
        self.result_text = scrolledtext.ScrolledText(self.master, wrap=tk.WORD, height=20, padx=5, pady=5)
        self.result_text.pack(fill='both', expand=True, padx=10, pady=10)
        self.result_text.insert(tk.END, f"準備完了。\n現在のモデル学習人数: {len(le.classes_)}人\n\n")

    def select_directory(self, var):
        """フォルダ選択ダイアログを開き、StringVarを更新する"""
        directory = filedialog.askdirectory()
        if directory:
            var.set(directory)

    def log(self, message):
        """ログを結果エリアに追記する"""
        self.result_text.insert(tk.END, message + "\n")
        self.result_text.see(tk.END) # 最下行までスクロール
        self.master.update()

    def start_sorting_thread(self):
        """GUIをフリーズさせないために、振り分け処理を別スレッドで開始する"""
        self.sort_button.config(state=tk.DISABLED, text="処理中...")
        self.result_text.delete('1.0', tk.END)
        self.log("--- 振り分け処理を開始します ---")
        
        # 別スレッドで実行
        threading.Thread(target=self.run_sorting_process).start()

    def run_sorting_process(self):
        """sort_faces.pyのコアロジックを実装する（全顔チェック対応）"""
        
        try:
            # 入力値の取得と検証 (変更なし)
            test_dir = self.input_dir_var.get()
            output_dir = self.output_dir_var.get()
            try:
                conf_threshold = float(self.threshold_var.get())
            except ValueError:
                self.log("🚨 エラー: しきい値が不正です。数値を入力してください。")
                return

            if not os.path.isdir(test_dir):
                self.log(f"🚨 エラー: 入力フォルダ '{test_dir}' が見つかりません。")
                return
            
            # --- コアロジックの開始 ---
            os.makedirs(output_dir, exist_ok=True)
            sorted_results = defaultdict(list)
            total_files_processed = 0
            
            file_list = [f for f in os.listdir(test_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            total_files = len(file_list)

            if not file_list:
                self.log("🚨 警告: 入力フォルダに画像ファイルが見つかりませんでした。")
                return

            self.log(f"✅ 設定: しきい値={conf_threshold}, 処理対象ファイル数={len(file_list)}")
            
            for i, filename in enumerate(file_list):
                
                current_count = i + 1
                
                # ログ出力（進捗）- 🚨 修正：条件を削除し、常にログを出力 🚨
                # ファイル名とその時点での進捗を毎回表示します
                self.log(f"  > 処理中: {current_count} / {total_files} ファイル ({filename})")
                
                # プログレスバーの更新 (変更なし)
                self.progress_bar.config(value=current_count)
                self.master.update() # GUIを更新
                
                image_path = os.path.join(test_dir, filename)
                total_files_processed += 1
                
                # 初期設定
                final_predicted_name = "Unknown"
                best_confidence = 0.0

                try:
                    # 1. 顔検出とエンコーディング抽出
                    image = face_recognition.load_image_file(image_path)
                    
                    #検出方法
                    #face_locations = face_recognition.face_locations(image, model="hog" , number_of_times_to_upsample=2)
                    face_locations = face_recognition.face_locations(image, model="cnn") 
                    
                    if len(face_locations) > 0:
                        # 検出されたすべての顔をチェックするループ
                        encodings = face_recognition.face_encodings(image, face_locations)
                        
                        for test_encoding in encodings:
                            test_encoding = test_encoding.reshape(1, -1)
                            
                            # 2. 識別と信頼度計算
                            probabilities = clf.predict_proba(test_encoding)[0]
                            max_proba = np.max(probabilities)
                            max_index = np.argmax(probabilities)
                            
                            # 3. しきい値に基づいて人物名を決定
                            current_predicted_name = "Unknown"
                            if max_proba >= conf_threshold:
                                prediction_numeric = np.array([max_index])
                                current_predicted_name = le.inverse_transform(prediction_numeric)[0]

                            # 4. 振り分け名の決定: Unknownではない、かつ、より高い確信度の場合に採用
                            if current_predicted_name != "Unknown":
                                if max_proba > best_confidence:
                                    best_confidence = max_proba
                                    final_predicted_name = current_predicted_name
                            
                        # 5. ループ終了後の最終判定
                        if final_predicted_name != "Unknown":
                            sorted_results[final_predicted_name].append((filename, best_confidence))
                        else:
                            # すべての顔がUnknownだった場合
                            sorted_results["Unknown"].append((filename, best_confidence))
                        
                    else:
                        final_predicted_name = "Unknown (No Face)"
                        sorted_results[final_predicted_name].append((filename, 0.0))

                except Exception as e:
                    self.log(f"⚠️ ファイル {filename} の処理中にエラーが発生しました: {e}")
                    sorted_results["Unknown (Error)"].append((filename, 0.0))
            
            
            # --- 4. フォルダへの振り分けと結果の表示 (変更なし) ---
            self.log("\n--- 4. ファイルの振り分けと結果まとめ ---")
            
            for name, files_with_proba in sorted_results.items(): 
                output_folder_path = os.path.join(output_dir, name)
                os.makedirs(output_folder_path, exist_ok=True)
                
                self.log(f"\n👤 フォルダ '{name}' に {len(files_with_proba)} 枚を振り分けます。")
                
                for filename, proba in files_with_proba: 
                    source_path = os.path.join(test_dir, filename)
                    dest_path = os.path.join(output_folder_path, filename)
                    
                    # ファイルをコピーして振り分け
                    shutil.copy(source_path, dest_path) 
            
            self.log("\n==================================================")
            self.log(f"✅ 処理完了！ {total_files_processed} ファイルを振り分けました。")
            self.log(f"結果は '{output_dir}' にコピーされています。")
            self.log("==================================================")
            
        except Exception as e:
            self.log(f"\n致命的なエラーが発生しました: {e}")
            messagebox.showerror("エラー", f"予期せぬエラー: {e}")
            
        finally:
            self.sort_button.config(state=tk.NORMAL, text="🚀 振り分け実行 (ファイルをコピーします)")
            


# --- 5. アプリケーションの実行 ---
if __name__ == "__main__":
    root = tk.Tk()
    app = FaceSorterApp(root)
    root.mainloop()