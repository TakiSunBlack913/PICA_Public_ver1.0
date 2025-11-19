# face_app_tk.py

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import os
import pickle
import numpy as np
import face_recognition
from io import BytesIO
from collections import defaultdict
import datetime
import math # スクロールバーのための数学関数

# --- 1. 定数設定 ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__)) 
MODEL_FILE = os.path.join(PROJECT_ROOT, "face_classifier_model.pkl")
CONFIDENCE_THRESHOLD = 0.78
MAP_FILE = os.path.join(PROJECT_ROOT, "name_id_map.pkl") 

# --- 2. モデルのロード ---
def load_model():
    """モデル、エンコーダー、IDマップを読み込む"""
    try:
        with open(MODEL_FILE, 'rb') as f:
            (clf, le) = pickle.load(f)
        
        id_name_map = None
        if os.path.exists(MAP_FILE):
             with open(MAP_FILE, 'rb') as f:
                id_name_map = pickle.load(f)
                
        return clf, le, id_name_map
        
    except FileNotFoundError:
        messagebox.showerror("エラー", f"モデルファイルが見つかりません: {MODEL_FILE}\n先に学習を実行してください。")
        return None, None, None
    except Exception as e:
        messagebox.showerror("エラー", f"モデルロード中に予期せぬエラーが発生しました: {e}")
        return None, None, None

# アプリ起動時にモデルをロード
clf, le, id_name_map = load_model()

# --- 3. メインアプリの定義 ---

class FaceIdentificationApp:
    def __init__(self, master):
        self.master = master
        master.title("👤 顔識別アプリ (Tkinter)")
        master.geometry("800x600") # ウィンドウサイズを少し大きく設定

        if clf is None:
            tk.Label(master, text="🚨 モデルがロードされていません。アプリを終了します。", fg="red").pack(pady=20)
            master.protocol("WM_DELETE_WINDOW", master.quit)
            return
        
        # 識別結果の表示コンテナ（キャンバスとスクロールバーを含む）
        self.create_result_area(master)

        # UI要素の配置
        tk.Label(master, text="検証画像の選択と識別", font=('Helvetica', 16, 'bold')).pack(pady=10)
        
        # ファイル選択ボタン
        self.select_button = tk.Button(
            master,
            text="📂 画像ファイルを選択",
            command=self.select_files,
            font=('Helvetica', 12),
            bg='lightblue',
            padx=10,
            pady=5
        )
        self.select_button.pack(pady=5)

        # ステータスラベル
        self.status_label = tk.Label(master, text=f"準備完了 | 学習人数: {len(le.classes_)}人", pady=10)
        self.status_label.pack()
        
        # PIL.ImageをTkinter.PhotoImageに変換したものを保持するための辞書
        self.tk_images = {} 


    def create_result_area(self, master):
        """結果表示用のキャンバスとフレームをセットアップする"""
        
        # 結果表示コンテナ（スクロールバーが必要なためCanvasを使用）
        self.canvas = tk.Canvas(master, borderwidth=0, background="#ffffff")
        self.canvas.pack(side="top", fill="both", expand=True, padx=10, pady=10)

        # スクロールバーのセットアップ
        self.vsb = tk.Scrollbar(master, orient="vertical", command=self.canvas.yview)
        self.vsb.pack(side="right", fill="y")
        self.canvas.configure(yscrollcommand=self.vsb.set)

        # 結果を配置するためのフレームをキャンバス上に作成
        self.results_frame = tk.Frame(self.canvas, background="#ffffff")
        self.canvas.create_window((0, 0), window=self.results_frame, anchor="nw")

        # フレームサイズが変わったときにキャンバスのスクロール領域を更新
        self.results_frame.bind("<Configure>", lambda event: self.canvas.configure(
            scrollregion=self.canvas.bbox("all")
        ))
        
    # --- 4. ファイル選択処理 ---
    def select_files(self):
        """ファイル選択ダイアログを開き、ファイルパスを取得する"""
        
        file_paths = filedialog.askopenfilenames(
            defaultextension=".jpg",
            filetypes=[("Image files", "*.jpg *.jpeg *.png"), ("All files", "*.*")],
            title="検証対象の画像を選択してください (複数選択可)"
        )

        if not file_paths:
            return

        # 既存の結果をクリア
        self.tk_images.clear() # 前の画像を保持する辞書をクリア
        for widget in self.results_frame.winfo_children():
            widget.destroy()
            
        # 識別処理を開始
        self.process_files(list(file_paths))

    # --- 5. 識別処理の統合 (Fletロジックを移植) ---
    
    def identify_face(self, face_encoding):
        """
        単一の顔エンコーディングを学習済みモデルで識別する
        戻り値: (予測名, 信頼度)
        """
        # SVMモデルによる識別
        probabilities = clf.predict_proba([face_encoding])[0]
        max_prob_index = np.argmax(probabilities)
        max_prob = probabilities[max_prob_index]
        
        # 信頼度の低い結果は "Unknown" とする
        if max_prob < CONFIDENCE_THRESHOLD:
            predicted_id = "Unknown"
            confidence = max_prob * 100
        else:
            predicted_id = le.classes_[max_prob_index]
            confidence = max_prob * 100
            
        # IDを日本語名に変換（将来的な拡張を見据えて）
        predicted_name = id_name_map.get(predicted_id, predicted_id) if id_name_map else predicted_id
        
        return predicted_name, confidence

    # def apply_best_match_logic(self, raw_predictions):
    #     """
    #     同じ画像内で同一人物と誤認された顔を、最高信頼度の顔以外はUnknownに強制変更する。
    #     """
    #     best_matches = {} # {name: highest_confidence}
    #     all_matches = []  # [(index, name, confidence), ...]
        
    #     # 1. 最高信頼度の記録
    #     for idx, (name, confidence) in enumerate(raw_predictions):
    #         all_matches.append((idx, name, confidence))
            
    #         # 信頼度がしきい値を超えている、かつUnknownでない場合のみ処理対象
    #         if confidence >= CONFIDENCE_THRESHOLD * 100 and name != "Unknown": # confidenceはここで%表示なので*100が必要
    #             if name not in best_matches or confidence > best_matches[name]:
    #                 best_matches[name] = confidence

    #     final_predictions = [] 

    #     # 2. 最終判定: 最高信頼度の顔以外はUnknownに強制変更
    #     for idx, name, confidence in all_matches:
            
    #         # Unknownまたはしきい値未満はそのまま
    #         if name == "Unknown" or confidence < CONFIDENCE_THRESHOLD * 100:
    #             final_predictions.append((name, confidence))
    #             continue
            
    #         # 最高信頼度として記録されたものか？
    #         if confidence == best_matches.get(name):
    #             # 採用: そのまま採用
    #             final_predictions.append((name, confidence))
    #             # 🚨 重要: 他の同じ名前の顔が採用されないように、この人物の最高信頼度を無効化
    #             best_matches[name] = -1.0 
    #         else:
    #             # 誤認と判断: Unknownに強制変更
    #             final_predictions.append(("Unknown", confidence))
                
    #     return final_predictions
    
    # def apply_best_match_logic(self, raw_predictions):
    #     """
    #     同じ画像内で同一人物と誤認された顔を、最高信頼度の顔以外はUnknownに強制変更する。
    #     """
    #     best_matches_index = {} # {name: (highest_confidence, index)}
        
    #     # 1. 各人物の最高信頼度とそのインデックスを記録
    #     for idx, (name, confidence) in enumerate(raw_predictions):
            
    #         # 信頼度がしきい値を超えている、かつUnknownでない場合のみ処理対象
    #         if confidence >= CONFIDENCE_THRESHOLD * 100 and name != "Unknown":
                
    #             # 現在の信頼度が記録されている最高信頼度よりも高い場合、または未記録の場合
    #             if name not in best_matches_index or confidence > best_matches_index[name][0]:
    #                 best_matches_index[name] = (confidence, idx) # (信頼度, インデックス) を記録

    #     final_predictions = [] 
        
    #     # 2. 最終判定: 最高信頼度の顔のインデックスだけを採用し、その他はUnknownに強制変更
    #     for idx, (name, confidence) in enumerate(raw_predictions):
            
    #         # Unknownまたはしきい値未満はそのまま
    #         if name == "Unknown" or confidence < CONFIDENCE_THRESHOLD * 100:
    #             final_predictions.append((name, confidence))
    #             continue
            
    #         # 🚨 判定: 現在のインデックスが、最高信頼度として記録されたインデックスと一致するか？
    #         if name in best_matches_index and idx == best_matches_index[name][1]:
    #             # 採用: 最高信頼度の顔なのでそのまま採用
    #             final_predictions.append((name, confidence))
    #         else:
    #             # 排除: 最高信頼度ではない（あるいは、最高信頼度だが同率の別顔）のでUnknownに強制変更
    #             final_predictions.append(("Unknown", confidence))
                
    #     return final_predictions

    def apply_best_match_logic(self, raw_predictions):
        """
        同じ画像内で検出された顔について、各人物名の予測のうち、
        最も信頼度の高い1つの顔のみを採用し、他をUnknownに強制変更するロジック。
        """
        best_match_index_per_name = {} # {name: (highest_confidence, index)}
        
        # 1. 各人物名について、最も信頼度の高い顔のインデックスを記録
        for idx, (name, confidence) in enumerate(raw_predictions):
            
            # しきい値を超えている、かつUnknownでない場合のみ処理
            # confidenceはパーセンテージ(0-100)として扱います
            if confidence >= CONFIDENCE_THRESHOLD * 100 and name != "Unknown":
                
                # 記録されている最高信頼度よりも高い場合、または未記録の場合に更新
                # 同率の場合、先に記録されたもの(idxが若いもの)が優先される
                if name not in best_match_index_per_name or confidence > best_match_index_per_name[name][0]:
                    best_match_index_per_name[name] = (confidence, idx) # (信頼度, インデックス) を記録

        final_predictions = [] 
        
        # 2. 最終判定: 記録された最高信頼度のインデックスのみを採用
        adopted_indices = set() # 採用されたインデックスを保持するセット
        
        # 採用インデックスをセットに格納
        for confidence, idx in best_match_index_per_name.values():
            adopted_indices.add(idx)
            
        # 最終的な予測リストを構築
        for idx, (name, confidence) in enumerate(raw_predictions):
            
            # 採用インデックスに現在のインデックスがあるかチェック
            if idx in adopted_indices:
                # 採用された顔: そのまま採用
                final_predictions.append((name, confidence))
            else:
                # 排除された顔: Unknownに強制変更 (Unknownにしきい値未満も含む)
                final_predictions.append(("Unknown", confidence))
                
        return final_predictions

    def process_files(self, file_paths):
        """
        選択されたファイルを処理し、顔識別を実行して結果を描画する
        """
        self.status_label.config(text=f"{len(file_paths)} 個のファイルを処理中...")
        self.master.update()
        
        col = 0
        row = 0
        max_cols = 3 # 一行に表示する最大枚数

        for file_path in file_paths:
            try:
                # 1. 画像の読み込みと顔検出
                image = face_recognition.load_image_file(file_path)
                #face_locations = face_recognition.face_locations(image, model="hog")
                face_locations = face_recognition.face_locations(image, model="cnn")
                face_encodings = face_recognition.face_encodings(image, face_locations)

                if not face_encodings:
                    self.display_result_item(file_path, "顔未検出", 0, row, col)
                    col += 1
                    if col >= max_cols:
                        col = 0
                        row += 1
                    continue

                #識別結果と切り抜き画像を格納するリストを用意
                raw_predictions = [] # [(name, confidence), ...]
                cropped_faces_data = [] # 切り抜き画像などのデータ格納
                
                # 2. 識別処理と結果表示
                for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
                    
                    # 識別
                    predicted_name, confidence = self.identify_face(face_encoding)
                    
                    # 3. 顔の切り抜き（PILを使用）
                    # face_recognitionの座標は(top, right, bottom, left)
                    pil_image = Image.fromarray(image)
                    
                    # 顔の領域にパディングを追加 (顔の輪郭を捉えるため)
                    padding = 50
                    cropped_face = pil_image.crop((
                        max(0, left - padding), 
                        max(0, top - padding), 
                        min(pil_image.width, right + padding), 
                        min(pil_image.height, bottom + padding)
                    ))

                    # 収集: 結果をリストに格納 (描画はまだ行わない)
                    raw_predictions.append((predicted_name, confidence))
                    cropped_faces_data.append(cropped_face)

                    #あと処理ロジック（同一人物誤認をUnknownに修正）
                    final_predictions = self.apply_best_match_logic(raw_predictions)

                    for i, cropped_face in enumerate(cropped_faces_data):
                        final_name, final_confidence = final_predictions[i]

                    # 結果をGUIに描画
                    self.display_result_item(file_path, final_name, final_confidence, row, col, cropped_face)
                    
                    col += 1
                    if col >= max_cols:
                        col = 0
                        row += 1

            except Exception as e:
                self.status_label.config(text=f"エラー: {file_path} の処理中にエラーが発生しました: {e}")
                col += 1
                if col >= max_cols:
                    col = 0
                    row += 1

        self.status_label.config(text=f"処理完了！")
        # 処理完了後、スクロールバーを再調整
        self.results_frame.update_idletasks()
        self.canvas.config(scrollregion=self.canvas.bbox("all"))

    def display_result_item(self, file_path, name, confidence, row, col, cropped_face=None):
        """
        単一の識別結果をresults_frameに表示する
        """
        item_frame = tk.Frame(self.results_frame, bd=2, relief="groove", padx=10, pady=10)
        item_frame.grid(row=row, column=col, padx=10, pady=10, sticky="n")

        # 1. 画像表示 (顔のサムネイル)
        if cropped_face:
            # 画像をリサイズ
            display_size = (150, 150)
            resized_image = cropped_face.resize(display_size, Image.Resampling.LANCZOS)
            
            # Tkinterで表示可能な形式に変換
            tk_img = ImageTk.PhotoImage(resized_image)
            
            # 画像のライフサイクル管理のために、クラス変数に保持
            # これをしないと、ガベージコレクションによって画像が消えてしまう
            # キーをファイルパスとrow,colの組み合わせにして一意にする
            img_key = f"{file_path}_{row}_{col}"
            self.tk_images[img_key] = tk_img 
            
            img_label = tk.Label(item_frame, image=tk_img)
            img_label.pack(pady=5)
        else:
            tk.Label(item_frame, text="画像なし / 顔未検出", width=20, height=8).pack(pady=5)

        # 2. 結果テキスト
        
        # 信頼度に基づく色分け
        if name == "Unknown" or confidence < CONFIDENCE_THRESHOLD * 100:
            color = "red"
        elif name == "顔未検出":
             color = "orange"
        else:
            color = "green"
            
        result_text = f"名前: {name}\n信頼度: {confidence:.2f}%"
        
        tk.Label(item_frame, text=result_text, fg=color, font=('Helvetica', 10, 'bold')).pack(pady=5)
        
        # ファイル名表示
        tk.Label(item_frame, text=os.path.basename(file_path), font=('Helvetica', 8)).pack()
        
        
# --- 6. アプリケーションの実行 ---
if __name__ == "__main__":
    root = tk.Tk()
    app = FaceIdentificationApp(root)
    root.mainloop()