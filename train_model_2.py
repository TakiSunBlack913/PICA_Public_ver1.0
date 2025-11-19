# train_model_2.py
import platform  # OSを判別するため
import subprocess # Mac/Linuxでフォルダを開くため
import face_recognition
import numpy as np
import os
import pickle
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import tkinter as tk
from tkinter import messagebox, filedialog, ttk
import time # 処理時間計測用

# --- 1. 定数設定 ---
TRAIN_DIR = "train_data"
MODEL_FILE = "face_classifier_model.pkl"
ENCODINGS_FILE = "face_encodings.pkl"

# --- 2. モデル学習ロジック（GUIから呼び出す関数） ---

def run_training_logic(root, status_label, progress_bar, time_label):
    """メインの学習ロジックを実行し、GUIにステータスと進捗を反映させる"""
    
    status_label.config(text="処理開始: 初期準備中...")
    status_label.update()
    root.update()

    known_encodings = []
    known_names = []
    total_images = 0 # 全体の画像数をカウントするための変数

    try:
        if not os.path.exists(TRAIN_DIR):
            messagebox.showerror("エラー", f"訓練データフォルダ '{TRAIN_DIR}' が見つかりません。")
            status_label.config(text="待機中...")
            return
        
        # --- ステップ 1: 全体の画像数を事前カウント ---
        for name in os.listdir(TRAIN_DIR):
            if name.startswith('.'): continue
            person_dir = os.path.join(TRAIN_DIR, name)
            if not os.path.isdir(person_dir): continue
            for filename in os.listdir(person_dir):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    total_images += 1
        
        if total_images == 0:
            messagebox.showinfo("警告", "訓練データフォルダに画像が見つかりません。")
            status_label.config(text="待機中...")
            return

        # 処理状況カウンターを初期化
        processed_count = 0
        
        # --- ステップ 2: 特徴量抽出と進捗更新 ---
        start_time = time.time()
        
        for name in os.listdir(TRAIN_DIR):
            if name.startswith('.'): continue
            person_dir = os.path.join(TRAIN_DIR, name)
            if not os.path.isdir(person_dir): continue
            
            for filename in os.listdir(person_dir):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_path = os.path.join(person_dir, filename)
                    
                    # 1. 画像の読み込みと特徴量抽出
                    image = face_recognition.load_image_file(image_path)
                    #face_locations = face_recognition.face_locations(image, model="hog")
                    face_locations = face_recognition.face_locations(image, model="cnn")
                    encodings = face_recognition.face_encodings(image, face_locations)

                    if len(encodings) > 0:
                        known_encodings.append(encodings[0])
                        known_names.append(name)

                    # 2. 進捗と残り時間の更新
                    processed_count += 1
                    
                    # 進捗率の計算
                    progress_percent = int((processed_count / total_images) * 100)
                    
                    # 経過時間の計算と残り時間の予測
                    elapsed_time = time.time() - start_time
                    time_per_image = elapsed_time / processed_count
                    remaining_time_sec = (total_images - processed_count) * time_per_image
                    
                    remaining_time_str = time.strftime("%H:%M:%S", time.gmtime(remaining_time_sec))

                    # GUI要素の更新
                    progress_bar['value'] = progress_percent
                    status_label.config(text=f"特徴量抽出中: {name} さんの写真 ({processed_count}/{total_images} 枚)")
                    time_label.config(text=f"進捗: {progress_percent}% | 予想残り時間: {remaining_time_str}")
                    root.update() # GUIの描画を強制的に更新

        # --- ステップ 3: モデルの学習と保存 ---
        status_label.config(text="モデル学習中: SVM分類器の学習を開始...")
        root.update()
        
        le = LabelEncoder()
        names_numeric = le.fit_transform(known_names)
        clf = SVC(kernel='linear', C=1, gamma='scale', probability=True)
        clf.fit(known_encodings, names_numeric)

        with open(MODEL_FILE, 'wb') as f:
            pickle.dump((clf, le), f)

        # 最終的な表示
        progress_bar['value'] = 100
        messagebox.showinfo("成功", f"学習済みモデルを {MODEL_FILE} に保存しました。\n学習完了！")
        status_label.config(text="完了: 新しいモデルが保存されました。")
        time_label.config(text="進捗: 100% | 処理時間: " + time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
        
    except Exception as e:
        messagebox.showerror("エラー", f"学習中にエラーが発生しました: {e}")
        status_label.config(text="エラーが発生しました。")
        time_label.config(text="進捗: 0% | エラー")

# --- 3. Tkinter GUI の設定 ---

def create_gui():
    root = tk.Tk()
    root.title("モデル学習ツール v2")
    root.geometry("400x350")

    # 訓練フォルダのパス表示
    dir_label = tk.Label(root, text=f"訓練データフォルダ: {TRAIN_DIR}", pady=5)
    dir_label.pack()

    #⚠️警告文
    fixed_warning = tk.Label(root, text="フォルダの人物名はローマ字で入力してください \n(例：Taro_Yamada))", fg='red', font=('Helvetica', 10, 'italic'))
    fixed_warning.pack(pady=5)

    #ディレクトリを開くボタン
    open_dir_button = tk.Button(
        root,
        text="📂 学習フォルダを開く (画像配置用)",
        command=open_train_directory, 
        font=('Helvetica', 10),
        bg='lightblue',
        padx=5,
        pady=3
    )
    open_dir_button.pack(pady=(5, 15))

    # 学習開始ボタン
    train_button = tk.Button(
        root,
        text="モデル学習開始",
        # コマンドの引数としてroot, status_label, progress_bar, time_labelを渡す
        command=lambda: run_training_logic(root, status_label, progress_bar, time_label),
        font=('Helvetica', 12),
        bg='lightgreen',
        padx=20,
        pady=10
    )
    train_button.pack(pady=10)

    # 進捗バー
    progress_bar = ttk.Progressbar(root, orient='horizontal', length=300, mode='determinate')
    progress_bar.pack(pady=10)

    # 進捗率と残り時間の表示ラベル
    time_label = tk.Label(root, text="進捗: 0% | 予想残り時間: --:--:--", fg='blue', bg='lightgreen',)
    time_label.pack()
    
    # ステータス表示ラベル
    status_label = tk.Label(root, text="待機中...", pady=10)
    status_label.pack()


    root.mainloop()

def open_train_directory():
    """OSに応じて学習データフォルダをエクスプローラー/Finderで開く"""
    
    # TRAIN_DIR が定義されていることを確認してください
    # TRAIN_DIR = os.path.join(PROJECT_ROOT, "train_data")
    
    # フォルダが存在しなければ作成
    if not os.path.exists(TRAIN_DIR):
        os.makedirs(TRAIN_DIR)
        
    system = platform.system()
    try:
        if system == "Windows":
            # Windowsの場合: os.startfile を使用
            os.startfile(TRAIN_DIR)
        elif system == "Darwin": # Mac OS X
            # Macの場合: 'open' コマンドを使用
            subprocess.Popen(["open", TRAIN_DIR])
        else: # Linuxなどの場合
            # その他の場合: 'xdg-open' コマンドを使用
            subprocess.Popen(["xdg-open", TRAIN_DIR])
            
    except Exception as e:
        messagebox.showerror("エラー", f"フォルダを開けませんでした。手動で開いてください。\nパス: {TRAIN_DIR}\nエラー: {e}")

if __name__ == "__main__":
    create_gui()