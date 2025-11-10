# train_model.py

import face_recognition
import numpy as np
import os
import pickle
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import tkinter as tk
from tkinter import messagebox, filedialog, ttk
from PIL import Image # Tkinterでの画像表示に必要

# --- 1. 定数設定 ---
# 訓練データフォルダは静的に設定するか、GUIで選択できるようにする
TRAIN_DIR = "train_data"
MODEL_FILE = "face_classifier_model.pkl"

# --- 2. モデル学習ロジック（GUIから呼び出す関数） ---

def run_training_logic(status_label):
    """メインの学習ロジックを実行し、GUIにステータスを反映させる"""
    
    # 🚨 注意: Tkinterはシングルスレッドです。
    # 学習中はGUIが一時的にフリーズしますが、完了後にメッセージが表示されます。
    
    status_label.config(text="処理中: 学習データの読み込みと特徴量抽出を開始...")
    status_label.update() # GUIを即時更新

    known_encodings = []
    known_names = []

    try:
        if not os.path.exists(TRAIN_DIR):
            messagebox.showerror("エラー", f"訓練データフォルダ '{TRAIN_DIR}' が見つかりません。")
            status_label.config(text="待機中...")
            return

        # 既存の学習ロジックをここに挿入
        for name in os.listdir(TRAIN_DIR):
            if name.startswith('.'): continue
            person_dir = os.path.join(TRAIN_DIR, name)
            if not os.path.isdir(person_dir): continue
            
            status_label.config(text=f"処理中: {name} さんの写真を読み込んでいます...")
            status_label.update()
            
            for filename in os.listdir(person_dir):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_path = os.path.join(person_dir, filename)
                    image = face_recognition.load_image_file(image_path)
                    face_locations = face_recognition.face_locations(image, model="hog")
                    encodings = face_recognition.face_encodings(image, face_locations)

                    if len(encodings) > 0:
                        known_encodings.append(encodings[0])
                        known_names.append(name)
                    # エラー処理は簡略化

        # --- モデルの学習 ---
        le = LabelEncoder()
        names_numeric = le.fit_transform(known_names)
        
        status_label.config(text="処理中: scikit-learnモデル（SVM）の学習を開始...")
        status_label.update()

        clf = SVC(kernel='linear', C=1, gamma='scale', probability=True)
        clf.fit(known_encodings, names_numeric)

        # --- モデルの保存 ---
        with open(MODEL_FILE, 'wb') as f:
            pickle.dump((clf, le), f)

        messagebox.showinfo("成功", f"学習済みモデルを {MODEL_FILE} に保存しました。\n学習完了！")
        status_label.config(text="完了: 新しいモデルが保存されました。")
        
    except Exception as e:
        messagebox.showerror("エラー", f"学習中にエラーが発生しました: {e}")
        status_label.config(text="エラーが発生しました。")
        
# --- 3. Tkinter GUI の設定 ---

def create_gui():
    root = tk.Tk()
    root.title("モデル学習ツール")
    root.geometry("400x200")

    # ステータス表示ラベル
    status_label = tk.Label(root, text="待機中...", pady=10)
    status_label.pack()

    # 学習開始ボタン
    train_button = tk.Button(
        root,
        text="モデル学習開始",
        command=lambda: run_training_logic(status_label),
        font=('Helvetica', 12),
        bg='lightblue',
        padx=20,
        pady=10
    )
    train_button.pack(pady=20)
    
    # 訓練フォルダのパス表示 (オプション)
    dir_label = tk.Label(root, text=f"訓練データフォルダ: {TRAIN_DIR}")
    dir_label.pack()

    root.mainloop()

if __name__ == "__main__":
    create_gui()