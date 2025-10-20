import face_recognition
import numpy as np
import os
import pickle
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder


# --- 1.設定と準備 ---


TRAIN_DIR = "train_data"

ENCODINGS_FILE = "face_encodings.pkl"

MODEL_FILE = "face_classifier_model.pkl"



known_encodings = []
known_names = []


print("--- 1.学習データの読み込みと特徴量抽出を開始します ---")


for name in os.listdir(TRAIN_DIR):

    if name.startswith('.'):
        continue


    person_dir = os.path.join(TRAIN_DIR, name)
    if not os.path.isdir(person_dir):
        continue
    print(f"\n[処理中]: {name} さんの写真を読み込んでいます...")


    for filename in os.listdir(person_dir):

        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(person_dir, filename)

            # --- 追加デバッグ ---
            print(f"  [DEBUG] ファイルを読み込み中: {image_path}")
            # --- ここまで ---


            image = face_recognition.load_image_file(image_path)


            face_locations = face_recognition.face_locations(image, model="hog")


            encodings = face_recognition.face_encodings(image, face_locations)


            if len(encodings) > 0:

                known_encodings.append(encodings[0])

                known_names.append(name)
                # print(f"  [成功] {filename}: 特徴量を抽出しました")

            else:
                print(f"  [警告] {filename}: 顔が検出されませんでした。スキップします。")


print("\n--- 2.特徴量の保存 ---")

with open(ENCODINGS_FILE, 'wb') as f:
    pickle.dump((known_encodings, known_names), f)
print(f"特徴量トラベルを {ENCODINGS_FILE} に保存しました。")

# --- 3. モデルの学習 ---

# ラベル（人物名）を機械学習モデルが扱える数値に変換 (Taro:0, Hanako:1, ...)
le = LabelEncoder()
names_numeric = le.fit_transform(known_names)


print("\n--- 3. scikit-learnモデル（SVM）の学習を開始します ---")

# サポートベクターマシン（SVM）分類器を初期化
# gamma='scale'はscikit-learnの推奨設定
# Cは正則化パラメータ。大きくすると訓練データへの適合度が上がるが過学習の可能性も。
clf = SVC(kernel='linear', C=1, gamma='scale', probability=True)

# 特徴量ベクトル（known_encodings）と数値ラベル（names_numeric）を使って学習
clf.fit(known_encodings, names_numeric)

print("--- 4. 学習完了とモデルの保存 ---")
# 学習済みモデル（分類器とラベルエンコーダー）を保存
with open(MODEL_FILE, 'wb') as f:
    pickle.dump((clf, le), f)

print(f"学習済みモデルとエンコーダーを {MODEL_FILE} に保存しました。")
print("モデルの学習と保存が完了しました！")
