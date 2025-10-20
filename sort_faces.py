import face_recognition
import numpy as np
import os
import pickle
from collections import defaultdict
import shutil # ファイル操作用

# --- 1. 設定 ---

# テスト/振り分け対象の画像が入っているフォルダ名
TEST_DIR = "test_data"
# 学習済みモデルファイル
MODEL_FILE = "face_classifier_model.pkl"
# 識別結果を振り分けるフォルダ
OUTPUT_DIR = "sorted_output"

# 識別のしきい値 (この数値が小さいほど、厳密な一致が求められる)
# 0.5〜0.6程度が一般的。今回は分類器を使うため、一旦予測結果を信頼します。
# 確信度が低い場合は「Unknown」として扱うオプションも後で追加可能です。

# --- 2. モデルとエンコーダーの読み込み ---

try:
    with open(MODEL_FILE, 'rb') as f:
        (clf, le) = pickle.load(f)
    print("✅ モデルとラベルエンコーダーの読み込みに成功しました。")
except FileNotFoundError:
    print(f"🚨 エラー: モデルファイル ({MODEL_FILE}) が見つかりません。学習を実行してください。")
    exit()

# 結果格納用のディクショナリ。人物名ごとにファイルリストを格納
# { 'Taro': ['img1.jpg', 'img2.jpg'], 'Unknown': ['img3.jpg'] }
sorted_results = defaultdict(list)
total_files_processed = 0

# --- 3. テストデータの処理と識別 ---

print(f"\n--- 3. 識別処理を開始します ({TEST_DIR} フォルダ内の画像を処理) ---")

if not os.path.isdir(TEST_DIR):
    print(f"🚨 警告: 振り分け対象フォルダ '{TEST_DIR}' が見つかりません。作成して画像を入れてください。")
    exit()

for filename in os.listdir(TEST_DIR):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(TEST_DIR, filename)
        total_files_processed += 1
        
        try:
            # 画像の読み込み
            image = face_recognition.load_image_file(image_path)
            # 顔の位置を検出 (高速なhogモデルを使用)
            face_locations = face_recognition.face_locations(image, model="hog")
            
            # 顔が検出された場合
            if len(face_locations) > 0:
                # 検出された全ての顔の特徴量を抽出 (このアプリでは最初の顔を識別対象とします)
                encodings = face_recognition.face_encodings(image, [face_locations[0]])
                
                # 特徴量をモデルが扱える形式に変換
                test_encoding = encodings[0].reshape(1, -1)
                
                # SVMモデルで人物を予測
                prediction_numeric = clf.predict(test_encoding)
                
                # 数値予測を元の人物名（ラベル）に戻す
                predicted_name = le.inverse_transform(prediction_numeric)[0]
                
                sorted_results[predicted_name].append(filename)
                
            else:
                # 顔が検出されなかった場合
                sorted_results["Unknown (No Face)"].append(filename)

        except Exception as e:
            # 画像ファイル破損などのエラー処理
            print(f"⚠️ ファイル {filename} の処理中にエラーが発生しました: {e}")
            sorted_results["Unknown (Error)"].append(filename)


# --- 4. 識別結果の提示 (コア要件) ---

print("\n" + "="*50)
print(f"🎯 識別結果の概要 ({total_files_processed} ファイル処理済み)")
print("="*50)

# 人物名ごとに結果をソートして表示
for name, files in sorted_results.items():
    print(f"\n👤 **人物名: {name} ({len(files)} 枚)**")
    
    # 簡潔にリスト表示
    print("  [ファイル一覧]:")
    # ファイル名が多い場合は、表示を一部省略
    if len(files) > 5:
        print(f"    - {', '.join(files[:5])}, ... ({len(files)-5} more)")
    else:
        print(f"    - {', '.join(files)}")

# --- 5. フォルダへの振り分け (副次的な機能) ---

print("\n--- 5. ファイルの振り分け (オプション) ---")

if total_files_processed > 0:
    # 出力ディレクトリを作成
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    for name, files in sorted_results.items():
        # Unknownファイルも専用フォルダへ
        output_dir = os.path.join(OUTPUT_DIR, name)
        os.makedirs(output_dir, exist_ok=True)
        
        for filename in files:
            source_path = os.path.join(TEST_DIR, filename)
            dest_path = os.path.join(output_dir, filename)
            
            # ファイルをコピーして振り分け（移動したい場合はshutil.moveに変更）
            shutil.copy(source_path, dest_path) 
    
    print(f"✅ 識別結果に基づき、ファイルを '{OUTPUT_DIR}' フォルダ内にコピーしました。")
    print("==================================================")
else:
    print("処理対象の画像が見つかりませんでした。")