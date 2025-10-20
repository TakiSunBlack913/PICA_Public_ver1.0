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
# 識別のしきい値 (この確率未満の場合、Unknownとして分類する)
# 0.70 = 70% の確信度を最低限要求する
CONFIDENCE_THRESHOLD = 0.70

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
                # 検出された最初の顔の特徴量を抽出
                encodings = face_recognition.face_encodings(image, [face_locations[0]])
                test_encoding = encodings[0].reshape(1, -1)
                
                # --- Unknown識別と信頼度計算のロジック ---
                
                # 1. 各クラス（人物）に対する予測確率を取得
                probabilities = clf.predict_proba(test_encoding)[0]
                
                # 2. 最も高い確率（確信度）と、そのインデックスを取得
                max_proba = np.max(probabilities)
                max_index = np.argmax(probabilities)
                
                # 3. しきい値に基づいて人物名を決定
                if max_proba >= CONFIDENCE_THRESHOLD:
                    # 確信度がしきい値以上の場合、人物を特定
                    prediction_numeric = np.array([max_index])
                    predicted_name = le.inverse_transform(prediction_numeric)[0]
                else:
                    # 確信度が低い場合、Unknownとする
                    predicted_name = "Unknown"
                    
                # 予測結果を格納 (ファイル名と確信度をタプルで格納)
                sorted_results[predicted_name].append((filename, max_proba))
                
            else:
                # 顔が検出されなかった場合
                # 確信度0.0として格納
                sorted_results["Unknown (No Face)"].append((filename, 0.0))

        except Exception as e:
            # 画像ファイル破損などのエラー処理
            print(f"⚠️ ファイル {filename} の処理中にエラーが発生しました: {e}")
            sorted_results["Unknown (Error)"].append((filename, 0.0)) # エラーの場合もタプル形式で格納


# --- 4. 識別結果の提示 (コア要件) ---

print("\n" + "="*50)
print(f"🎯 識別結果の概要 ({total_files_processed} ファイル処理済み)")
print("="*50)

# 人物名ごとに結果をソートして表示
for name, files_with_proba in sorted_results.items(): # ✅ OK (files_with_probaを使用)
    print(f"\n👤 **人物名: {name} ({len(files_with_proba)} 枚)**")

    # 簡潔にリスト表示
    print("  [ファイル一覧]:")

    # 確信度を表示するようにリストを整形
    display_files = []
    # files_with_proba は [(filename, proba), ...] のタプルリスト
    for filename, proba in files_with_proba:
        # Unknown/エラーの場合、確信度は表示しない
        if name in ["Unknown (No Face)", "Unknown (Error)", "Unknown"]:
            confidence_str = ""
        else:
            # それ以外の場合、確信度をパーセンテージで表示
            confidence_str = f" ({proba * 100:.2f}%)"
            
        display_files.append(f"{filename}{confidence_str}")


    # ファイル名が多い場合は、表示を一部省略
    if len(files_with_proba) > 5: # ✅ 修正: files_with_proba を使用
        print(f"    - {', '.join(display_files[:5])}, ... ({len(files_with_proba)-5} more)") # ✅ 修正: display_files と files_with_proba を使用
    else:
        print(f"    - {', '.join(display_files)}") # ✅ 修正: display_files を使用


# --- 5. フォルダへの振り分け (副次的な機能) ---

print("\n--- 5. ファイルの振り分け (オプション) ---")

if total_files_processed > 0:
    # 出力ディレクトリを作成
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # ✅ 修正: files_with_proba を使用
    for name, files_with_proba in sorted_results.items(): 
        # Unknownファイルも専用フォルダへ
        output_dir = os.path.join(OUTPUT_DIR, name)
        os.makedirs(output_dir, exist_ok=True)
        
        # ✅ 修正: タプルから filename と proba を展開
        for filename, proba in files_with_proba: 
            source_path = os.path.join(TEST_DIR, filename)
            dest_path = os.path.join(output_dir, filename)
            
            # ファイルをコピーして振り分け（移動したい場合はshutil.moveに変更）
            shutil.copy(source_path, dest_path) 
    
    print(f"✅ 識別結果に基づき、ファイルを '{OUTPUT_DIR}' フォルダ内にコピーしました。")
    print("==================================================")
else:
    print("処理対象の画像が見つかりませんでした。")