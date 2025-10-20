# app.py の先頭のインポートを確認
import streamlit as st
import numpy as np
import os
import pickle
import face_recognition
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from io import BytesIO
from PIL import Image # これらが全てインポートされていること
import io           # この行が特に重要

# --- 1. 設定 ---
# 🚨 ここを修正: ご自身のプロジェクトの絶対パスに置き換えてください！############################################
PROJECT_ROOT = "/Users/takisunblack/Documents/PICA/PICA-Person-Identification-and-Classification-APP"
######################################################################################################

#MODEL_FILE = "face_classifier_model.pkl"
MODEL_FILE = os.path.join(PROJECT_ROOT, "face_classifier_model.pkl")
# 識別のしきい値 (この確率未満の場合、Unknownとして分類する)
CONFIDENCE_THRESHOLD = 0.70 
# 複数顔を識別するため、結果格納用のリストはグローバルではなく関数内で管理します

# --- 2. モデルのロード (キャッシュ化) ---
# Streamlitのキャッシュ機能を使って、モデルの読み込みを高速化します
@st.cache_resource
def load_model():
    """モデルとエンコーダーを読み込み、正常性チェックを行う"""
    try:
        with open(MODEL_FILE, 'rb') as f:
            (clf, le) = pickle.load(f)
        return clf, le
    except FileNotFoundError:
        st.error(f"🚨 エラー: モデルファイル ({MODEL_FILE}) が見つかりません。train_model.pyを実行してください。")
        return None, None
    except Exception as e:
        st.error(f"🚨 モデルの読み込み中に予期せぬエラーが発生しました: {e}")
        return None, None

# モデルとエンコーダーをロード
clf, le = load_model()

# --- 3. 識別処理関数 ---
def identify_faces(image_np, uploaded_file_name, clf, le, CONFIDENCE_THRESHOLD):
    """
    画像内の全ての顔を検出し、識別処理を実行する

    Returns:
        list: [(ファイル名, 人物名, 信頼度, 顔座標)] のリスト
    """
    results_list = []
    
    # 1. 画像から全ての顔の位置を検出
    face_locations = face_recognition.face_locations(image_np, model="hog")
    
    # 2. 検出された全ての顔の特徴量を抽出
    encodings = face_recognition.face_encodings(image_np, face_locations)

    if len(encodings) > 0:
        # 検出された顔ごとに識別処理を実行
        for face_encoding, face_location in zip(encodings, face_locations):
            
            test_encoding = face_encoding.reshape(1, -1)
            
            # 信頼度計算ロジック
            probabilities = clf.predict_proba(test_encoding)[0]
            max_proba = np.max(probabilities)
            max_index = np.argmax(probabilities)
            
            # しきい値に基づいて人物名を決定
            if max_proba >= CONFIDENCE_THRESHOLD:
                prediction_numeric = np.array([max_index])
                predicted_name = le.inverse_transform(prediction_numeric)[0]
            else:
                predicted_name = "Unknown"
                
            # 結果をリストに追加 (顔の位置情報も追加)
            results_list.append({
                'ファイル名': uploaded_file_name,
                '人物名': predicted_name,
                '信頼度': max_proba,
                '画像データ': image_np, # サムネイル表示用
                '顔座標': face_location # 顔の切り抜き用 (top, right, bottom, left)
            })
            
    else:
        # 顔が検出されなかった場合
        results_list.append({
            'ファイル名': uploaded_file_name,
            '人物名': "Unknown (No Face)",
            '信頼度': 0.0,
            '画像データ': image_np,
            '顔座標': None
        })
        
    return results_list


# --- 4. メインアプリ ---
st.title("👤 顔画像ソート＆識別アプリ (Streamlit)")
st.caption(f"学習人数: {len(le.classes_)}人, 識別しきい値: {CONFIDENCE_THRESHOLD*100:.0f}%")

if clf is None:
    st.stop() # モデルが読み込めなかったら処理を停止

st.header("検証画像のアップロード")

# Streamlitのファイルアップローダー (ドラッグ＆ドロップ対応)
uploaded_files = st.file_uploader(
    "検証したい画像をドラッグ＆ドロップ、またはクリックして選択してください (JPG/PNGのみ)",
    type=['jpg', 'jpeg', 'png'],
    accept_multiple_files=True
)

if uploaded_files:
    
    # 結果格納用のリストを初期化
    all_results = []
    
    # プログレスバーの表示
    progress_bar = st.progress(0)
    
    # 処理開始メッセージ
    st.subheader(f"処理中: {len(uploaded_files)} 枚の画像を識別")

    for i, uploaded_file in enumerate(uploaded_files):
        
        try:
            # StreamlitのバイトデータをPILで読み込み、numpy配列に変換
            image_bytes = uploaded_file.read()
            image_pil = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            image_np = np.array(image_pil) 
            
            # 識別処理を実行
            results_for_file = identify_faces(image_np, uploaded_file.name, clf, le, CONFIDENCE_THRESHOLD)
            all_results.extend(results_for_file)
        
        except Exception as e:
            st.warning(f"⚠️ ファイル {uploaded_file.name} の処理中にエラーが発生しました: {e}")
            all_results.append({
                'ファイル名': uploaded_file.name,
                '人物名': "Unknown (Error)",
                '信頼度': 0.0,
                '画像データ': None,
                '顔座標': None
            })

        # プログレスバーの更新
        progress_bar.progress((i + 1) / len(uploaded_files))

    st.success(f"処理が完了しました。合計 {len(all_results)} 個の顔（またはファイル）を識別しました。")


    # --- 5. 結果の表示（サムネイルと詳細） ---
    
    # 人物名ごとに結果をグループ化
    grouped_results = defaultdict(list)
    for result in all_results:
        # 信頼度を文字列で表示用に変換
        result['信頼度_str'] = f"{result['信頼度']*100:.2f}%" if result['人物名'] not in ["Unknown (No Face)", "Unknown (Error)", "Unknown"] else "---"
        grouped_results[result['人物名']].append(result)
        
    st.header("検証結果（サムネイル）")

    # 人物名ごとにコンテナで区切って表示
    for name, group in grouped_results.items():
        # Unknownファイルも含むため、合計枚数ではなく顔の数を表示
        st.subheader(f"👤 {name} ({len(group)} 個の顔を検出)")
        
        # Streamlitのcolumn機能で画像を横に並べる
        cols = st.columns(5) # 5列で表示

        for i, result in enumerate(group):
            with cols[i % 5]: # 5枚ごとに次の行へ
                
                # 顔の切り抜きロジック
                image_np = result.get('画像データ')
                face_location = result.get('顔座標')
                
                if image_np is not None and face_location is not None:
                    # top, right, bottom, left
                    top, right, bottom, left = face_location
                    
                    # PILを使って画像を切り抜き
                    image_pil = Image.fromarray(image_np)
                    # 切り抜き範囲を少し広げる（パディング）
                    padding = 50
                    
                    # 座標が画像範囲を超えないようにクリッピング
                    img_width, img_height = image_pil.size
                    crop_area = (
                        max(0, left - padding), 
                        max(0, top - padding), 
                        min(img_width, right + padding), 
                        min(img_height, bottom + padding)
                    )
                    
                    cropped_face = image_pil.crop(crop_area)
                    
                    # キャプションの設定
                    caption_text = f"{result['ファイル名']}\n({result['信頼度_str']})"

                    st.image(cropped_face, caption=caption_text, use_column_width=True)
                
                else:
                    # 顔が検出されなかった、またはエラーの画像は、そのまま表示（エラーの場合は画像がNone）
                    caption_text = f"{result['ファイル名']}\n({result['人物名']})"
                    if image_np is not None:
                         st.image(image_np, caption=caption_text, use_column_width=True)
                    else:
                         st.text(caption_text)


    # --- 6. 識別結果の詳細テーブル (オプション) ---
    if all_results:
        st.header("詳細データ")
        
        # データフレーム作成
        df = pd.DataFrame(all_results)
        # 不要な列を削除し、信頼度をパーセンテージ表示に整形
        df['信頼度'] = df['信頼度'].apply(lambda x: f"{x*100:.2f}%")
        
        st.dataframe(df[['ファイル名', '人物名', '信頼度']], use_container_width=True)