import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from PIL import Image

#python 3.10.11で実行

Hz = '20kHz'
# データセットパス
DATASET_PATH = "C:/Users/rushi/myresearch/my_research_b4_data/test_audio_spectrogram_data/"+ Hz

# データとラベルを格納
Imagedata = []
Label_data = []

# 各フォルダ（ラベル）ごとに画像を読み込み
for label in os.listdir(DATASET_PATH):      #サブフォルダの名前を1つずつ取得
    label_path = os.path.join(DATASET_PATH, label)      #サブフォルダのパスを取得
    
    if not os.path.isdir(label_path):       #フォルダでないときスキップ
        continue
    for filename in os.listdir(label_path):     #サブフォルダ内のデータをすべて探索
        if filename.endswith(".png"):       #pngファイルの時
            img_path = os.path.join(label_path, filename)       #画像へのパスを取得
            img = Image.open(img_path).convert("L").resize((64, 64))  # グレースケール(L)＋リサイズ
            img_array = np.array(img).flatten()  # flattenして1次元ベクトルに
            Imagedata.append(img_array)
            Label_data.append(label)
            
#データをNumpy配列に変換して機械学習で扱いやすい形式にする
Imagedata = np.array(Imagedata)
Label_data = np.array(Label_data)

# ラベルを数値に変換
le = LabelEncoder()
y_encoded = le.fit_transform(Label_data)

# 特徴量のスケーリング
scaler = StandardScaler()
Imagedata_scaled = scaler.fit_transform(Imagedata)

# 訓練とテストに分割
Imagedata_train, Imagedata_test, y_train, y_test = train_test_split(Imagedata_scaled, y_encoded, test_size=0.2, random_state=42)

# SVMモデルの作成と学習
clf = SVC(kernel='rbf', C=1.0, gamma='scale')   #C:正則化パラメータ(誤差の許容具合大=>許容しない)
clf.fit(Imagedata_train, y_train)

# 予測と評価
y_pred = clf.predict(Imagedata_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))
