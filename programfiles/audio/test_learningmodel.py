# モジュールのインポート
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, models
import numpy as np
from tensorflow.keras.preprocessing import image

#python 3.10.11で実行

#画像サイズの決定
img_height = 224
img_width = 224
#バッチサイズ(1度に処理する画像の数)
batch_size = 32
Hz = "20kHz"

train_ds = tf.keras.utils.image_dataset_from_directory(     #トレーニング用データセットをフォルダから自動的に読み込む
    "C:/Users/rushi/research_dataandprogram/my_research_b4_data/test_audio_spectrogram_data/"+ Hz,
    validation_split=0.2,   #全体の20%をテスト用にする
    subset="training",      
    seed=123,               #分割の方法を固定
    image_size=(img_height, img_width),
    batch_size=batch_size
)

val_ds = tf.keras.utils.image_dataset_from_directory(       #検証用データセットをフォルダから自動的に読み込む
    "C:/Users/rushi/research_dataandprogram/my_research_b4_data/test_audio_spectrogram_data/" + Hz,
    validation_split=0.2,   #全体の20%をテスト用にする
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

class_names = train_ds.class_names      #データフォルダの各サブフォルダをクラス名として取得
print("分類ラベル:", class_names)

base_model = EfficientNetB0(input_shape=(img_height, img_width, 3),     #画像認識に強い事前学習積みの出るEfficientNetB0を使用
                            include_top=False,      #上位の分類層は取り除く
                            weights='imagenet')

# EfficientNetB0 は ImageNet データで学習済みなので、猫・犬・飛行機などの画像から「輪郭」「質感」「パターン」などの特徴を検出する能力を持っています。

# あなたは音声から作った スペクトログラム画像を使って、ジェスチャーを分類したい。

# 画像の「低レベル特徴」はスペクトログラムにも有効なので、その特徴抽出器はそのまま利用したい。

# そこで、「EfficientNet の重みは学習させずに固定（＝凍結）」し、最後の Dense 層だけを学習させてあなたのタスクに合わせるのです。

base_model.trainable = False  # 転移学習のため凍結


model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.2),
    layers.Dense(len(class_names), activation='softmax')    #クラス数分だけ出力層(softmaxで出力)
])

model.compile(optimizer='adam',     #高性能な最適化アルゴリズムを使用
              loss='sparse_categorical_crossentropy',       #クラスラベルが整数であるときに使用する損失関数
              metrics=['accuracy'])     #精度を評価指標とする

epochs = 30     #学習を10回繰り返す

history = model.fit(        #トレーニングデータと検証データを使用して10エポックで学習
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)

def predict_image(img_path):    #新しい画像を分類するための関数
    img = image.load_img(img_path, target_size=(img_height, img_width))     #指定された画像ファイルを読み込み、モデル入力サイズにリサイズ
    img_array = image.img_to_array(img)       #PIL形式からNumpy配列に変換
    img_array = tf.expand_dims(img_array, 0)  # バッチ次元を追加

    predictions = model.predict(img_array)    #モデルで推論を実行し、各クラスの確率を取得
    #最も確率の高いクラストその確率を求める
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = 100 * np.max(predictions[0])

    print(f"推定結果: {predicted_class} ({confidence:.2f}%)")   #推定結果を出力
    return predicted_class