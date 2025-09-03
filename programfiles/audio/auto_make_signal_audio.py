import wave
import pyaudio
import numpy as np  
import matplotlib.pyplot as plt
from scipy.signal import spectrogram,butter,filtfilt
import os

# ジェスチャと周波数の選択肢
GestureList = {
    1: "FingerSewipe-D/FingerSewipe-D_try",      
    2: "Grip-C/Grip-C_try",
    3: "handswipe-TF/handswipe-TF_try",
    4: "Phone-D/Phone-D_try",
    5: "Sqeeze-C/Sqeeze-C_try",
    6: "Twist-OF/Twist-OF_try",
    7: "Twist-TF/Twist-TF_try",
    8: "Glasses-on/Glasses-on_try",
    9: "Glasses-off/Glasses-off_try"
}

HzNumberList = {
    20: "C:/Users/rushi/research_dataandprogram/my_research_b4_data/20kHz.wav",
    21: "C:/Users/rushi/research_dataandprogram/my_research_b4_data/21kHz.wav"
}

# ---------- 入力受付 ----------
print("【ジェスチャを選択してください】")
for key, val in GestureList.items():
    gesture_name = val.split('/')[0]
    print(f"{key}: {gesture_name}")
GestureNumber = int(input("ジェスチャ番号を入力: "))

print("\n【周波数を選択してください】")
for hz in HzNumberList:
    print(f"{hz}kHz")
HzNumber = int(input("Hz番号を入力 (20 または 21): "))

# パスと保存名ベースを取得
gesture_prefix = GestureList[GestureNumber]
wav_file_path = HzNumberList[HzNumber]

#関数の定義
def bandstop_filter(before_cut_audio_data, SamplingRate, lowcut, highcut, order = 4):   #帯域阻止フィルタの関数
    Nyq = 0.5*SamplingRate      #ナイキスト周波数
    Low = lowcut/Nyq        #butterに渡す周波数帯域はナイキスト周波数に対する比率で渡さなければならない
    High = highcut/Nyq
    b,a = butter(order, [Low, High], btype ='bandstop')      #oderはフィルタ次数
    filtered = filtfilt(b, a, before_cut_audio_data)   #位相遅延なしでフィルタ処理を行う
    return filtered
    

# ---------- 試行開始 ----------
Attempts = 1  # 試行番号（最大20）

while Attempts <= 20:
    SpectrogramName = f"{gesture_prefix}{Attempts}.png"

    # 音声ファイル読み込み
    wf = wave.open(wav_file_path, 'rb')

    # 音声ファイルの情報取得
    SamplingRate = wf.getframerate()    #サンプリングレートの取得
    Channel = wf.getnchannels()     #チャンネル数(左右同じ音を出すか、左右違う音を出すか)
    SampleWidth = wf.getsampwidth()
    Duration = 2.2     #録音時間
    Chunk = 1024      #バッファサイズの指定

    # PyAudio設定
    p = pyaudio.PyAudio()
    output_stream = p.open(format=p.get_format_from_width(SampleWidth),     #音声の出力の設定
                           channels=Channel,
                           rate=SamplingRate,
                           output=True)
    input_stream = p.open(format=pyaudio.paInt16,         #音声の録音の設定
                          channels=Channel,
                          rate=SamplingRate,
                          input=True,
                          frames_per_buffer=Chunk)

    print(f"\n[試行 {Attempts}/20] 再生と録音を開始...")
    RecordedData = []
    total_chunks = int(SamplingRate / Chunk * Duration)

    for _ in range(total_chunks):
        data_chunk = wf.readframes(Chunk)   #バッファ分の出力音声データを抽出する
        if len(data_chunk) == 0:
            break
        output_stream.write(data_chunk)     #バッファ分を出力する
        input_data = input_stream.read(Chunk)   #バッファ分の録音をする
        RecordedData.append(input_data)     #録音したデータを保存する

    print("録音完了。スペクトログラム画像を生成中...")


    before_cut_audio_data = np.frombuffer(b''.join(RecordedData), dtype=np.int16)      #録音データをnumpy配列に変換する
    audio_data = bandstop_filter(before_cut_audio_data, SamplingRate = SamplingRate, lowcut = HzNumber*1000-10, highcut = HzNumber*1000 + 10)
    cut_samples = int(SamplingRate * 0.2)   #最初の0.4秒をカット
    audio_data = audio_data[cut_samples:]
   
    # スペクトログラム計算
    frequencies, times, Sxx = spectrogram(audio_data, fs=SamplingRate, nperseg = 2048, noverlap = 1024)     #nperseg:フーリエ変換に使用するサンプル数、noverlap:隣接するセグメント間の重なり

    db_spectrogram = 10*np.log10(Sxx)
    # 指定周波数のみ抽出（例：19.8kHz ～ 20.2kHz）
    freq_mask = (frequencies >= HzNumber * 1000 - 200) & (frequencies <= HzNumber * 1000 + 200)
  
    frequencies_trimmed = frequencies[freq_mask]
    db_spectrogram_trimmed = db_spectrogram[freq_mask, :]
    
    # imshow で描画（vmin, vmax 指定）
    fig = plt.figure(figsize=(10, 10))
    plt.imshow(db_spectrogram_trimmed,
               extent=[times.min(), times.max(), frequencies_trimmed.min(), frequencies_trimmed.max()],
               aspect='auto',
               origin='lower',
               cmap='jet',
               interpolation='bilinear',  # 補間を追加して滑らかに
               vmin=-28, vmax=0)  # 色の範囲を手動で指定
    plt.axis('off')
    plt.tight_layout(pad=0)
    full_path = os.path.join("C:/Users/rushi/research_dataandprogram/my_research_b4_data/test_audio_spectrogram_data/", SpectrogramName)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    plt.savefig(full_path, bbox_inches='tight', pad_inches=0)
    plt.show()
    plt.close()
    
 
    print(f"スペクトログラム画像を保存しました: {full_path}")

    # ストリーム終了
    output_stream.stop_stream()
    output_stream.close()
    input_stream.stop_stream()
    input_stream.close()
    p.terminate()
    wf.close()

    # 次の試行か再試行か確認
    while True:
        answer = input("次の試行に進みますか？ (y/n): ").strip().lower()
        if answer == 'y':
            Attempts += 1
            break
        elif answer == 'n':
            print(f"[試行 {Attempts}] を再実行します。")
            break
        else:
            print("無効な入力です。'y' または 'n' を入力してください。")

