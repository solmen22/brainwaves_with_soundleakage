import wave
import pyaudio
import numpy as np  
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
import os

#保存するスペクトルグラム画像の名前指定
Attempts = 20
SpectrogramName = "Phone-D/Phone-D_try"+str(Attempts)+".png"


#音声ファイル読み込み
# wav_file_path = "C:/Users/rushi/myresearch/my_research_b4_data/20kHz.wav"
wav_file_path = "C:/Users/rushi/myresearch/my_research_b4_data/21kHz.wav"

wf = wave.open(wav_file_path, 'rb')

#音声ファイルの情報取得
SamplingRate = wf.getframerate()        #サンプリング周波数Hz
Channel = wf.getnchannels()             #モノラルはすべてのスピーカーから同じ音を再生する。ステレオは左右で異なる音を作成
SampleWidth = wf.getsampwidth()
Duration = 2.0      #再生、録音時間を指定
Chunk = 1024        #バッファサイズの指定


# PyAudio再生
p = pyaudio.PyAudio()   #pyaudioのインスタンスを作成
output_stream = p.open(format=p.get_format_from_width(SampleWidth),     #16bit整数フォーマットで再生
                channels=Channel,                 #モノラル(1チャンネル)
                rate=SamplingRate,         #サンプリングレート
                output=True)                #出力用として使用することを指定

input_stream = p.open(format=pyaudio.paInt16,
                      channels = Channel,
                      rate = SamplingRate,
                      input = True,
                      frames_per_buffer = Chunk)

print("再生と録音を開始...")
RecordedData = []     #音声データを格納するリスト

total_chunks = int(SamplingRate / Chunk * Duration)     #チャンクを繰り返す回数

for _ in range(total_chunks):
    data_chunk = wf.readframes(Chunk)
    if len(data_chunk) == 0:
        break  # WAVファイルが短すぎたら終了
    output_stream.write(data_chunk)
    input_data = input_stream.read(Chunk)
    RecordedData.append(input_data)
    
print("録音完了。スペクトログラム画像を生成中...")

# 録音データをNumPy配列に変換（int16）
audio_data = np.frombuffer(b''.join(RecordedData), dtype=np.int16)      

# NumPy配列: audio_data（録音データ）, SamplingRate（Hz）を使う前提

# スペクトログラム計算
frequencies, times, Sxx = spectrogram(audio_data, fs=SamplingRate, nperseg=2048, noverlap=1024)

# 周波数範囲 (19800〜20200Hz) に対応するインデックスを取得
freq_mask = (frequencies >= 19800) & (frequencies <= 20200)
frequencies_trimmed = frequencies[freq_mask]
Sxx_trimmed = Sxx[freq_mask, :]

# 画像描画設定（80x45ピクセル）
fig = plt.figure(figsize=(10, 10))
plt.pcolormesh(times, frequencies_trimmed, 10 * np.log10(Sxx_trimmed), shading='gouraud', cmap='jet')
plt.axis('off')
plt.tight_layout(pad=0)


# フォルダ部分だけを抽出
folder_path = os.path.dirname(SpectrogramName)

# フォルダが存在しない場合は作成
# フルパスを作成
full_path = os.path.join("C:/Users/rushi/myresearch/my_research_b4_data/test_audio_spectrogram_data/", SpectrogramName)

# 保存先のディレクトリを作成（なければ）
os.makedirs(os.path.dirname(full_path), exist_ok=True)
# 画像保存
plt.savefig("C:/Users/rushi/myresearch/my_research_b4_data/test_audio_spectrogram_data/" + SpectrogramName)
plt.show()

# ストリームを閉じる
output_stream.stop_stream()
output_stream.close()
input_stream.stop_stream()
input_stream.close()
p.terminate()