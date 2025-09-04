import time
import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# 監視するフォルダ
WATCH_FOLDER = "C:/Users/rushi/research_dataandprogram/my_research_b4_data/brainwave_data_first_test_controlhair"
def wait_until_file_ready(path, timeout=10):
    """ファイルが使用中でなくなるまで待つ"""
    start_time = time.time()
    while True:
        try:
            os.rename(path, path)  # 自分自身にリネームしてテスト
            return True
        except PermissionError:
            if time.time() - start_time > timeout:
                raise TimeoutError(f"{path} が使用中のままです")
            time.sleep(0.1)  # 100ms 待つ

# 新しいファイルが作成されたときに呼ばれるハンドラー
class CSVHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.src_path.endswith(".csv"):
            original_path = event.src_path
            wait_until_file_ready(original_path)  # ファイルが使えるまで待つ
            new_name = f"EEG_{int(time.time()*1000)}.csv"
            new_path = os.path.join(os.path.dirname(original_path), new_name)
            os.rename(original_path, new_path)
            print(f"リネーム完了: {new_path}")
if __name__ == "__main__":
    event_handler = CSVHandler()
    observer = Observer()
    observer.schedule(event_handler, WATCH_FOLDER, recursive=False)
    observer.start()
    print(f"Watching folder: {WATCH_FOLDER}")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()