from pythonosc import dispatcher, osc_server

def handler(address, *args):
    print(f"OSC Address: {address} | Values: {args}")

# ポート5005で受信
dispatcher = dispatcher.Dispatcher()
dispatcher.map("/*", handler)  # すべてのアドレスを受信

ip = "0.0.0.0"  # 自分のPCの全インターフェースで受信
port = 49152
server = osc_server.ThreadingOSCUDPServer((ip, port), dispatcher)
print(f"Listening on {ip}:{port}")
server.serve_forever()