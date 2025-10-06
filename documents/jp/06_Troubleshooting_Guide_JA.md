# トラブルシューティングガイド

## 目次
1. [クイック診断](#クイック診断)
2. [インストール問題](#インストール問題)
3. [実行時問題](#実行時問題)
4. [パフォーマンス問題](#パフォーマンス問題)
5. [ネットワーク・接続問題](#ネットワーク・接続問題)
6. [カメラ・メディア問題](#カメラ・メディア問題)
7. [モデル・AI関連問題](#モデル・AI関連問題)
8. [ブラウザ・フロントエンド問題](#ブラウザ・フロントエンド問題)
9. [システム固有の問題](#システム固有の問題)
10. [高度なデバッグ](#高度なデバッグ)

## クイック診断

### システムヘルスチェックスクリプト

以下のPythonスクリプトを作成して実行し、システムの状態を確認してください：

```python
# health_check.py
import sys
import platform
import subprocess
import importlib
import torch
import cv2
import requests
from pathlib import Path

def check_python_version():
    """Python バージョンチェック"""
    version = sys.version_info
    print(f"Python バージョン: {version.major}.{version.minor}.{version.micro}")
    
    if version.major != 3 or version.minor < 8:
        print("❌ Python 3.8以上が必要です")
        return False
    print("✅ Python バージョン OK")
    return True

def check_required_packages():
    """必要なパッケージの確認"""
    required_packages = {
        'fastapi': '0.100.0',
        'uvicorn': '0.20.0', 
        'ultralytics': '8.0.0',
        'torch': '2.0.0',
        'cv2': '4.8.0',
        'PIL': '10.0.0',
        'numpy': '1.21.0'
    }
    
    all_ok = True
    for package, min_version in required_packages.items():
        try:
            if package == 'cv2':
                module = importlib.import_module('cv2')
                version = cv2.__version__
            elif package == 'PIL':
                module = importlib.import_module('PIL')
                version = module.__version__
            else:
                module = importlib.import_module(package)
                version = getattr(module, '__version__', 'Unknown')
            
            print(f"✅ {package}: {version}")
        except ImportError:
            print(f"❌ {package}: インストールされていません")
            all_ok = False
        except Exception as e:
            print(f"⚠️ {package}: エラー - {str(e)}")
            
    return all_ok

def check_gpu_availability():
    """GPU利用可能性チェック"""
    try:
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"✅ GPU利用可能: {gpu_name} ({memory_gb:.1f}GB)")
            print(f"   GPU数: {gpu_count}")
            return True
        else:
            print("⚠️ GPU利用不可 - CPUモードで動作")
            return False
    except Exception as e:
        print(f"❌ GPU チェックエラー: {str(e)}")
        return False

def check_model_file():
    """YOLOモデルファイルの確認"""
    model_path = Path("yolo11x.pt")
    if model_path.exists():
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"✅ YOLOモデル存在: {size_mb:.1f}MB")
        return True
    else:
        print("❌ yolo11x.pt が見つかりません")
        print("   解決方法: from ultralytics import YOLO; YOLO('yolo11x.pt')")
        return False

def check_port_availability():
    """ポート8000の利用可能性チェック"""
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print("✅ アプリケーション実行中")
            return True
        else:
            print("⚠️ アプリケーションはアクセス可能ですが、レスポンスが異常です")
            return False
    except requests.exceptions.ConnectionError:
        print("✅ ポート8000利用可能（アプリケーション未実行）")
        return True
    except Exception as e:
        print(f"❌ ポートチェックエラー: {str(e)}")
        return False

def check_opencv_camera():
    """OpenCVカメラアクセステスト"""
    try:
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                height, width = frame.shape[:2]
                print(f"✅ カメラアクセス OK ({width}x{height})")
                cap.release()
                return True
            else:
                print("❌ カメラからフレーム読み取り失敗")
                cap.release()
                return False
        else:
            print("❌ カメラアクセス失敗")
            return False
    except Exception as e:
        print(f"❌ カメラテストエラー: {str(e)}")
        return False

def main():
    print("🔍 物体検出アプリケーション ヘルスチェック")
    print("=" * 50)
    
    checks = [
        ("Python バージョン", check_python_version),
        ("必要パッケージ", check_required_packages),
        ("GPU 利用可能性", check_gpu_availability),
        ("YOLO モデル", check_model_file),
        ("ポート 8000", check_port_availability),
        ("カメラアクセス", check_opencv_camera)
    ]
    
    results = []
    for name, check_func in checks:
        print(f"\n📋 {name} チェック中...")
        result = check_func()
        results.append((name, result))
    
    print("\n" + "=" * 50)
    print("📊 ヘルスチェック結果:")
    
    all_passed = True
    for name, result in results:
        status = "✅ OK" if result else "❌ 問題あり"
        print(f"   {name}: {status}")
        if not result:
            all_passed = False
    
    if all_passed:
        print("\n🎉 すべてのチェックが完了しました！")
        print("アプリケーションを開始できます: python app.py")
    else:
        print("\n⚠️ 問題が検出されました。上記のエラーを修正してください。")

if __name__ == "__main__":
    main()
```

実行方法：
```bash
python health_check.py
```

## インストール問題

### 1. パッケージインストールエラー

**問題:** `pip install -r requirements.txt` が失敗する

**解決方法:**

```bash
# Python・pipのアップグレード
python -m pip install --upgrade pip

# 仮想環境の作成（推奨）
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux  
source venv/bin/activate

# キャッシュクリアしてインストール
pip install --no-cache-dir -r requirements.txt

# 個別インストール（エラー時）
pip install fastapi>=0.100.0
pip install uvicorn[standard]>=0.20.0
pip install ultralytics>=8.0.0
```

**よくあるエラーと対策:**

**Microsoft Visual C++ エラー (Windows):**
```bash
# Visual Studio Build Tools をインストール
# または事前コンパイル済みパッケージを使用
pip install --only-binary=all torch torchvision
```

**macOS arm64 互換性問題:**
```bash
# Apple Silicon Mac 向け
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

**Linux CUDA エラー:**
```bash
# CUDA 対応PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 2. YOLO モデルダウンロード問題

**問題:** `yolo11x.pt` モデルのダウンロードが失敗

**解決方法:**

```python
# 手動ダウンロード
from ultralytics import YOLO

# 自動ダウンロード（推奨）
model = YOLO('yolo11x.pt')  # 初回実行時に自動ダウンロード

# 手動ダウンロード URL
# https://github.com/ultralytics/assets/releases/download/v0.0.0/yolo11x.pt
```

**プロキシ環境での対処:**
```python
import os
os.environ['http_proxy'] = 'http://proxy.company.com:8080'
os.environ['https_proxy'] = 'http://proxy.company.com:8080'

# または直接ダウンロード
import wget
wget.download('https://github.com/ultralytics/assets/releases/download/v0.0.0/yolo11x.pt')
```

### 3. 権限エラー

**問題:** ファイルアクセス権限エラー

**解決方法:**

```bash
# Windows (管理者権限でコマンドプロンプト実行)
# または
pip install --user -r requirements.txt

# macOS/Linux
sudo pip install -r requirements.txt
# または
pip install --user -r requirements.txt

# ディレクトリ権限修正
chmod 755 ./
chmod 644 *.py
```

## 実行時問題

### 1. アプリケーション起動エラー

**問題:** `python app.py` 実行時のエラー

**一般的なエラーと解決方法:**

**ポート使用中エラー:**
```bash
# ポート使用状況確認
# Windows
netstat -ano | findstr :8000
# macOS/Linux
lsof -i :8000

# プロセス終了
# Windows
taskkill /PID <PID> /F
# macOS/Linux
kill -9 <PID>

# 別ポートでの起動
uvicorn app:app --host 0.0.0.0 --port 8080
```

**モジュール未検出エラー:**
```python
# PYTHONPATH 設定
import sys
sys.path.append('.')

# または環境変数設定
export PYTHONPATH="${PYTHONPATH}:."
```

**CUDA エラー:**
```python
# CPUモード強制
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

# または app.py 内で
import torch
torch.cuda.is_available = lambda: False
```

### 2. メモリ不足エラー

**問題:** GPU/CPUメモリ不足

**解決方法:**

```python
# メモリ監視・クリーンアップ
import gc
import torch

def cleanup_memory():
    """メモリクリーンアップ"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

# 大きな画像のリサイズ
def resize_large_image(image_path, max_size=1024):
    from PIL import Image
    
    with Image.open(image_path) as img:
        if max(img.size) > max_size:
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        return img

# バッチサイズ削減
model = YOLO('yolo11x.pt')
model.conf = 0.5  # 信頼度閾値上げる
model.iou = 0.7   # NMS閾値調整
```

### 3. セッション管理エラー

**問題:** セッションが見つからない、期限切れエラー

**デバッグ方法:**

```python
# セッション状態確認
def debug_sessions():
    print("現在のセッション:")
    for session_id, session_data in sessions.items():
        print(f"  {session_id}: {session_data}")

# セッション手動クリーンアップ
def cleanup_expired_sessions():
    from datetime import datetime, timedelta
    
    current_time = datetime.now()
    expired_sessions = []
    
    for session_id, session_data in sessions.items():
        if current_time - session_data['created_at'] > timedelta(hours=1):
            expired_sessions.append(session_id)
    
    for session_id in expired_sessions:
        del sessions[session_id]
        print(f"期限切れセッション削除: {session_id}")

# 新しいセッション強制作成
def force_create_session():
    import uuid
    session_id = str(uuid.uuid4())
    sessions[session_id] = {
        'created_at': datetime.now(),
        'last_activity': datetime.now(),
        'data': {}
    }
    return session_id
```

## パフォーマンス問題

### 1. 検出速度が遅い

**問題:** 物体検出の処理時間が長い

**最適化方法:**

```python
# 1. モデル最適化
model = YOLO('yolo11x.pt')
model.fuse()  # モデル層の融合

# 2. GPU最適化
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    model.cuda()

# 3. 画像サイズ最適化
def optimize_image_size(image):
    # 640x640 が YOLO の最適サイズ
    return cv2.resize(image, (640, 640))

# 4. 信頼度閾値調整
model.conf = 0.7  # デフォルト 0.25 から上げる
model.iou = 0.5   # NMS 閾値

# 5. 推論モード設定
with torch.no_grad():
    results = model(image)
```

**より高度な最適化:**

```python
# TensorRT 最適化 (GPU)
model.export(format='engine')  # TensorRT エンジン生成
model = YOLO('yolo11x.engine')  # 最適化モデル使用

# ONNX 最適化
model.export(format='onnx')
import onnxruntime as ort
session = ort.InferenceSession('yolo11x.onnx')

# 量子化（精度とのトレードオフ）
model.export(format='onnx', int8=True)
```

### 2. WebSocket接続の遅延

**問題:** リアルタイム動画で遅延が発生

**解決方法:**

```python
# フレームスキッピング
class FrameSkipper:
    def __init__(self, skip_rate=2):
        self.skip_rate = skip_rate
        self.frame_count = 0
    
    def should_process(self):
        self.frame_count += 1
        return self.frame_count % (self.skip_rate + 1) == 0

# 非同期処理改善
import asyncio
from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor(max_workers=2)

async def process_frame_async(frame_data):
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        executor, 
        detection_service.detect_objects, 
        frame_data
    )
    return result

# バッファサイズ調整
@app.websocket("/ws/video/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await websocket.accept()
    
    # 送信バッファサイズ制限
    websocket._send_queue = asyncio.Queue(maxsize=2)
```

### 3. メモリリーク

**問題:** 長時間実行でメモリ使用量が増加

**対策:**

```python
import psutil
import os

class MemoryMonitor:
    def __init__(self, threshold_mb=1000):
        self.threshold_mb = threshold_mb
        self.process = psutil.Process(os.getpid())
    
    def check_memory(self):
        memory_mb = self.process.memory_info().rss / 1024 / 1024
        if memory_mb > self.threshold_mb:
            self.cleanup()
            print(f"メモリクリーンアップ実行: {memory_mb:.1f}MB")
        return memory_mb
    
    def cleanup(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# 定期クリーンアップ
import threading
import time

def periodic_cleanup():
    monitor = MemoryMonitor()
    while True:
        monitor.check_memory()
        time.sleep(300)  # 5分間隔

cleanup_thread = threading.Thread(target=periodic_cleanup, daemon=True)
cleanup_thread.start()
```

## ネットワーク・接続問題

### 1. CORS エラー

**問題:** ブラウザで CORS エラーが発生

**解決方法:**

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 開発時のみ。本番では特定ドメインを指定
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# より安全な設定（本番用）
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:8080",
        "https://yourdomain.com"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
```

### 2. LAN アクセス問題

**問題:** 同じネットワークの他デバイスからアクセスできない

**解決方法:**

```bash
# ホスト設定を 0.0.0.0 に変更
uvicorn app:app --host 0.0.0.0 --port 8000

# ファイアウォール設定確認
# Windows
netsh advfirewall firewall add rule name="Python App" dir=in action=allow protocol=TCP localport=8000

# macOS
sudo pfctl -d  # ファイアウォール一時無効

# Linux Ubuntu
sudo ufw allow 8000
```

**IP アドレス確認:**
```python
import socket

def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('10.255.255.255', 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP

print(f"アクセス URL: http://{get_local_ip()}:8000")
```

### 3. プロキシ環境での問題

**問題:** 企業ネットワークでモデルダウンロードが失敗

**解決方法:**

```python
import os

# プロキシ設定
os.environ['http_proxy'] = 'http://proxy.company.com:8080'
os.environ['https_proxy'] = 'http://proxy.company.com:8080'
os.environ['no_proxy'] = 'localhost,127.0.0.1'

# 認証付きプロキシ
os.environ['http_proxy'] = 'http://username:password@proxy.company.com:8080'

# SSL 証明書検証スキップ（非推奨、テスト時のみ）
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
```

## カメラ・メディア問題

### 1. カメラアクセス拒否

**問題:** ブラウザでカメラへのアクセスが拒否される

**解決方法:**

**ブラウザ設定:**
- Chrome: 設定 > プライバシーとセキュリティ > サイトの設定 > カメラ
- Firefox: 設定 > プライバシーとセキュリティ > 許可設定 > カメラ
- Safari: 設定 > Webサイト > カメラ

**HTTPS 要件:**
```bash
# 開発用SSL証明書生成
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes

# HTTPS サーバー起動
uvicorn app:app --host 0.0.0.0 --port 8000 --ssl-keyfile key.pem --ssl-certfile cert.pem
```

**JavaScript カメラアクセステスト:**
```javascript
// カメラアクセステスト
async function testCamera() {
    try {
        const devices = await navigator.mediaDevices.enumerateDevices();
        const cameras = devices.filter(device => device.kind === 'videoinput');
        console.log('利用可能なカメラ:', cameras);
        
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { width: 1280, height: 720 }
        });
        console.log('カメラアクセス成功');
        
        // テスト後にストリーム停止
        stream.getTracks().forEach(track => track.stop());
        
    } catch (error) {
        console.error('カメラアクセスエラー:', error);
        
        if (error.name === 'NotAllowedError') {
            alert('カメラアクセスが拒否されました。ブラウザ設定を確認してください。');
        } else if (error.name === 'NotFoundError') {
            alert('カメラが見つかりません。');
        }
    }
}
```

### 2. OpenCV カメラ問題

**問題:** OpenCV でカメラにアクセスできない

**解決方法:**

```python
import cv2

def diagnose_camera():
    """カメラ診断"""
    print("利用可能なカメラデバイス検索中...")
    
    working_cameras = []
    for i in range(10):  # カメラID 0-9 をテスト
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                height, width = frame.shape[:2]
                working_cameras.append({
                    'id': i,
                    'resolution': f"{width}x{height}"
                })
                print(f"カメラ {i}: OK ({width}x{height})")
            cap.release()
        
    if not working_cameras:
        print("利用可能なカメラが見つかりません")
        return None
        
    return working_cameras

# カメラ設定最適化
def setup_camera(camera_id=0):
    cap = cv2.VideoCapture(camera_id)
    
    # よく使われる設定
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # バッファサイズ削減
    
    return cap

# DirectShow 使用 (Windows)
def setup_camera_directshow(camera_id=0):
    cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
    return cap
```

### 3. 画像形式・エンコーディング問題

**問題:** 画像の読み込み・変換エラー

**解決方法:**

```python
from PIL import Image
import io
import base64

def robust_image_processing(image_input):
    """堅牢な画像処理"""
    
    if isinstance(image_input, str):
        # Base64 文字列の場合
        try:
            # データURL形式の処理
            if image_input.startswith('data:image'):
                header, data = image_input.split(',', 1)
                image_input = data
            
            image_data = base64.b64decode(image_input)
            image = Image.open(io.BytesIO(image_data))
            
        except Exception as e:
            print(f"Base64デコードエラー: {e}")
            return None
            
    elif isinstance(image_input, bytes):
        # バイナリデータの場合
        try:
            image = Image.open(io.BytesIO(image_input))
        except Exception as e:
            print(f"バイナリ画像読み込みエラー: {e}")
            return None
    else:
        return None
    
    # 画像形式変換
    if image.mode in ['RGBA', 'LA']:
        # 透明度チャンネル削除
        background = Image.new('RGB', image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[-1] if image.mode == 'RGBA' else None)
        image = background
    elif image.mode != 'RGB':
        image = image.convert('RGB')
    
    return image

# 画像サイズ制限
def limit_image_size(image, max_size_mb=10):
    """画像サイズ制限"""
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG')
    size_mb = len(buffer.getvalue()) / (1024 * 1024)
    
    if size_mb > max_size_mb:
        # 品質を下げてサイズ削減
        quality = int(95 * (max_size_mb / size_mb))
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=max(quality, 50))
    
    return buffer.getvalue()
```

## モデル・AI関連問題

### 1. モデル読み込みエラー

**問題:** YOLO モデルの読み込みに失敗

**診断・解決方法:**

```python
def diagnose_model_loading():
    """モデル読み込み診断"""
    
    # 1. ファイル存在確認
    model_path = 'yolo11x.pt'
    if not os.path.exists(model_path):
        print("❌ モデルファイルが存在しません")
        print("解決方法: from ultralytics import YOLO; YOLO('yolo11x.pt')")
        return False
    
    # 2. ファイルサイズ確認
    file_size = os.path.getsize(model_path) / (1024 * 1024)
    print(f"モデルファイルサイズ: {file_size:.1f}MB")
    
    if file_size < 80:  # YOLO11x は約87MB
        print("⚠️ ファイルサイズが小さすぎます（破損の可能性）")
        os.remove(model_path)
        print("破損ファイルを削除しました。再ダウンロードしてください。")
        return False
    
    # 3. モデル読み込みテスト
    try:
        from ultralytics import YOLO
        model = YOLO(model_path)
        print("✅ モデル読み込み成功")
        
        # 4. 簡単な推論テスト
        import numpy as np
        test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        results = model(test_image)
        print("✅ 推論テスト成功")
        
        return True
        
    except Exception as e:
        print(f"❌ モデル読み込みエラー: {e}")
        return False

# モデル再ダウンロード
def redownload_model():
    """モデル再ダウンロード"""
    model_path = 'yolo11x.pt'
    
    if os.path.exists(model_path):
        os.remove(model_path)
        print("古いモデルファイルを削除しました")
    
    from ultralytics import YOLO
    model = YOLO('yolo11x.pt')  # 自動ダウンロード
    print("新しいモデルをダウンロードしました")
```

### 2. 予測精度の問題

**問題:** 物体検出の精度が低い

**調整方法:**

```python
# 1. 信頼度閾値調整
model.conf = 0.25  # デフォルト値、下げると検出数増加
model.iou = 0.45   # NMS閾値、下げると重複削減

# 2. 画像前処理改善
def enhance_image_quality(image):
    """画像品質向上"""
    import cv2
    
    # コントラスト・明度調整
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    
    enhanced = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
    
    return enhanced

# 3. マルチスケール検出
def multi_scale_detection(image, model):
    """マルチスケール検出"""
    scales = [640, 800, 1024]
    all_results = []
    
    for scale in scales:
        resized = cv2.resize(image, (scale, scale))
        results = model(resized)
        all_results.extend(results)
    
    # NMS で重複削除
    return combine_results(all_results)

# 4. アンサンブル予測
def ensemble_prediction(image, models):
    """複数モデルのアンサンブル"""
    all_detections = []
    
    for model in models:
        results = model(image)
        all_detections.extend(results)
    
    # 投票による信頼度向上
    return vote_detections(all_detections)
```

### 3. GPU メモリ不足

**問題:** CUDA out of memory エラー

**解決方法:**

```python
# 1. バッチサイズ削減
def process_images_in_batches(images, model, batch_size=1):
    """バッチ処理でメモリ使用量削減"""
    results = []
    
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size]
        
        # メモリクリーンアップ
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        batch_results = model(batch)
        results.extend(batch_results)
        
        # 即座にCPUに移動
        del batch_results
        
    return results

# 2. 混合精度推論
def setup_half_precision(model):
    """半精度推論設定"""
    if torch.cuda.is_available():
        model.half()  # FP16モード
    return model

# 3. メモリ監視付き推論
def memory_safe_inference(image, model):
    """メモリ安全な推論"""
    try:
        if torch.cuda.is_available():
            # 利用可能メモリ確認
            memory_free = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
            memory_gb = memory_free / (1024**3)
            
            if memory_gb < 2.0:  # 2GB未満の場合
                torch.cuda.empty_cache()
                gc.collect()
        
        results = model(image)
        return results
        
    except RuntimeError as e:
        if "out of memory" in str(e):
            print("GPU メモリ不足、CPUモードにフォールバック")
            model.cpu()
            results = model(image)
            return results
        else:
            raise e
```

## ブラウザ・フロントエンド問題

### 1. JavaScript エラー

**問題:** ブラウザのコンソールにエラーが表示される

**一般的なエラーと解決方法:**

```javascript
// 1. Fetch API エラーハンドリング
async function robustFetch(url, options = {}) {
    const defaultOptions = {
        method: 'GET',
        headers: {
            'Content-Type': 'application/json',
        },
        timeout: 30000,  // 30秒タイムアウト
    };
    
    const finalOptions = { ...defaultOptions, ...options };
    
    try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), finalOptions.timeout);
        
        const response = await fetch(url, {
            ...finalOptions,
            signal: controller.signal
        });
        
        clearTimeout(timeoutId);
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        return data;
        
    } catch (error) {
        console.error('Fetch エラー:', error);
        
        if (error.name === 'AbortError') {
            throw new Error('リクエストがタイムアウトしました');
        }
        
        throw error;
    }
}

// 2. WebSocket エラーハンドリング
class RobustWebSocket {
    constructor(url) {
        this.url = url;
        this.ws = null;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.reconnectDelay = 1000;
        
        this.connect();
    }
    
    connect() {
        try {
            this.ws = new WebSocket(this.url);
            
            this.ws.onopen = () => {
                console.log('WebSocket 接続成功');
                this.reconnectAttempts = 0;
            };
            
            this.ws.onclose = (event) => {
                console.log('WebSocket 切断:', event);
                this.handleReconnect();
            };
            
            this.ws.onerror = (error) => {
                console.error('WebSocket エラー:', error);
            };
            
            this.ws.onmessage = (event) => {
                this.handleMessage(JSON.parse(event.data));
            };
            
        } catch (error) {
            console.error('WebSocket 接続エラー:', error);
            this.handleReconnect();
        }
    }
    
    handleReconnect() {
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
            this.reconnectAttempts++;
            console.log(`再接続試行 ${this.reconnectAttempts}/${this.maxReconnectAttempts}`);
            
            setTimeout(() => {
                this.connect();
            }, this.reconnectDelay * this.reconnectAttempts);
        } else {
            console.error('WebSocket 再接続失敗');
        }
    }
}

// 3. ファイルアップロードエラー対策
function validateFile(file) {
    const maxSize = 10 * 1024 * 1024; // 10MB
    const allowedTypes = ['image/jpeg', 'image/png', 'image/gif', 'image/webp'];
    
    if (!file) {
        throw new Error('ファイルが選択されていません');
    }
    
    if (file.size > maxSize) {
        throw new Error('ファイルサイズが大きすぎます（最大10MB）');
    }
    
    if (!allowedTypes.includes(file.type)) {
        throw new Error('サポートされていないファイル形式です');
    }
    
    return true;
}

// 4. Canvas操作エラー対策
function safeCanvasOperation(canvas, operation) {
    try {
        const ctx = canvas.getContext('2d');
        if (!ctx) {
            throw new Error('Canvas コンテキストを取得できません');
        }
        
        return operation(ctx);
        
    } catch (error) {
        console.error('Canvas 操作エラー:', error);
        throw error;
    }
}
```

### 2. レスポンシブデザインの問題

**問題:** モバイルデバイスでの表示問題

**CSS解決方法:**

```css
/* モバイル対応改善 */
@media (max-width: 768px) {
    .container {
        padding: 10px;
        margin: 0;
    }
    
    .image-upload-area {
        min-height: 200px; /* モバイルで高さ確保 */
    }
    
    .detection-results {
        font-size: 14px;
        overflow-x: auto; /* 横スクロール対応 */
    }
    
    .canvas-container {
        width: 100%;
        overflow-x: auto;
    }
    
    canvas {
        max-width: 100%;
        height: auto;
    }
}

/* タッチデバイス対応 */
@media (hover: none) and (pointer: coarse) {
    .button {
        min-height: 44px; /* タッチ操作しやすいサイズ */
        font-size: 16px;   /* iOS Safari のズーム防止 */
    }
    
    input[type="file"] {
        font-size: 16px;   /* iOS Safari のズーム防止 */
    }
}

/* 高DPI ディスプレイ対応 */
@media (-webkit-min-device-pixel-ratio: 2), (min-resolution: 192dpi) {
    .icon {
        background-size: contain;
    }
}
```

### 3. パフォーマンス問題

**問題:** フロントエンドの応答性が悪い

**最適化方法:**

```javascript
// 1. デバウンス機能
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// 検索フィールドでの使用例
const debouncedSearch = debounce(performSearch, 300);

// 2. レイジーローディング
function lazyLoadImages() {
    const images = document.querySelectorAll('img[data-src]');
    const imageObserver = new IntersectionObserver((entries, observer) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const img = entry.target;
                img.src = img.dataset.src;
                img.classList.remove('lazy');
                imageObserver.unobserve(img);
            }
        });
    });
    
    images.forEach(img => imageObserver.observe(img));
}

// 3. 効率的なDOM操作
function updateDetectionResults(results) {
    // DocumentFragment 使用で DOM 操作を最小化
    const fragment = document.createDocumentFragment();
    
    results.forEach(result => {
        const div = document.createElement('div');
        div.className = 'detection-item';
        div.innerHTML = `
            <span class="class-name">${result.class}</span>
            <span class="confidence">${(result.confidence * 100).toFixed(1)}%</span>
        `;
        fragment.appendChild(div);
    });
    
    const container = document.getElementById('results-container');
    container.innerHTML = ''; // 一度だけクリア
    container.appendChild(fragment); // 一度だけ追加
}

// 4. Web Workers使用（重い処理の分離）
// worker.js
self.onmessage = function(e) {
    const { imageData, operation } = e.data;
    
    // 重い画像処理をメインスレッドから分離
    const result = processImage(imageData, operation);
    
    self.postMessage(result);
};

// メインスレッド
const worker = new Worker('worker.js');
worker.postMessage({ imageData, operation: 'resize' });
worker.onmessage = (e) => {
    const processedImage = e.data;
    // 結果の処理
};
```

## システム固有の問題

### 1. Windows 固有の問題

**問題:** Windows での特有のエラー

**解決方法:**

```bash
# パス区切り文字問題
# Python 内でのパス処理
import os
from pathlib import Path

# 正しい方法
model_path = Path("yolo11x.pt")
image_dir = Path("uploads") / "images"

# 長いパス名問題（Windows）
# レジストリ設定で長いパス名を有効化
# または短いパス名を使用

# Windows Defender 除外設定
# Python実行ファイルとプロジェクトフォルダを除外対象に追加

# PowerShell実行ポリシー
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Windows特有のパッケージインストール:**
```bash
# Visual Studio Build Tools インストール後
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Microsoft Visual C++ 14.0 エラー対策
pip install wheel setuptools
pip install --only-binary=all opencv-python
```

### 2. macOS 固有の問題

**問題:** macOS での特有のエラー

**解決方法:**

```bash
# Apple Silicon (M1/M2) 対応
# Miniforge または Miniconda を使用
arch -arm64 brew install miniforge
conda create -n objdet python=3.9
conda activate objdet

# MPS (Metal Performance Shaders) 使用
pip install torch torchvision torchaudio

# Xcode Command Line Tools
xcode-select --install

# PermissionError 対策
sudo chown -R $(whoami) /usr/local/lib/python3.x/site-packages/
```

**macOS特有のカメラ問題:**
```python
# macOS カメラ許可確認
def check_macos_camera_permission():
    import platform
    if platform.system() == 'Darwin':
        try:
            cap = cv2.VideoCapture(0)
            ret = cap.isOpened()
            cap.release()
            
            if not ret:
                print("カメラアクセス拒否")
                print("システム環境設定 > セキュリティとプライバシー > カメラ")
                print("Python/ターミナルにカメラアクセスを許可してください")
                
            return ret
        except Exception:
            return False
    return True
```

### 3. Linux 固有の問題

**問題:** Linux での特有のエラー

**解決方法:**

```bash
# システムライブラリ依存関係
# Ubuntu/Debian
sudo apt update
sudo apt install python3-pip python3-venv
sudo apt install libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1

# CentOS/RHEL
sudo yum install python3-pip python3-venv
sudo yum install mesa-libGL glib2 libSM libXext libXrender gomp

# OpenCV依存ライブラリ
sudo apt install libopencv-dev python3-opencv

# GPU支援（NVIDIA）
# CUDA ドライバーインストール
sudo apt install nvidia-driver-xxx cuda

# cuDNN インストール
sudo apt install libcudnn8 libcudnn8-dev
```

**Linux権限問題:**
```bash
# ユーザーをビデオグループに追加（カメラアクセス）
sudo usermod -a -G video $USER

# USB カメラデバイス確認
ls -l /dev/video*
sudo chmod 666 /dev/video0

# ファイル権限修正
chmod +x app.py
chmod 755 static/
chmod 644 static/*
```

## 高度なデバッグ

### 1. ログ設定とデバッグ

**詳細ログ設定:**

```python
import logging
import sys
from datetime import datetime

# カスタムフォーマッター
class CustomFormatter(logging.Formatter):
    def format(self, record):
        # タイムスタンプ
        record.timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        
        # ファイル名と行番号
        record.location = f"{record.filename}:{record.lineno}"
        
        return super().format(record)

# ロガー設定
def setup_logging(log_level=logging.INFO):
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    # コンソールハンドラー
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = CustomFormatter(
        '%(timestamp)s - %(name)s - %(levelname)s - %(location)s - %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    
    # ファイルハンドラー
    file_handler = logging.FileHandler('debug.log', encoding='utf-8')
    file_formatter = CustomFormatter(
        '%(timestamp)s - %(name)s - %(levelname)s - %(location)s - %(funcName)s - %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

# 使用例
logger = setup_logging(logging.DEBUG)

# デコレータを使ったデバッグ
def debug_performance(func):
    """パフォーマンス測定デコレータ"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        
        logger.debug(f"{func.__name__} 開始")
        
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.info(f"{func.__name__} 完了 ({execution_time:.4f}s)")
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"{func.__name__} エラー ({execution_time:.4f}s): {str(e)}")
            raise
            
    return wrapper
```

### 2. プロファイリング

**パフォーマンスプロファイリング:**

```python
import cProfile
import pstats
from io import StringIO

def profile_function(func, *args, **kwargs):
    """関数のプロファイリング"""
    pr = cProfile.Profile()
    pr.enable()
    
    result = func(*args, **kwargs)
    
    pr.disable()
    
    # 結果出力
    s = StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats()
    
    print(s.getvalue())
    return result

# メモリプロファイリング
def memory_profile():
    """メモリ使用量プロファイリング"""
    import tracemalloc
    
    tracemalloc.start()
    
    # ここに測定したいコード
    
    current, peak = tracemalloc.get_traced_memory()
    print(f"現在のメモリ使用量: {current / 1024 / 1024:.1f} MB")
    print(f"ピークメモリ使用量: {peak / 1024 / 1024:.1f} MB")
    
    tracemalloc.stop()

# 使用例
@debug_performance
def detect_objects_profiled(image_path):
    return detection_service.detect_objects(image_path)
```

### 3. ネットワーク診断

**ネットワーク接続診断:**

```python
import socket
import requests
import time

def network_diagnostics():
    """ネットワーク診断"""
    
    print("🌐 ネットワーク診断開始")
    
    # 1. ローカル接続テスト
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        print(f"✅ ローカル接続: {response.status_code}")
    except Exception as e:
        print(f"❌ ローカル接続エラー: {e}")
    
    # 2. インターネット接続テスト
    try:
        response = requests.get("https://httpbin.org/get", timeout=5)
        print(f"✅ インターネット接続: {response.status_code}")
    except Exception as e:
        print(f"❌ インターネット接続エラー: {e}")
    
    # 3. DNS解決テスト
    try:
        ip = socket.gethostbyname("google.com")
        print(f"✅ DNS解決: google.com -> {ip}")
    except Exception as e:
        print(f"❌ DNS解決エラー: {e}")
    
    # 4. ポートスキャン
    def scan_port(host, port):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex((host, port))
            sock.close()
            return result == 0
        except Exception:
            return False
    
    common_ports = [80, 443, 8000, 8080]
    for port in common_ports:
        status = "開放" if scan_port("localhost", port) else "閉鎖"
        print(f"ポート {port}: {status}")

# WebSocket接続テスト
async def test_websocket_connection():
    """WebSocket接続テスト"""
    import websockets
    
    try:
        uri = "ws://localhost:8000/ws/video/test-client"
        async with websockets.connect(uri) as websocket:
            await websocket.send('{"test": "connection"}')
            response = await websocket.recv()
            print(f"✅ WebSocket接続成功: {response}")
            
    except Exception as e:
        print(f"❌ WebSocket接続エラー: {e}")
```

### 4. システム環境診断

**完全システム診断:**

```python
def complete_system_diagnosis():
    """完全システム診断"""
    
    print("🔍 完全システム診断")
    print("=" * 60)
    
    # システム情報
    import platform
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"アーキテクチャ: {platform.machine()}")
    print(f"Python: {platform.python_version()}")
    
    # CPU情報
    import psutil
    print(f"CPU コア数: {psutil.cpu_count()}")
    print(f"CPU 使用率: {psutil.cpu_percent(interval=1)}%")
    
    # メモリ情報
    memory = psutil.virtual_memory()
    print(f"メモリ: {memory.used / 1024**3:.1f}GB / {memory.total / 1024**3:.1f}GB")
    
    # ディスク情報
    disk = psutil.disk_usage('.')
    print(f"ディスク: {disk.used / 1024**3:.1f}GB / {disk.total / 1024**3:.1f}GB")
    
    # GPU情報
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU メモリ: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    else:
        print("GPU: 利用不可")
    
    # ネットワーク診断
    network_diagnostics()
    
    # パッケージ診断  
    check_required_packages()
    
    print("\n" + "=" * 60)
    print("診断完了")

if __name__ == "__main__":
    complete_system_diagnosis()
```

このトラブルシューティングガイドを使用して、物体検出アプリケーションで発生する可能性のあるほぼすべての問題を診断・解決できます。問題が解決しない場合は、診断結果とエラーメッセージを保存して技術サポートに連絡してください。
