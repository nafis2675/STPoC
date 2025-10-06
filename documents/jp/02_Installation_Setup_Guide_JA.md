# インストールとセットアップガイド

## 目次
1. [システム要件](#システム要件)
2. [クイックスタートガイド](#クイックスタートガイド)
3. [詳細インストール手順](#詳細インストール手順)
4. [設定オプション](#設定オプション)
5. [ネットワーク設定](#ネットワーク設定)
6. [インストール問題のトラブルシューティング](#インストール問題のトラブルシューティング)
7. [パフォーマンス最適化](#パフォーマンス最適化)

## システム要件

### 最小要件
```yaml
ハードウェア:
  - CPU: 4コア以上（Intel i5/AMD Ryzen 5以上）
  - RAM: 4GB最小、8GB推奨
  - ストレージ: モデルファイル用に2GB空き容量
  - GPU: オプション（高速化にはNVIDIA CUDA互換）
  - ウェブカム: オプション（リアルタイム検出用）

オペレーティングシステム:
  - Windows: Windows 10+（64ビット）
  - macOS: macOS 11+（Big Sur以降）
  - Linux: Ubuntu 20.04+または同等

ネットワーク:
  - インターネット: 初期YOLOモデルダウンロードに必要
  - LAN: オプション（マルチデバイスアクセス用）
```

### 推奨要件
```yaml
ハードウェア:
  - CPU: 8コア以上（Intel i7/AMD Ryzen 7）
  - RAM: 最適パフォーマンスに16GB
  - GPU: NVIDIA RTXシリーズまたはGTX 1060+ 6GB VRAM
  - SSD: より高速なモデル読み込みと処理用

ソフトウェア:
  - Python: 3.9+（3.10推奨）
  - ブラウザ: Chrome 90+, Firefox 88+, Safari 14+, Edge 90+
  - Git: バージョン管理用（オプション）
```

## クイックスタートガイド

### オプション1: 自動セットアップ（推奨）

#### Windows
```batch
# 1. プロジェクトをダウンロードして展開
# 2. run.batをダブルクリック
run.bat
```

#### Linux/macOS
```bash
# 1. プロジェクトをダウンロードして展開
# 2. スクリプトを実行可能にして実行
chmod +x run.sh
./run.sh
```

### オプション2: 手動インストール

```bash
# 1. Python依存関係をインストール
pip install -r requirements.txt

# 2. アプリケーション開始
python app.py

# 3. ブラウザでhttp://localhost:3000を開く
```

### 初回アクセス
1. ウェブブラウザを開く
2. `http://localhost:3000`に移動
3. 検出モード選択（画像比較またはリアルタイムビデオ）
4. 画像アップロードまたはカメラ開始で機能テスト

## 詳細インストール手順

### ステップ1: 環境準備

#### Python環境セットアップ
```bash
# Pythonバージョン確認（3.8+必要）
python --version

# 仮想環境作成（推奨）
python -m venv od_env

# 仮想環境有効化
# Windows:
od_env\Scripts\activate
# Linux/macOS:
source od_env/bin/activate

# pip更新
python -m pip install --upgrade pip
```

#### 代替: Conda使用
```bash
# conda環境作成
conda create -n od_env python=3.10
conda activate od_env

# pipパッケージインストール
pip install -r requirements.txt
```

### ステップ2: 依存関係インストール

#### コア依存関係分析
```yaml
requirements.txt内訳:
  fastapi>=0.100.0         # Webフレームワーク
  uvicorn[standard]>=0.20.0 # ASGIサーバー
  python-multipart>=0.0.5  # ファイルアップロードサポート
  ultralytics>=8.0.0       # YOLO実装
  opencv-python>=4.8.0     # コンピュータビジョン
  pillow>=10.0.0          # 画像処理
  numpy>=1.21.0           # 数値計算
  torch>=2.0.0            # 深層学習フレームワーク
  torchvision>=0.15.0     # ビジョンモデル
  torchaudio>=2.0.0       # 音声処理
  jinja2>=3.0.0           # HTMLテンプレート
```

#### バージョン確認付きインストール
```bash
# 全依存関係インストール
pip install -r requirements.txt

# 重要パッケージ確認
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
python -c "import ultralytics; print('YOLO利用可能')"
```

#### GPUサポート（オプションだが推奨）
```bash
# CUDA利用可能性確認
python -c "import torch; print(torch.cuda.is_available())"

# CUDAが利用不可の場合、CUDAありPyTorchをインストール
# 訪問: https://pytorch.org/get-started/locally/
# CUDA 11.8の例:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### ステップ3: YOLOモデルセットアップ

#### 自動モデルダウンロード
```bash
# 初回実行時にyolo11x.ptを自動ダウンロード
# インターネット速度により5-10分かかる可能性
python -c "from ultralytics import YOLO; YOLO('yolo11x.pt')"
```

#### 手動モデルダウンロード（必要時）
```bash
# Ultralyticsから直接ダウンロード
wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolo11x.pt
# またはWindowsでcurl使用:
curl -L -o yolo11x.pt https://github.com/ultralytics/assets/releases/download/v8.2.0/yolo11x.pt
```

#### モデルインストール確認
```python
# テストスクリプト - test_model.pyとして保存
from ultralytics import YOLO
import cv2
import numpy as np

try:
    # モデル読み込み
    model = YOLO('yolo11x.pt')
    print("✅ YOLOモデル正常読み込み")
    
    # ダミー画像でテスト
    test_img = np.zeros((480, 640, 3), dtype=np.uint8)
    results = model(test_img)
    print("✅ モデル推論テスト合格")
    
    # 利用可能クラス表示
    print(f"📊 モデルは{len(model.names)}オブジェクトクラスをサポート")
    
except Exception as e:
    print(f"❌ モデルテスト失敗: {e}")
```

### ステップ4: アプリケーション設定

#### 基本設定確認
```python
# config_check.py - セットアップ確認用実行
import os
import sys
from pathlib import Path

def check_configuration():
    print("🔍 設定確認中...")
    
    # Pythonバージョン確認
    python_version = sys.version_info
    print(f"Pythonバージョン: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version < (3, 8):
        print("❌ Python 3.8+が必要")
        return False
    
    # 必要ファイル確認
    required_files = [
        'app.py',
        'requirements.txt',
        'yolo11x.pt',
        'static/css/style.css',
        'static/js/app.js',
        'templates/index.html'
    ]
    
    for file in required_files:
        if Path(file).exists():
            print(f"✅ {file}")
        else:
            print(f"❌ 欠落: {file}")
            return False
    
    # インポート確認
    try:
        import fastapi, uvicorn, ultralytics, cv2, PIL, numpy, torch
        print("✅ 全必要パッケージ正常インポート")
        return True
    except ImportError as e:
        print(f"❌ インポートエラー: {e}")
        return False

if __name__ == "__main__":
    if check_configuration():
        print("🎉 設定確認合格！開始準備完了。")
    else:
        print("🔧 開始前に上記問題を修正してください。")
```

### ステップ5: 初回起動

#### アプリケーション開始
```bash
# 方法1: 直接Python実行
python app.py

# 方法2: Uvicorn直接使用
uvicorn app:app --host 0.0.0.0 --port 3000

# 方法3: 自動リロード付き開発モード
uvicorn app:app --host 0.0.0.0 --port 3000 --reload
```

#### 確認手順
```bash
# サーバー動作確認
curl http://localhost:3000/health

# 期待レスポンス:
# {"status":"healthy","model_loaded":true}
```

## 設定オプション

### ポート設定
```python
# app.py内、最終行を変更:
if __name__ == "__main__":
    import uvicorn
    # 3000が使用中の場合、ここでポート変更
    uvicorn.run(app, host="0.0.0.0", port=8000)  # 例: ポート8000
```

### 検出パラメータ
```python
# ObjectDetectionService内のデフォルトパラメータ
DEFAULT_CONF_THRESHOLD = 0.50    # 信頼度閾値
DEFAULT_IOU_THRESHOLD = 0.15     # NMS用IoU閾値
DEFAULT_MAX_DETECTION = 500      # 画像当たり最大検出数
DEFAULT_IMAGE_SIZE = 640         # YOLO入力画像サイズ
```

### セッション設定
```python
# セッションクリーンアップ設定
MAX_SESSION_AGE_HOURS = 24       # 24時間後自動クリーンアップ
CLEANUP_INTERVAL_SECONDS = 3600  # 1時間毎チェック
```

### パフォーマンス設定
```python
# リアルタイム処理設定
VIDEO_PROCESSING_FPS = 5         # ビデオ処理FPS
VIDEO_DISPLAY_FPS = 30           # 表示リフレッシュレート
FRAME_PROCESSING_QUALITY = 0.6   # フレーム処理用JPEG品質
REALTIME_IMAGE_SIZE = 416        # リアルタイム処理用小サイズ
```

## ネットワーク設定

### ローカルアクセス設定
```python
# デフォルトバインド（ローカルホストのみ）
uvicorn.run(app, host="127.0.0.1", port=3000)

# LANアクセス（全インターフェース）
uvicorn.run(app, host="0.0.0.0", port=3000)
```

### ファイアウォール設定

#### Windowsファイアウォール
```batch
# WindowsファイアウォールでPython許可
netsh advfirewall firewall add rule name="Object Detection App" dir=in action=allow program="C:\Python39\python.exe" enable=yes

# または特定ポート許可
netsh advfirewall firewall add rule name="OD Port 3000" dir=in action=allow protocol=TCP localport=3000
```

#### Linux iptables
```bash
# ポート3000許可
sudo iptables -A INPUT -p tcp --dport 3000 -j ACCEPT

# Ubuntu/Debian ufw使用
sudo ufw allow 3000
```

#### macOS
```bash
# macOSは通常ローカルネットワークアクセスをデフォルト許可
# 必要時: システム環境設定 > セキュリティとプライバシー > ファイアウォール確認
```

### ネットワークアクセス確認
```bash
# ローカルIPアドレス確認
# Windows:
ipconfig | findstr IPv4

# Linux/macOS:
hostname -I
# または
ifconfig | grep inet

# 他デバイスからテスト
curl http://[あなたのIP]:3000/health
```

## インストール問題のトラブルシューティング

### 一般的問題と解決策

#### 問題1: Pythonバージョン互換性
```bash
# 症状: パッケージインストール失敗
# 解決策: Pythonバージョン確認
python --version

# Python < 3.8の場合、Python更新:
# Windows: python.orgからダウンロード
# Ubuntu: sudo apt update && sudo apt install python3.10
# macOS: brew install python@3.10
```

#### 問題2: PyTorchインストール問題
```bash
# 症状: CUDA/GPU未検出
# 解決策: 適切CUDAバージョンでPyTorch再インストール
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CPUのみ（GPU無し）:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

#### 問題3: OpenCVインポートエラー
```bash
# 症状: "ImportError: libGL.so.1"
# Linux解決策:
sudo apt-get update
sudo apt-get install libgl1-mesa-glx libglib2.0-0

# 代替OpenCVインストール:
pip uninstall opencv-python
pip install opencv-python-headless
```

#### 問題4: YOLOモデルダウンロード失敗
```bash
# 症状: モデルダウンロードタイムアウト/失敗
# 解決策: 手動ダウンロードと配置
wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolo11x.pt
# プロジェクトルートディレクトリに配置
```

#### 問題5: ポート使用中
```bash
# 症状: "Port 3000 is already in use"
# ポート使用プロセス確認:
# Windows:
netstat -ano | findstr :3000
taskkill /PID <PID> /F

# Linux/macOS:
lsof -i :3000
kill -9 <PID>

# またはapp.pyでポート変更
```

#### 問題6: カメラアクセス拒否
```bash
# 症状: ブラウザでカメラアクセス不可
# 解決策:
1. HTTP代わりにHTTPS使用（モバイル必須）
2. ブラウザでカメラ許可付与
3. 他アプリケーションがカメラ使用中でないか確認
4. 異なるブラウザ試行
```

### デバッグモード設定
```python
# デバッグログ有効化
import logging
logging.basicConfig(level=logging.DEBUG)

# app.pyにデバッグプリント追加
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=3000,
        log_level="debug",
        access_log=True
    )
```

### メモリとパフォーマンス問題
```bash
# メモリ使用量確認
# Windows:
tasklist /fi "imagename eq python.exe"

# Linux/macOS:
ps aux | grep python
top -p $(pgrep python)

# メモリ使用量削減:
# - max_detectionパラメータ削減
# - 小さいYOLOモデル使用（yolo11x.ptの代わりにyolo11n.pt）
# - 処理前画像解像度削減
```

## パフォーマンス最適化

### ハードウェア最適化

#### GPU高速化セットアップ
```python
# GPU利用可能性確認とセットアップ
import torch

def setup_gpu():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✅ GPU利用可能: {torch.cuda.get_device_name(0)}")
        print(f"📊 GPUメモリ: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        # GPUメモリ管理設定
        torch.backends.cudnn.benchmark = True
        return device
    else:
        print("⚠️  GPU利用不可、CPU使用")
        return torch.device("cpu")

# ObjectDetectionServiceで適用
device = setup_gpu()
model = YOLO('yolo11x.pt')
model.to(device)
```

#### メモリ最適化
```python
# メモリ効率設定
MEMORY_EFFICIENT_CONFIG = {
    'max_sessions': 10,           # 同時セッション制限
    'image_max_size': 1920,       # 大画像リサイズ
    'video_buffer_size': 5,       # ビデオフレームバッファ制限
    'cleanup_interval': 1800      # より頻繁クリーンアップ（30分）
}
```

### ソフトウェア最適化

#### モデル最適化
```python
# ハードウェアベースの適切モデルサイズ使用
MODEL_CONFIGS = {
    'high_performance': 'yolo11x.pt',    # 最高精度、低速
    'balanced': 'yolo11l.pt',            # バランス良好
    'fast': 'yolo11m.pt',                # 高速処理
    'mobile': 'yolo11n.pt'               # 最高速、モバイル向け
}

# 要件ベースモデル切替:
# ObjectDetectionService.__init__()内:
self.model_name = MODEL_CONFIGS['balanced']  # 必要に応じ変更
```

#### 処理最適化
```python
# 速度用検出パラメータ最適化
FAST_DETECTION_CONFIG = {
    'conf_threshold': 0.3,        # 高閾値 = 少ない検出
    'iou_threshold': 0.5,         # 高閾値 = 少ない重複処理
    'max_detection': 100,         # 最大検出数制限
    'imgsz': 416,                 # 小入力サイズ = 高速処理
    'half': True                  # FP16精度使用（GPU対応時）
}
```

### ネットワーク最適化
```python
# より良いネットワークパフォーマンス設定
NETWORK_CONFIG = {
    'websocket_timeout': 30,
    'max_frame_size': 1024*1024,  # 最大フレームサイズ1MB
    'compression_quality': 0.7,   # フレーム用JPEG圧縮
    'batch_processing': False     # リアルタイム用バッチング無効
}
```

---

このインストールガイドはオブジェクト検出ウェブアプリケーションの包括的セットアップ手順を提供します。最もスムーズなインストール体験のため、順序通り手順に従ってください。アーキテクチャ詳細については、アーキテクチャドキュメントを参照してください。
