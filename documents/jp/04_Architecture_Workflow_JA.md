# システムアーキテクチャとワークフロー

## 目次
1. [システムアーキテクチャ概要](#システムアーキテクチャ概要)
2. [コンポーネント設計パターン](#コンポーネント設計パターン)
3. [ワークフロー実装](#ワークフロー実装)
4. [データフローアーキテクチャ](#データフローアーキテクチャ)
5. [リアルタイム処理設計](#リアルタイム処理設計)
6. [状態管理](#状態管理)
7. [スケーラビリティとパフォーマンス](#スケーラビリティとパフォーマンス)
8. [セキュリティアーキテクチャ](#セキュリティアーキテクチャ)

## システムアーキテクチャ概要

### アーキテクチャパターン
このアプリケーションは**レイヤードアーキテクチャ**パターンに従い、明確な関心の分離と保守性を実現しています。

```
┌─────────────────────────────────────────┐
│               プレゼンテーション層         │
│         (HTML/CSS/JavaScript)           │
├─────────────────────────────────────────┤
│                API層                   │
│             (FastAPI)                  │
├─────────────────────────────────────────┤
│               サービス層                 │
│    (ObjectDetectionService など)        │
├─────────────────────────────────────────┤
│               データ層                  │
│         (Session Management)           │
└─────────────────────────────────────────┘
```

### 主要コンポーネント

1. **WebAPIゲートウェイ** - FastAPIアプリケーション
2. **物体検出エンジン** - YOLO11xモデル統合
3. **セッション管理** - UUID ベースのセッション処理
4. **リアルタイム通信** - WebSocketハンドラー
5. **データ処理** - 画像/動画分析パイプライン

## コンポーネント設計パターン

### 1. サービス層パターン

**ObjectDetectionService** クラスは、AI関連のすべての操作をカプセル化します：

```python
class ObjectDetectionService:
    def __init__(self):
        self.model = YOLO('yolo11x.pt')  # モデルの初期化
        self.confidence_threshold = 0.5
        self.size_threshold = 0.01
    
    def detect_objects(self, image_path: str) -> Dict:
        """オブジェクト検出のコアロジック"""
        pass
    
    def analyze_results(self, results) -> Dict:
        """結果分析と後処理"""
        pass
    
    def compare_images(self, image1_results: Dict, image2_results: Dict) -> Dict:
        """画像比較ロジック"""
        pass
```

**設計原則:**
- **単一責任原則**: 各メソッドは特定の機能のみを処理
- **開放閉鎖原則**: 新しい検出アルゴリズムの追加が容易
- **依存性逆転**: インターフェースに依存、具体実装には依存しない

### 2. ファクトリーパターン

セッション作成にはファクトリーパターンを使用：

```python
class SessionManager:
    @staticmethod
    def create_session() -> str:
        """新しいセッションIDを生成"""
        session_id = str(uuid.uuid4())
        sessions[session_id] = {
            'created_at': datetime.now(),
            'last_activity': datetime.now(),
            'data': {}
        }
        return session_id
```

### 3. ビルダーパターン

API レスポンス構築：

```python
class ResponseBuilder:
    def __init__(self):
        self.response = {}
    
    def add_detection_results(self, results):
        self.response['detections'] = results
        return self
    
    def add_metadata(self, metadata):
        self.response['metadata'] = metadata
        return self
    
    def build(self):
        return self.response
```

### 4. ストラテジーパターン

さまざまな検出モードの処理：

```python
class DetectionStrategy:
    def detect(self, image_data):
        raise NotImplementedError

class ImageDetectionStrategy(DetectionStrategy):
    def detect(self, image_data):
        # 画像検出ロジック
        pass

class VideoDetectionStrategy(DetectionStrategy):
    def detect(self, frame_data):
        # 動画フレーム検出ロジック
        pass
```

## ワークフロー実装

### 1. 画像比較ワークフロー

```
┌─────────┐    ┌──────────┐    ┌─────────────┐    ┌──────────┐
│ 画像1   │───▶│ 物体検出  │───▶│ 結果保存     │───▶│ 比較待機  │
│ アップ  │    │          │    │            │    │          │
└─────────┘    └──────────┘    └─────────────┘    └──────────┘
                                                      │
┌─────────┐    ┌──────────┐    ┌─────────────┐       │
│ 比較結果 │◀───│ 画像比較  │◀───│ 結果保存     │◀──────┘
│ 表示    │    │ 処理     │    │            │
└─────────┘    └──────────┘    └─────────────┘
                   ▲
┌─────────┐    ┌──────────┐    ┌─────────────┐
│ 画像2   │───▶│ 物体検出  │───▶│ 結果保存     │
│ アップ  │    │          │    │            │
└─────────┘    └──────────┘    └─────────────┘
```

**実装詳細:**

```python
async def image_comparison_workflow(session_id: str, image_data):
    # 1. セッション検証
    if not validate_session(session_id):
        raise HTTPException(404, "セッションが見つかりません")
    
    # 2. 画像処理
    image_path = await save_uploaded_image(image_data)
    
    # 3. 物体検出
    detection_results = detection_service.detect_objects(image_path)
    
    # 4. セッションに結果保存
    sessions[session_id]['data']['detections'] = detection_results
    
    # 5. レスポンス構築
    return ResponseBuilder().add_detection_results(detection_results).build()
```

### 2. リアルタイム動画ワークフロー

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│ カメラフィード │───▶│ フレーム抽出   │───▶│ 物体検出     │
│             │    │             │    │            │
└─────────────┘    └──────────────┘    └─────────────┘
                                          │
┌─────────────┐    ┌──────────────┐       │
│ 結果送信     │◀───│ WebSocket    │◀──────┘
│ (WebSocket) │    │ ハンドラー    │
└─────────────┘    └──────────────┘
```

**実装詳細:**

```python
@app.websocket("/ws/video/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    # 1. WebSocket接続確立
    await video_manager.connect(websocket, client_id)
    
    try:
        while True:
            # 2. フレームデータ受信
            frame_data = await websocket.receive_bytes()
            
            # 3. 物体検出処理
            results = detection_service.process_video_frame(frame_data)
            
            # 4. 結果送信
            await websocket.send_json(results)
            
    except WebSocketDisconnect:
        # 5. 接続切断処理
        video_manager.disconnect(client_id)
```

### 3. インベントリ追跡ワークフロー

```
┌──────────────┐    ┌─────────────┐    ┌──────────────┐
│ ベースライン  │───▶│ 物体検出     │───▶│ ベースライン  │
│ 画像キャプチャ │    │            │    │ 設定         │
└──────────────┘    └─────────────┘    └──────────────┘
                                          │
┌──────────────┐    ┌─────────────┐       │
│ 変更検出     │◀───│ 差異計算     │◀──────┘
│ レポート     │    │            │
└──────────────┘    └─────────────┘
        ▲                  ▲
        │                  │
┌──────────────┐    ┌─────────────┐
│ リアルタイム  │───▶│ 物体検出     │
│ フィード     │    │            │
└──────────────┘    └─────────────┘
```

## データフローアーキテクチャ

### 1. データ変換パイプライン

```
Raw Image → Base64 Encoding → PIL Image → NumPy Array → YOLO Processing → Results Dict
    ↓
Metadata Extraction → Session Storage → API Response → Frontend Display
```

**詳細実装:**

```python
def process_image_pipeline(image_data):
    # 1. 画像デコード
    image = Image.open(io.BytesIO(base64.b64decode(image_data)))
    
    # 2. NumPy配列変換
    image_array = np.array(image)
    
    # 3. YOLO処理
    results = self.model(image_array)
    
    # 4. 結果解析
    processed_results = self.analyze_results(results)
    
    # 5. メタデータ追加
    processed_results['metadata'] = {
        'image_size': image.size,
        'timestamp': datetime.now().isoformat(),
        'model_version': 'yolo11x'
    }
    
    return processed_results
```

### 2. セッションデータフロー

```
Client Request → Session Validation → Data Retrieval → Processing → Response
      ↓
Session Update → Data Persistence → Cleanup (if needed)
```

## リアルタイム処理設計

### 1. WebSocket アーキテクチャ

```python
class VideoProcessingManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.processing_queue = asyncio.Queue()
    
    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
    
    async def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
    
    async def broadcast_to_client(self, client_id: str, data: dict):
        if client_id in self.active_connections:
            await self.active_connections[client_id].send_json(data)
```

### 2. 非同期処理パターン

```python
async def process_video_stream(client_id: str, frame_data: bytes):
    # 非ブロッキング処理
    loop = asyncio.get_event_loop()
    
    # CPU集約的タスクをスレッドプールで実行
    results = await loop.run_in_executor(
        None, 
        detection_service.detect_objects, 
        frame_data
    )
    
    # 結果を非同期で送信
    await video_manager.broadcast_to_client(client_id, results)
```

### 3. パフォーマンス最適化

**フレームスキッピング:**
```python
class FrameProcessor:
    def __init__(self, skip_frames=2):
        self.skip_frames = skip_frames
        self.frame_count = 0
    
    def should_process_frame(self):
        self.frame_count += 1
        return self.frame_count % (self.skip_frames + 1) == 0
```

**結果キャッシング:**
```python
from functools import lru_cache

@lru_cache(maxsize=100)
def cached_detection(image_hash: str):
    # 同じ画像の再検出を避ける
    return detection_service.detect_objects(image_hash)
```

## 状態管理

### 1. セッション状態

```python
SESSION_STATES = {
    'CREATED': 'セッション作成済み',
    'IMAGE1_UPLOADED': '画像1アップロード完了',
    'IMAGE1_PROCESSED': '画像1処理完了', 
    'IMAGE2_UPLOADED': '画像2アップロード完了',
    'IMAGE2_PROCESSED': '画像2処理完了',
    'COMPARISON_COMPLETE': '比較完了',
    'EXPIRED': 'セッション期限切れ'
}

def update_session_state(session_id: str, new_state: str):
    if session_id in sessions:
        sessions[session_id]['state'] = new_state
        sessions[session_id]['last_activity'] = datetime.now()
```

### 2. アプリケーション状態

```python
class ApplicationState:
    def __init__(self):
        self.total_sessions = 0
        self.active_sessions = 0
        self.total_detections = 0
        self.model_loaded = False
        self.system_health = 'OK'
    
    def update_metrics(self):
        self.active_sessions = len([s for s in sessions.values() 
                                 if s['state'] != 'EXPIRED'])
```

## スケーラビリティとパフォーマンス

### 1. 水平スケーリング戦略

**ロードバランサーサポート:**
```python
# セッションアフィニティなしでのステートレス設計
@app.middleware("http")
async def add_session_header(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Server-ID"] = os.environ.get("SERVER_ID", "server-1")
    return response
```

**データベース統合準備:**
```python
class SessionStore:
    async def save_session(self, session_id: str, data: dict):
        # Redis/MongoDB統合ポイント
        pass
    
    async def load_session(self, session_id: str) -> dict:
        # セッションデータ取得
        pass
```

### 2. パフォーマンス監視

```python
import time
from functools import wraps

def monitor_performance(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        execution_time = time.time() - start_time
        
        # メトリクス記録
        performance_metrics[func.__name__] = execution_time
        return result
    return wrapper

@monitor_performance
async def detect_objects_monitored(image_data):
    return await detection_service.detect_objects(image_data)
```

### 3. メモリ管理

**画像データキャッシュ:**
```python
class ImageCache:
    def __init__(self, max_size_mb=100):
        self.cache = {}
        self.max_size = max_size_mb * 1024 * 1024
        self.current_size = 0
    
    def add_image(self, key: str, image_data: bytes):
        if self.current_size + len(image_data) > self.max_size:
            self._evict_oldest()
        
        self.cache[key] = {
            'data': image_data,
            'timestamp': datetime.now(),
            'size': len(image_data)
        }
        self.current_size += len(image_data)
```

## セキュリティアーキテクチャ

### 1. セッションセキュリティ

```python
import secrets
import hmac

class SecureSessionManager:
    def __init__(self, secret_key: str):
        self.secret_key = secret_key.encode()
    
    def create_secure_session(self) -> str:
        session_id = secrets.token_urlsafe(32)
        timestamp = str(int(time.time()))
        
        # HMAC署名生成
        signature = hmac.new(
            self.secret_key,
            f"{session_id}:{timestamp}".encode(),
            hashlib.sha256
        ).hexdigest()
        
        return f"{session_id}:{timestamp}:{signature}"
    
    def validate_session(self, session_token: str) -> bool:
        try:
            session_id, timestamp, signature = session_token.split(':')
            expected_signature = hmac.new(
                self.secret_key,
                f"{session_id}:{timestamp}".encode(),
                hashlib.sha256
            ).hexdigest()
            
            return hmac.compare_digest(signature, expected_signature)
        except ValueError:
            return False
```

### 2. 入力検証

```python
from pydantic import BaseModel, validator

class ImageUploadRequest(BaseModel):
    image_data: str
    session_id: str
    
    @validator('image_data')
    def validate_base64_image(cls, v):
        try:
            decoded = base64.b64decode(v)
            if len(decoded) > 10 * 1024 * 1024:  # 10MB制限
                raise ValueError("画像サイズが大きすぎます")
            return v
        except Exception:
            raise ValueError("不正なBase64画像データです")
    
    @validator('session_id') 
    def validate_session_id(cls, v):
        if not re.match(r'^[a-f0-9\-]{36}$', v):
            raise ValueError("不正なセッションIDです")
        return v
```

### 3. レート制限

```python
from collections import defaultdict
import time

class RateLimiter:
    def __init__(self, max_requests=100, time_window=3600):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = defaultdict(list)
    
    def is_allowed(self, client_id: str) -> bool:
        now = time.time()
        client_requests = self.requests[client_id]
        
        # 古いリクエストを削除
        client_requests[:] = [req_time for req_time in client_requests 
                             if now - req_time < self.time_window]
        
        # レート制限チェック
        if len(client_requests) >= self.max_requests:
            return False
        
        client_requests.append(now)
        return True

# ミドルウェアとして使用
rate_limiter = RateLimiter()

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    client_ip = request.client.host
    if not rate_limiter.is_allowed(client_ip):
        return JSONResponse(
            status_code=429,
            content={"error": "レート制限に達しました"}
        )
    
    response = await call_next(request)
    return response
```

## 結論

このアーキテクチャ設計は以下の原則に基づいています：

1. **モジュラリティ** - 明確に分離されたコンポーネント
2. **スケーラビリティ** - 水平・垂直スケーリング対応
3. **保守性** - 標準的な設計パターンの使用
4. **パフォーマンス** - 非同期処理とキャッシング戦略
5. **セキュリティ** - 多層防御アプローチ
6. **可観測性** - 包括的な監視とログ記録

この設計により、物体検出WebアプリケーションはエンタープライズレベルのスケーラビリティとReliabilityを実現しています。
