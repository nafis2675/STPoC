# API ドキュメント

## 目次
1. [API概要](#api概要)
2. [認証とセッション](#認証とセッション)
3. [HTTPエンドポイント](#httpエンドポイント)
4. [WebSocket API](#websocket-api)
5. [データモデル](#データモデル)
6. [エラーハンドリング](#エラーハンドリング)
7. [レート制限とパフォーマンス](#レート制限とパフォーマンス)
8. [コード例](#コード例)

## API概要

オブジェクト検出APIはFastAPI上に構築され、リアルタイム処理のためのRESTful HTTPエンドポイントとWebSocket接続の両方を提供します。APIは画像比較ワークフローとライブビデオ検出を包括的なセッション管理と共にサポートします。

### 基本情報
```yaml
ベースURL: http://localhost:3000
プロトコル: HTTP/1.1, WebSocket
Content-Type: application/json (JSONエンドポイント用)
ファイルアップロード: multipart/form-data
リアルタイム: WebSocket (ws://localhost:3000/ws/video/{client_id})
```

### APIアーキテクチャ
```
┌─────────────────────────────────────────────────────────┐
│                    クライアント層                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│  │ファイル     │  │    JSON     │  │  WebSocket  │    │
│  │アップロード │  │   リクエスト │  │  クライアント│    │
│  └─────────────┘  └─────────────┘  └─────────────┘    │
└─────────────────────────────────────────────────────────┘
                           │
                    HTTP/WebSocket
                           │
┌─────────────────────────────────────────────────────────┐
│                 FastAPIサーバー                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│  │  セッション │  │    YOLO     │  │  WebSocket  │    │
│  │エンドポイント│  │   処理      │  │  ハンドラー │    │
│  └─────────────┘  └─────────────┘  └─────────────┘    │
└─────────────────────────────────────────────────────────┘
```

## 認証とセッション

### セッション管理
APIはユーザーデータの分離とマルチユーザーサポートのためUUIDベースのセッション管理を使用します。

#### セッション作成
```http
POST /create-session
```

**レスポンス:**
```json
{
  "success": true,
  "session_id": "123e4567-e89b-12d3-a456-426614174000",
  "message": "New session created successfully"
}
```

#### セッションストレージ
各セッションは以下を保存します:
- `image1`: 第1画像検出結果
- `image2`: 第2画像検出結果
- `created_at`: セッション作成タイムスタンプ
- 24時間後自動クリーンアップ

### セッション検証
すべての処理エンドポイントには有効な`session_id`が必要です。セッションは自動的に検証・クリーンアップされます。

## HTTPエンドポイント

### 1. ページレンダリングエンドポイント

#### ランディングページ
```http
GET /
```
モード選択付きのメインアプリケーションインターフェースを返します。

**レスポンス:** ワークフロー選択付きHTMLページ

#### 画像比較ページ  
```http
GET /image-comparison
```
**レスポンス:** 画像比較ワークフロー用HTMLページ

#### ビデオ検出ページ
```http
GET /video-detection  
```
**レスポンス:** リアルタイムビデオ検出用HTMLページ

#### 追跡ダッシュボード
```http
GET /tracking-dashboard
```
**レスポンス:** オブジェクト追跡ダッシュボード用HTMLページ

#### ネットワークアクセスガイド
```http
GET /network-access
```
**レスポンス:** LAN共有手順とIPアドレス付きHTMLページ

### 2. 検出エンドポイント

#### 画像アップロード検出
```http
POST /detect/{image_number}
Content-Type: multipart/form-data
```

**パラメータ:**
- `image_number`: 1または2（パスパラメータ）
- `image`: 画像ファイル（フォームデータ）
- `conf_threshold`: Float（0.05-0.95、デフォルト: 0.50）
- `iou_threshold`: Float（0.1-0.9、デフォルト: 0.15）
- `max_detection`: Integer（10-1000、デフォルト: 500）
- `session_id`: String（必須）

**リクエスト例:**
```bash
curl -X POST "http://localhost:3000/detect/1" \
  -H "Content-Type: multipart/form-data" \
  -F "image=@image1.jpg" \
  -F "conf_threshold=0.50" \
  -F "iou_threshold=0.15" \
  -F "max_detection=500" \
  -F "session_id=123e4567-e89b-12d3-a456-426614174000"
```

**レスポンス:**
```json
{
  "success": true,
  "report": {
    "image_name": "Image 1: image1.jpg",
    "total_objects": 15,
    "unique_classes": 8,
    "class_counts": {
      "person": 3,
      "chair": 4,
      "table": 1,
      "laptop": 2,
      "book": 3,
      "cup": 1,
      "phone": 1
    },
    "detections": [
      {
        "class_name": "person",
        "confidence": 0.95,
        "bbox": [100.5, 200.3, 300.8, 450.2],
        "area": 50062.5
      }
    ]
  },
  "original_image": "data:image/jpeg;base64,/9j/4AAQ...",
  "annotated_image": "data:image/jpeg;base64,/9j/4AAQ...",
  "parameters": {
    "conf_threshold": 0.50,
    "iou_threshold": 0.15,
    "max_detection": 500
  },
  "ready_for_comparison": false,
  "session_id": "123e4567-e89b-12d3-a456-426614174000"
}
```

#### Base64画像検出
```http
POST /detect-base64/{image_number}
Content-Type: application/json
```

**リクエストボディ:**
```json
{
  "image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD...",
  "conf_threshold": 0.50,
  "iou_threshold": 0.15,
  "max_detection": 500,
  "session_id": "123e4567-e89b-12d3-a456-426614174000"
}
```

**レスポンス:** アップロード検出エンドポイントと同じ

### 3. 分析エンドポイント

#### 画像比較
```http
POST /compare
Content-Type: application/json
```

**リクエストボディ:**
```json
{
  "session_id": "123e4567-e89b-12d3-a456-426614174000"
}
```

**レスポンス:**
```json
{
  "success": true,
  "comparison": {
    "total_objects": {
      "image1": 15,
      "image2": 18, 
      "change": 3
    },
    "class_changes": {
      "person": {
        "image1": 3,
        "image2": 4,
        "change": 1
      },
      "chair": {
        "image1": 4,
        "image2": 3,
        "change": -1
      }
    },
    "new_classes": ["bottle", "mouse"],
    "removed_classes": ["book"]
  },
  "image1": {...}, 
  "image2": {...},
  "session_id": "123e4567-e89b-12d3-a456-426614174000"
}
```

#### パラメータ調整
```http
POST /adjust-parameters
Content-Type: application/json
```

新しい検出パラメータで両画像を再分析します。

**リクエストボディ:**
```json
{
  "conf_threshold": 0.30,
  "iou_threshold": 0.25,
  "max_detection": 1000,
  "session_id": "123e4567-e89b-12d3-a456-426614174000"
}
```

**レスポンス:**
```json
{
  "success": true,
  "adjusted_results": {
    "image1": {...},
    "image2": {...}
  },
  "comparison": {...},
  "parameters_used": {
    "conf_threshold": 0.30,
    "iou_threshold": 0.25,
    "max_detection": 1000
  },
  "session_id": "123e4567-e89b-12d3-a456-426614174000"
}
```

#### 結果保存
```http
POST /save-results
Content-Type: application/json
```

**リクエストボディ:**
```json
{
  "session_id": "123e4567-e89b-12d3-a456-426614174000"
}
```

**レスポンス:**
```json
{
  "success": true,
  "report_text": "オブジェクト検出比較レポート\n=====================================\n\n画像1レポート:\n総オブジェクト数: 15\n...",
  "full_results": {...},
  "comparison": {...},
  "download_ready": true,
  "session_id": "123e4567-e89b-12d3-a456-426614174000"
}
```

### 4. ユーティリティエンドポイント

#### ワークフローリセット
```http
POST /reset-workflow
Content-Type: application/json
```

**リクエストボディ:**
```json
{
  "session_id": "123e4567-e89b-12d3-a456-426614174000"
}
```

**レスポンス:**
```json
{
  "success": true,
  "message": "Workflow reset successfully",
  "session_id": "123e4567-e89b-12d3-a456-426614174000"
}
```

#### ステータス取得
```http
GET /status/{session_id}
```

**レスポンス:**
```json
{
  "image1_processed": true,
  "image2_processed": false,
  "ready_for_comparison": false,
  "next_step": "image2",
  "session_id": "123e4567-e89b-12d3-a456-426614174000"
}
```

#### ヘルスチェック
```http
GET /health
```

**レスポンス:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

#### セッションクリーンアップ
```http
POST /cleanup-sessions
```

**レスポンス:**
```json
{
  "success": true,
  "message": "Cleaned up 5 old sessions",
  "removed_sessions": 5
}
```

## WebSocket API

### 接続
```
ws://localhost:3000/ws/video/{client_id}
```

WebSocket APIはライブオブジェクト検出のためのリアルタイムビデオ処理を可能にします。

#### 接続パラメータ
- `client_id`: クライアントセッションの一意識別子

#### メッセージ形式
すべてのWebSocketメッセージはJSON形式を使用:

```json
{
  "action": "message_type",
  "data": {...}
}
```

### クライアントメッセージ

#### フレーム処理
```json
{
  "action": "process_frame",
  "frame": "data:image/jpeg;base64,/9j/4AAQ...",
  "conf_threshold": 0.50,
  "iou_threshold": 0.15,
  "max_detection": 100
}
```

#### キープアライブPing
```json
{
  "action": "ping"
}
```

### サーバーメッセージ

#### 検出結果
```json
{
  "success": true,
  "detections": [
    {
      "class_name": "person",
      "confidence": 0.92,
      "bbox": [120.5, 180.3, 280.8, 420.2],
      "area": 38440.0
    }
  ],
  "class_counts": {
    "person": 2,
    "chair": 1,
    "table": 1
  },
  "total_objects": 4,
  "timestamp": 1698765432.123
}
```

#### Pongレスポンス
```json
{
  "type": "pong",
  "timestamp": 1698765432.123
}
```

#### エラーレスポンス
```json
{
  "success": false,
  "error": "Detection processing failed",
  "timestamp": 1698765432.123
}
```

### WebSocket接続管理

#### 接続ライフサイクル
```python
# 接続確立
async def connect(websocket: WebSocket, client_id: str):
    await websocket.accept()
    # 接続を保存し処理状態を初期化

# メッセージハンドリング
async def handle_message(message: dict):
    if message["action"] == "process_frame":
        # フレーム処理し結果送信
    elif message["action"] == "ping":
        # pongレスポンス送信

# 切断クリーンアップ
def disconnect(client_id: str):
    # 接続削除しリソースクリーンアップ
```

## データモデル

### 検出オブジェクト
```python
{
  "class_name": str,        # オブジェクトクラス（例: "person", "car"）
  "confidence": float,      # 検出信頼度（0.0-1.0）
  "bbox": [float, float, float, float],  # [x1, y1, x2, y2]座標
  "area": float            # バウンディングボックス面積（ピクセル）
}
```

### 検出レポート
```python
{
  "image_name": str,        # 画像の説明名
  "total_objects": int,     # 検出されたオブジェクトの総数
  "unique_classes": int,    # 異なるオブジェクトクラス数
  "class_counts": Dict[str, int],  # オブジェクトクラス毎の数
  "detections": List[Detection]    # 個別検出のリスト
}
```

### 比較結果
```python
{
  "total_objects": {
    "image1": int,
    "image2": int,
    "change": int
  },
  "class_changes": Dict[str, {
    "image1": int,
    "image2": int,
    "change": int
  }],
  "new_classes": List[str],      # 画像2のみで見つかったクラス
  "removed_classes": List[str]   # 画像1のみで見つかったクラス
}
```

### セッションデータ構造
```python
{
  "session_id": str,
  "created_at": float,      # Unixタイムスタンプ
  "image1": {
    "report": DetectionReport,
    "original_image": str,   # Base64エンコード画像
    "annotated_image": str,  # Base64エンコード注釈付き画像
    "parameters": Dict,      # 使用された検出パラメータ
    "detections": List[Detection],
    "class_counts": Dict[str, int]
  },
  "image2": {...}           # image1と同じ構造
}
```

## エラーハンドリング

### HTTPエラーレスポンス

#### 400 Bad Request
```json
{
  "detail": "Invalid parameter: conf_threshold must be between 0.05 and 0.95"
}
```

#### 404 Not Found
```json
{
  "detail": "Session not found. Please create a new session."
}
```

#### 500 Internal Server Error
```json
{
  "detail": "Detection failed: YOLO model not loaded"
}
```

### 一般的エラーシナリオ

#### 無効セッションID
```http
Status: 404 Not Found
{
  "detail": "Session not found. Please create a new session."
}
```

#### 必須パラメータ欠落
```http
Status: 400 Bad Request
{
  "detail": "Session ID required"
}
```

#### 無効ファイルタイプ
```http
Status: 400 Bad Request  
{
  "detail": "Please select a valid image file"
}
```

#### モデル処理エラー
```http
Status: 500 Internal Server Error
{
  "detail": "Detection failed: Model inference error"
}
```

### WebSocketエラーハンドリング

#### 接続エラー
- **接続拒否**: サーバーが実行中でポートがアクセス可能か確認
- **WebSocketアップグレード失敗**: 適切なWebSocketヘッダーを確保
- **タイムアウト**: 指数バックオフによる再接続ロジック実装

#### 処理エラー
```json
{
  "success": false,
  "error": "Frame processing failed: Invalid image format",
  "timestamp": 1698765432.123
}
```

## レート制限とパフォーマンス

### リクエスト制限
```yaml
セッション作成: IP毎分間10回
ファイルアップロード: 最大ファイルサイズ100MB
WebSocket接続: IP毎同時5接続
セッションストレージ: セッション毎最大50MB
処理キュー: 同時検出10回
```

### パフォーマンス考慮事項

#### 最適化パラメータ
```python
# より高速処理のため、これらのパラメータを調整:
PERFORMANCE_CONFIG = {
  "conf_threshold": 0.5,     # 高い = 検出少ない、高速
  "max_detection": 100,      # 低い = 処理時間短縮
  "image_resize": 640,       # 小さい = 推論高速
  "batch_size": 1,           # リアルタイムでは1を維持
  "half_precision": True     # GPUサポート時FP16使用
}
```

#### メモリ管理
- セッション24時間後自動クリーンアップ
- 画像データはメモリのみ保存（ディスク書き込みなし）
- 定期クリーンアップエンドポイント利用可能
- リアルタイム処理はフレームバッファリング使用

## コード例

### Pythonクライアント例

#### 画像検出
```python
import requests
import base64

# セッション作成
session_resp = requests.post("http://localhost:3000/create-session")
session_id = session_resp.json()["session_id"]

# 画像アップロードと検出
with open("image1.jpg", "rb") as f:
    files = {"image": f}
    data = {
        "conf_threshold": 0.5,
        "iou_threshold": 0.15,
        "max_detection": 500,
        "session_id": session_id
    }
    
    response = requests.post(
        "http://localhost:3000/detect/1",
        files=files,
        data=data
    )
    
    result = response.json()
    print(f"{result['report']['total_objects']}個のオブジェクトを検出")
```

#### Base64画像検出
```python
import requests
import base64

# 画像をbase64に変換
with open("image.jpg", "rb") as f:
    image_data = base64.b64encode(f.read()).decode()
    image_b64 = f"data:image/jpeg;base64,{image_data}"

# セッション作成
session_resp = requests.post("http://localhost:3000/create-session")
session_id = session_resp.json()["session_id"]

# オブジェクト検出
payload = {
    "image": image_b64,
    "conf_threshold": 0.5,
    "iou_threshold": 0.15, 
    "max_detection": 500,
    "session_id": session_id
}

response = requests.post(
    "http://localhost:3000/detect-base64/1",
    json=payload
)

result = response.json()
for detection in result["detections"]:
    print(f"{detection['class_name']}を信頼度{detection['confidence']:.2f}で発見")
```

### JavaScript WebSocket例

#### リアルタイムビデオ処理
```javascript
// WebSocketに接続
const clientId = 'client_' + Math.random().toString(36).substr(2, 9);
const ws = new WebSocket(`ws://localhost:3000/ws/video/${clientId}`);

// 接続ハンドリング
ws.onopen = function() {
    console.log('検出サービスに接続');
};

// メッセージハンドリング
ws.onmessage = function(event) {
    const result = JSON.parse(event.data);
    
    if (result.success) {
        console.log(`${result.total_objects}個のオブジェクトを検出`);
        // 検出結果でUI更新
        updateDetectionDisplay(result.detections);
    } else {
        console.error('検出エラー:', result.error);
    }
};

// 処理用フレーム送信
function processFrame(canvas) {
    const frameData = canvas.toDataURL('image/jpeg', 0.8);
    
    const message = {
        action: 'process_frame',
        frame: frameData,
        conf_threshold: 0.5,
        iou_threshold: 0.15,
        max_detection: 100
    };
    
    if (ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify(message));
    }
}

// 接続維持
setInterval(() => {
    if (ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ action: 'ping' }));
    }
}, 30000); // 30秒毎Ping
```

### curl例

#### ヘルスチェック
```bash
curl -X GET "http://localhost:3000/health"
```

#### セッション作成
```bash
curl -X POST "http://localhost:3000/create-session"
```

#### ファイルアップロード画像検出
```bash
curl -X POST "http://localhost:3000/detect/1" \
  -H "Content-Type: multipart/form-data" \
  -F "image=@test_image.jpg" \
  -F "conf_threshold=0.5" \
  -F "iou_threshold=0.15" \
  -F "max_detection=500" \
  -F "session_id=your-session-id-here"
```

#### 画像比較
```bash
curl -X POST "http://localhost:3000/compare" \
  -H "Content-Type: application/json" \
  -d '{"session_id": "your-session-id-here"}'
```

#### セッションリセット
```bash
curl -X POST "http://localhost:3000/reset-workflow" \
  -H "Content-Type: application/json" \
  -d '{"session_id": "your-session-id-here"}'
```

---

このAPIドキュメントは全エンドポイントと機能の包括的カバレッジを提供します。実装例については、アーキテクチャドキュメントを参照してください。API問題のトラブルシューティングについては、トラブルシューティングガイドを参照してください。
