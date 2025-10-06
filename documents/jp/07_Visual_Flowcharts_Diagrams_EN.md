# Visual Flowcharts and Diagrams

## Table of Contents
1. [System Architecture Diagrams](#system-architecture-diagrams)
2. [Workflow Flowcharts](#workflow-flowcharts)
3. [Data Flow Diagrams](#data-flow-diagrams)
4. [Component Interaction Diagrams](#component-interaction-diagrams)
5. [Network Communication Diagrams](#network-communication-diagrams)
6. [Error Handling Flow](#error-handling-flow)
7. [Performance Monitoring Diagrams](#performance-monitoring-diagrams)

## System Architecture Diagrams

### Overall System Architecture
```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              OBJECT DETECTION SYSTEM                            │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐  │
│  │   PRESENTATION      │    │    APPLICATION      │    │      BUSINESS       │  │
│  │      LAYER          │    │       LAYER         │    │       LAYER         │  │
│  │                     │    │                     │    │                     │  │
│  │  ┌───────────────┐  │    │  ┌───────────────┐  │    │  ┌───────────────┐  │  │
│  │  │   HTML/CSS    │  │    │  │   FastAPI     │  │    │  │   Detection   │  │  │
│  │  │   Frontend    │  │◄───┤  │   Router      │  │◄───┤  │   Service     │  │  │
│  │  └───────────────┘  │    │  └───────────────┘  │    │  └───────────────┘  │  │
│  │                     │    │                     │    │                     │  │
│  │  ┌───────────────┐  │    │  ┌───────────────┐  │    │  ┌───────────────┐  │  │
│  │  │  JavaScript   │  │    │  │  WebSocket    │  │    │  │   Session     │  │  │
│  │  │   Client      │  │◄───┤  │   Handler     │  │    │  │   Manager     │  │  │
│  │  └───────────────┘  │    │  └───────────────┘  │    │  └───────────────┘  │  │
│  │                     │    │                     │    │                     │  │
│  └─────────────────────┘    └─────────────────────┘    └─────────────────────┘  │
│                                       │                           │            │
│                                       ▼                           ▼            │
│  ┌─────────────────────────────────────────────────────────────────────────────┐  │
│  │                              MODEL LAYER                                   │  │
│  │                                                                             │  │
│  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │  │
│  │  │   YOLO11x   │    │   OpenCV    │    │    PIL      │    │   NumPy     │  │  │
│  │  │   Model     │    │ Processing  │    │ Operations  │    │  Arrays     │  │  │
│  │  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘  │  │
│  │                                                                             │  │
│  └─────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Component Relationships
```
                    ┌─────────────────┐
                    │     Browser     │
                    │   (Frontend)    │
                    └─────────────────┘
                             │
                    HTTP/WebSocket Requests
                             │
                             ▼
                    ┌─────────────────┐
                    │    FastAPI      │
                    │   Application   │
                    └─────────────────┘
                             │
                        ┌────┴────┐
                        │         │
                        ▼         ▼
              ┌─────────────┐  ┌─────────────┐
              │  Session    │  │  WebSocket  │
              │  Manager    │  │  Manager    │
              └─────────────┘  └─────────────┘
                        │         │
                        └────┬────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │   Detection     │
                    │   Service       │
                    └─────────────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │   YOLO Model    │
                    │   Processing    │
                    └─────────────────┘
```

## Workflow Flowcharts

### Image Comparison Workflow
```
                         START
                           │
                           ▼
                  ┌─────────────────┐
                  │  Initialize     │
                  │  Session        │
                  └─────────────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │  Upload/Capture │
                  │  First Image    │
                  └─────────────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │  Validate       │
                  │  Image Format   │
                  └─────────────────┘
                           │
                      ┌────┴────┐
                      │         │
                   Valid?    Invalid
                      │         │
                      ▼         ▼
               ┌─────────────┐  ┌─────────────┐
               │  Convert    │  │  Show       │
               │  to RGB     │  │  Error      │
               └─────────────┘  └─────────────┘
                      │                │
                      ▼                │
               ┌─────────────┐         │
               │  YOLO       │         │
               │  Detection  │         │
               └─────────────┘         │
                      │                │
                      ▼                │
               ┌─────────────┐         │
               │  Analyze    │         │
               │  Results    │         │
               └─────────────┘         │
                      │                │
                      ▼                │
               ┌─────────────┐         │
               │  Store in   │         │
               │  Session    │         │
               └─────────────┘         │
                      │                │
                      ▼                │
               ┌─────────────┐         │
               │  Display    │         │
               │  Results    │         │
               └─────────────┘         │
                      │                │
                      ▼                │
               ┌─────────────┐         │
               │  Ready for  │         │
               │  Image 2?   │         │
               └─────────────┘         │
                      │                │
                  ┌───┴───┐            │
                  │       │            │
                 Yes      No           │
                  │       │            │
                  ▼       ▼            │
          ┌─────────────┐ │            │
          │  Process    │ │            │
          │  Second     │ │            │
          │  Image      │ │◄───────────┘
          └─────────────┘ │
                  │       │
                  ▼       ▼
          ┌─────────────┐ │
          │  Compare    │ │
          │  Results    │ │
          └─────────────┘ │
                  │       │
                  ▼       │
          ┌─────────────┐ │
          │  Generate   │ │
          │  Report     │ │
          └─────────────┘ │
                  │       │
                  ▼       │
          ┌─────────────┐ │
          │  Display    │ │
          │  Comparison │ │
          └─────────────┘ │
                  │       │
                  └───────┘
                          │
                          ▼
                        END
```

### Real-time Video Processing Flow
```
                         START
                           │
                           ▼
                  ┌─────────────────┐
                  │  Initialize     │
                  │  WebSocket      │
                  │  Connection     │
                  └─────────────────┘
                           │
                      ┌────┴────┐
                      │         │
                  Success?   Failure
                      │         │
                      ▼         ▼
               ┌─────────────┐  ┌─────────────┐
               │  Start      │  │  Show       │
               │  Camera     │  │  Error      │
               └─────────────┘  └─────────────┘
                      │                │
                      ▼                │
               ┌─────────────┐         │
               │  Capture    │         │
               │  Frame      │         │
               └─────────────┘         │
                      │                │
                      ▼                │
               ┌─────────────┐         │
               │  Frame      │         │
               │  Queuing    │         │
               └─────────────┘         │
                      │                │
                      ▼                │
               ┌─────────────┐         │
               │  Process    │         │
               │  Frame      │         │
               │  (YOLO)     │         │
               └─────────────┘         │
                      │                │
                      ▼                │
               ┌─────────────┐         │
               │  Send       │         │
               │  Results    │         │
               │  to Client  │         │
               └─────────────┘         │
                      │                │
                      ▼                │
               ┌─────────────┐         │
               │  Update     │         │
               │  UI Overlay │         │
               └─────────────┘         │
                      │                │
                      ▼                │
               ┌─────────────┐         │
               │  Continue   │         │
               │  Processing?│         │
               └─────────────┘         │
                      │                │
                  ┌───┴───┐            │
                  │       │            │
                 Yes      No           │
                  │       │            │
                  │◄──────┘            │
                  │                    │
                  ▼                    │
            ┌──────────┐               │
            │   LOOP   │◄──────────────┘
            └──────────┘
                  │
                  ▼
                END
```

### Inventory Tracking State Machine
```
                    ┌─────────────┐
                    │    IDLE     │◄─────────────────┐
                    └─────────────┘                  │
                           │                         │
                    start_tracking()                 │
                           │                         │
                           ▼                         │
                    ┌─────────────┐                  │
                    │  BASELINE   │                  │
                    │ ESTABLISHED │                  │
                    └─────────────┘                  │
                           │                         │
                    baseline_ready()                 │
                           │                         │
                           ▼                         │
              ┌─────────────────────────┐            │
              │      MONITORING         │            │
              │                         │            │
              │  ┌─────────────────┐   │            │
              │  │  No Changes     │   │            │
              │  │  Detected       │   │            │
              │  └─────────────────┘   │            │
              │            │           │            │
              │            ▼           │            │
              │  ┌─────────────────┐   │            │
              │  │  Continue       │   │            │
              │  │  Monitoring     │   │            │
              │  └─────────────────┘   │            │
              │            │           │            │
              │            └───────────┤            │
              └─────────────────────────┘            │
                           │                         │
                    object_changed()                 │
                           │                         │
                           ▼                         │
                    ┌─────────────┐                  │
                    │   CHANGE    │                  │
                    │  DETECTED   │                  │
                    └─────────────┘                  │
                           │                         │
                    log_change()                     │
                           │                         │
                           ▼                         │
                    ┌─────────────┐                  │
                    │   UPDATE    │                  │
                    │  TIMELINE   │                  │
                    └─────────────┘                  │
                           │                         │
                           │─────────────────────────┘
                           │
                    stop_tracking()
                           │
                           ▼
                    ┌─────────────┐
                    │   REPORT    │
                    │ GENERATION  │
                    └─────────────┘
                           │
                           ▼
                    ┌─────────────┐
                    │   FINISHED  │
                    └─────────────┘
```

## Data Flow Diagrams

### Image Upload and Processing Data Flow
```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Browser   │    │   FastAPI   │    │  Detection  │    │    YOLO     │
│             │    │   Server    │    │   Service   │    │   Model     │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
        │                   │                   │                   │
        │ 1. File Upload    │                   │                   │
        │──────────────────►│                   │                   │
        │   (multipart)     │                   │                   │
        │                   │                   │                   │
        │                   │ 2. Validate File  │                   │
        │                   │──────────────────►│                   │
        │                   │                   │                   │
        │                   │ 3. Convert Image  │                   │
        │                   │◄──────────────────│                   │
        │                   │                   │                   │
        │                   │                   │ 4. Run Inference  │
        │                   │                   │──────────────────►│
        │                   │                   │                   │
        │                   │                   │ 5. Raw Results    │
        │                   │                   │◄──────────────────│
        │                   │                   │                   │
        │                   │ 6. Processed Data │                   │
        │                   │◄──────────────────│                   │
        │                   │                   │                   │
        │ 7. JSON Response  │                   │                   │
        │◄──────────────────│                   │                   │
        │                   │                   │                   │
        │ 8. Update UI      │                   │                   │
        │                   │                   │                   │
```

### WebSocket Real-time Data Flow
```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Client    │    │  WebSocket  │    │   Video     │    │  Detection  │
│ JavaScript  │    │   Handler   │    │  Manager    │    │   Service   │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
        │                   │                   │                   │
        │ 1. Connect        │                   │                   │
        │──────────────────►│                   │                   │
        │                   │                   │                   │
        │                   │ 2. Register       │                   │
        │                   │──────────────────►│                   │
        │                   │                   │                   │
        │ 3. Send Frame     │                   │                   │
        │──────────────────►│                   │                   │
        │                   │                   │                   │
        │                   │ 4. Queue Frame    │                   │
        │                   │──────────────────►│                   │
        │                   │                   │                   │
        │                   │                   │ 5. Process Frame  │
        │                   │                   │──────────────────►│
        │                   │                   │                   │
        │                   │                   │ 6. Detection Data │
        │                   │                   │◄──────────────────│
        │                   │                   │                   │
        │                   │ 7. Send Results   │                   │
        │                   │◄──────────────────│                   │
        │                   │                   │                   │
        │ 8. JSON Results   │                   │                   │
        │◄──────────────────│                   │                   │
        │                   │                   │                   │
        │ 9. Update Overlay │                   │                   │
        │                   │                   │                   │
```

### Session Management Data Flow
```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Client    │    │  Session    │    │   Memory    │
│  Request    │    │  Manager    │    │   Storage   │
└─────────────┘    └─────────────┘    └─────────────┘
        │                   │                   │
        │ 1. Create Session │                   │
        │──────────────────►│                   │
        │                   │                   │
        │                   │ 2. Generate UUID  │
        │                   │──────────────────►│
        │                   │                   │
        │                   │ 3. Store Session  │
        │                   │◄──────────────────│
        │                   │                   │
        │ 4. Session ID     │                   │
        │◄──────────────────│                   │
        │                   │                   │
        │ 5. Store Data     │                   │
        │──────────────────►│                   │
        │                   │                   │
        │                   │ 6. Update Store   │
        │                   │──────────────────►│
        │                   │                   │
        │ 7. Retrieve Data  │                   │
        │──────────────────►│                   │
        │                   │                   │
        │                   │ 8. Get Data       │
        │                   │──────────────────►│
        │                   │                   │
        │                   │ 9. Return Data    │
        │                   │◄──────────────────│
        │                   │                   │
        │ 10. Response Data │                   │
        │◄──────────────────│                   │
        │                   │                   │
```

## Component Interaction Diagrams

### YOLO Detection Service Components
```
┌─────────────────────────────────────────────────────────────────┐
│                  ObjectDetectionService                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐        │
│  │    Model    │    │   Image     │    │  Results    │        │
│  │   Loading   │───►│ Processing  │───►│ Analysis    │        │
│  └─────────────┘    └─────────────┘    └─────────────┘        │
│         │                   │                   │             │
│         ▼                   ▼                   ▼             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐        │
│  │   YOLO      │    │   OpenCV    │    │ Structured  │        │
│  │ yolo11x.pt  │    │ Operations  │    │    Data     │        │
│  └─────────────┘    └─────────────┘    └─────────────┘        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Output Processing                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐          │
│  │ Bounding    │    │ Confidence  │    │ Class Name  │          │
│  │ Box Coords  │    │   Scores    │    │ Resolution  │          │
│  └─────────────┘    └─────────────┘    └─────────────┘          │
│         │                   │                   │               │
│         └─────────────┬─────────────────────────┘               │
│                       ▼                                         │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │              Detection Report                             │  │
│  │  - Total objects: N                                       │  │
│  │  - Unique classes: M                                      │  │
│  │  - Class counts: {class: count, ...}                      │  │
│  │  - Individual detections: [...]                           │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Real-time Processing Pipeline
```
┌─────────────────────────────────────────────────────────────────┐
│                   Video Processing Pipeline                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Camera Stream                                                  │
│        │                                                       │
│        ▼                                                       │
│  ┌─────────────┐                                              │
│  │   Capture   │                                              │
│  │   Frame     │                                              │
│  └─────────────┘                                              │
│        │                                                       │
│        ▼                                                       │
│  ┌─────────────┐    Frame Queue                               │
│  │  Canvas     │◄─────────────────┐                          │
│  │ Rendering   │                  │                          │
│  └─────────────┘                  │                          │
│        │                          │                          │
│        ▼                          │                          │
│  ┌─────────────┐                  │                          │
│  │ Base64      │                  │                          │
│  │ Encoding    │                  │                          │
│  └─────────────┘                  │                          │
│        │                          │                          │
│        ▼                          │                          │
│  ┌─────────────┐                  │                          │
│  │  WebSocket  │                  │                          │
│  │    Send     │                  │                          │
│  └─────────────┘                  │                          │
│        │                          │                          │
│        ▼                          │                          │
│  ┌─────────────┐    Processing    │                          │
│  │   Server    │    Queue         │                          │
│  │ Processing  │◄─────────────────┘                          │
│  └─────────────┘                                             │
│        │                                                       │
│        ▼                                                       │
│  ┌─────────────┐                                              │
│  │    YOLO     │                                              │
│  │ Inference   │                                              │
│  └─────────────┘                                              │
│        │                                                       │
│        ▼                                                       │
│  ┌─────────────┐                                              │
│  │   Results   │                                              │
│  │   Return    │                                              │
│  └─────────────┘                                              │
│        │                                                       │
│        ▼                                                       │
│  ┌─────────────┐                                              │
│  │    UI       │                                              │
│  │   Update    │                                              │
│  └─────────────┘                                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Network Communication Diagrams

### HTTP Request/Response Cycle
```
┌─────────────┐                    ┌─────────────┐
│   Browser   │                    │   Server    │
│             │                    │             │
└─────────────┘                    └─────────────┘
        │                                   │
        │ 1. HTTP POST /detect/1            │
        │──────────────────────────────────►│
        │   Content-Type: multipart/form-data│
        │   - image file                     │
        │   - session_id                     │
        │   - parameters                     │
        │                                   │
        │                                   │ 2. Process Request
        │                                   │    - Validate input
        │                                   │    - Run YOLO detection
        │                                   │    - Generate response
        │                                   │
        │ 3. HTTP 200 OK                   │
        │◄──────────────────────────────────│
        │   Content-Type: application/json  │
        │   {                               │
        │     "success": true,              │
        │     "report": {...},              │
        │     "annotated_image": "base64",  │
        │     "session_id": "uuid"          │
        │   }                               │
        │                                   │
        │ 4. Update UI                      │
        │                                   │
```

### WebSocket Connection Flow
```
┌─────────────┐                    ┌─────────────┐
│   Client    │                    │   Server    │
│             │                    │             │
└─────────────┘                    └─────────────┘
        │                                   │
        │ 1. WebSocket Handshake            │
        │──────────────────────────────────►│
        │   Upgrade: websocket              │
        │   Connection: Upgrade             │
        │                                   │
        │ 2. Connection Established         │
        │◄──────────────────────────────────│
        │   101 Switching Protocols         │
        │                                   │
        │ 3. Frame Data Message             │
        │──────────────────────────────────►│
        │   {                               │
        │     "action": "process_frame",    │
        │     "frame": "base64_data",       │
        │     "conf_threshold": 0.5         │
        │   }                               │
        │                                   │
        │                                   │ 4. Process Frame
        │                                   │    - Decode image
        │                                   │    - Run detection
        │                                   │    - Format results
        │                                   │
        │ 5. Detection Results              │
        │◄──────────────────────────────────│
        │   {                               │
        │     "success": true,              │
        │     "detections": [...],          │
        │     "timestamp": 1698765432       │
        │   }                               │
        │                                   │
        │ 6. Keep-Alive Ping               │
        │──────────────────────────────────►│
        │   {"action": "ping"}              │
        │                                   │
        │ 7. Pong Response                  │
        │◄──────────────────────────────────│
        │   {"type": "pong", "timestamp"}   │
        │                                   │
```

### Multi-Client WebSocket Management
```
                    ┌─────────────────┐
                    │  WebSocket      │
                    │  Manager        │
                    └─────────────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
              ▼              ▼              ▼
    ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
    │  Client A   │  │  Client B   │  │  Client C   │
    │ Connection  │  │ Connection  │  │ Connection  │
    └─────────────┘  └─────────────┘  └─────────────┘
           │                 │                 │
           ▼                 ▼                 ▼
    ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
    │   Frame     │  │   Frame     │  │   Frame     │
    │  Queue A    │  │  Queue B    │  │  Queue C    │
    └─────────────┘  └─────────────┘  └─────────────┘
           │                 │                 │
           └─────────────────┼─────────────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │   Processing    │
                    │     Pool        │
                    └─────────────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │   YOLO Model    │
                    │   (Shared)      │
                    └─────────────────┘
```

## Error Handling Flow

### Error Propagation Chain
```
┌─────────────────────────────────────────────────────────────────┐
│                    Error Handling Flow                          │
└─────────────────────────────────────────────────────────────────┘

    User Action
         │
         ▼
   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
   │  Frontend   │────►│  Validation │────►│  Backend    │
   │   Error     │     │   Error     │     │    Error    │
   └─────────────┘     └─────────────┘     └─────────────┘
         │                     │                     │
         ▼                     ▼                     ▼
   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
   │   Show      │     │   Show      │     │   HTTP      │
   │   Toast     │     │ Field Error │     │   Error     │
   │ Message     │     │   Message   │     │ Response    │
   └─────────────┘     └─────────────┘     └─────────────┘
                                                   │
                                                   ▼
                                          ┌─────────────┐
                                          │  Frontend   │
                                          │   Error     │
                                          │  Handler    │
                                          └─────────────┘
                                                   │
                                                   ▼
                                          ┌─────────────┐
                                          │   Display   │
                                          │User-Friendly│
                                          │   Message   │
                                          └─────────────┘
```

### Error Recovery Strategies
```
        Error Detected
              │
              ▼
      ┌─────────────┐
      │   Log Error │
      └─────────────┘
              │
              ▼
      ┌─────────────┐
      │  Determine  │
      │ Error Type  │
      └─────────────┘
              │
        ┌─────┴─────┐
        │           │
        ▼           ▼
┌─────────────┐ ┌─────────────┐
│ Recoverable │ │   Fatal     │
│   Error     │ │   Error     │
└─────────────┘ └─────────────┘
        │               │
        ▼               ▼
┌─────────────┐ ┌─────────────┐
│   Retry     │ │  Show Error │
│ Operation   │ │  & Reset    │
└─────────────┘ └─────────────┘
        │               │
        ▼               ▼
┌─────────────┐ ┌─────────────┐
│  Success?   │ │  Manual     │
└─────────────┘ │ Intervention│
        │       └─────────────┘
    ┌───┴────┐
    │        │
   Yes       No
    │        │
    ▼        ▼
┌────────┐ ┌─────────────┐
│Continue│ │   Fallback  │
└────────┘ │  Strategy   │
           └─────────────┘
```

## Performance Monitoring Diagrams

### Resource Usage Monitoring
```
┌─────────────────────────────────────────────────────────────────┐
│                    System Resource Monitor                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐        │
│  │     CPU     │    │   Memory    │    │     GPU     │        │
│  │             │    │             │    │             │        │
│  │ ████████░░  │    │ ██████░░░░  │    │ ███████░░░  │        │
│  │    80%      │    │    60%      │    │    70%      │        │
│  └─────────────┘    └─────────────┘    └─────────────┘        │
│        │                   │                   │             │
│        ▼                   ▼                   ▼             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐        │
│  │  Alert at   │    │  Alert at   │    │  Alert at   │        │
│  │    90%      │    │    85%      │    │    90%      │        │
│  └─────────────┘    └─────────────┘    └─────────────┘        │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                    Detection Performance                        │
│                                                                 │
│  Average Response Time: 2.3s                                   │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │ 0s    1s    2s    3s    4s    5s                        │  │
│  │ ├─────┼─────┼─────┼─────┼─────┼─────►                  │  │
│  │       │     █                                           │  │
│  │       │   ████                                          │  │
│  │     ███ ██████                                          │  │
│  │   ███████████                                           │  │
│  └─────────────────────────────────────────────────────────┘  │
│                                                                 │
│  Processing Queue: 3 items                                     │
│  Active Sessions: 8                                            │
│  WebSocket Connections: 5                                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Request Flow Timing
```
Request Timeline (Total: 2.3s)
├─ Network Latency ────────────────────────────────── 50ms
├─ Request Validation ─────────────────────────────── 20ms  
├─ Image Processing ───────────────────────────────── 100ms
├─ YOLO Inference ────────────────────────────────── 1800ms ← Bottleneck
├─ Result Processing ──────────────────────────────── 200ms
├─ Response Generation ────────────────────────────── 80ms
└─ Network Response ───────────────────────────────── 50ms

Optimization Targets:
1. YOLO Inference (78% of total time)
   - Use smaller model (yolo11n vs yolo11x)
   - GPU acceleration
   - Batch processing
   
2. Image Processing (4% of total time)  
   - Pre-resize images
   - Optimize format conversion
   
3. Result Processing (9% of total time)
   - Cache repeated calculations
   - Optimize data structures
```

### Concurrent User Load Analysis
```
                Concurrent Users vs Performance
                
Performance │
   (req/s)  │  ┌─────┐
         50 │  │     │
            │  │     │
         40 │  │     │ ┌─────┐
            │  │     │ │     │
         30 │  │     │ │     │ ┌─────┐
            │  │     │ │     │ │     │
         20 │  │     │ │     │ │     │ ┌─────┐
            │  │     │ │     │ │     │ │     │
         10 │  │     │ │     │ │     │ │     │ ┌─────┐
            │  │     │ │     │ │     │ │     │ │     │
          0 └──┴─────┴─┴─────┴─┴─────┴─┴─────┴─┴─────┴─►
              1-2   3-5   6-10  11-15  16-20    Users

Memory Usage:
- 1-2 users:   600MB  (Baseline + 1 session)
- 3-5 users:   800MB  (3 concurrent sessions)  
- 6-10 users:  1.2GB  (Peak efficiency)
- 11-15 users: 1.8GB  (Performance degradation)
- 16-20 users: 2.5GB  (Memory pressure)

Recommendations:
- Optimal: 6-10 concurrent users
- Scale: Add more instances beyond 10 users  
- Monitor: Memory usage and response times
```

---

These visual diagrams provide comprehensive understanding of the system architecture, data flows, and operational characteristics. They serve as both documentation and troubleshooting aids for developers and system administrators working with the Object Detection Application.
