# 🧠 Algorithm Documentation

## Table of Contents
1. [YOLOv5 Architecture](#yolov5-architecture)
2. [Data Flow Pipeline](#data-flow-pipeline)
3. [Training Process](#training-process)
4. [Inference Pipeline](#inference-pipeline)

---

## YOLOv5 Architecture

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#ffd8d8', 'primaryBorderColor': '#ff6b6b', 'lineColor': '#ff8787', 'secondaryColor': '#d0f0fd', 'tertiaryColor': '#d0f0fd'}}}%%
graph TD
    A[Input Image 640x640x3] --> B[Focus Layer]
    B --> C[Conv 3x3, 32]
    C --> D[BottleneckCSP, 64]
    D --> E[Conv 3x3, s=2]
    E --> F[BottleneckCSP, 128]
    F --> G[Conv 3x3, s=2]
    G --> H[BottleneckCSP, 256]
    H --> I[SPP Layer]
    I --> J[PANet Path Aggregation]
    J --> K[Detection Head]
    K --> L[Output: Bounding Boxes + Class Probabilities]
    
    style A fill:#4dabf7,stroke:#339af0,color:white
    style L fill:#40c057,stroke:#2b8a3e,color:white
```

### Architecture Components:
1. **Focus Layer**: Reduces spatial dimensions while increasing channel depth
2. **BottleneckCSP**: Cross Stage Partial Network for better gradient flow
3. **SPP Layer**: Spatial Pyramid Pooling for multi-scale feature extraction
4. **PANet**: Path Aggregation Network for better feature fusion
5. **Detection Head**: Predicts bounding boxes and class probabilities

---

## Data Flow Pipeline

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#e3fafc', 'primaryBorderColor': '#3bc9db', 'lineColor': '#66d9e8'}}}%%
sequenceDiagram
    participant V as Video Input
    participant P as Preprocessing
    participant M as Model
    participant O as Output
    
    V->>P: Raw Video Frames
    Note right of P: 1. Resize to 640x640<br>2. Normalize [0,1]<br>3. Data Augmentation
    P->>M: Processed Tensor
    M->>O: Predictions
    O->>+O: Post-processing
    O-->>-User: Visualized Output
```

### Preprocessing Steps:
1. Frame extraction at 5 FPS
2. Resize to 640x640
3. Normalize pixel values
4. Apply augmentations (Mosaic, MixUp, HSV)

---

## Training Process

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#fff3bf', 'primaryBorderColor': '#ffd43b'}}}%%
graph LR
    A[Initialize Model] --> B[Load Dataset]
    B --> C[Forward Pass]
    C --> D[Compute Loss]
    D --> E[Backpropagation]
    E --> F[Update Weights]
    F --> C
    D --> G[Validation]
    G -->|mAP > Best| H[Save Checkpoint]
    G -->|mAP ≤ Best| C
```

### Training Parameters:
- **Optimizer**: SGD with momentum (0.937)
- **Learning Rate**: 0.01 with cosine annealing
- **Batch Size**: 16
- **Epochs**: 100
- **Loss Function**: CIoU Loss + BCE Loss

---

## Inference Pipeline

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#e6fcf5', 'primaryBorderColor': '#20c997'}}}%%
flowchart TD
    A[Input Frame] --> B[Preprocess]
    B --> C[Model Forward Pass]
    C --> D[Decode Predictions]
    D --> E[Non-Max Suppression]
    E --> F[Draw Bounding Boxes]
    F --> G[Display Output]
    
    style A fill:#96f2d7
    style G fill:#96f2d7
```

### Post-processing:
1. **Confidence Thresholding**: Remove predictions < 0.25 confidence
2. **NMS**: IoU threshold of 0.45 to remove overlapping boxes
3. **Label Mapping**: Convert class IDs to human-readable labels

---

## Performance Optimization

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#f3d9fa', 'primaryBorderColor': '#cc5de8'}}}%%
pie
    title Model Optimization Techniques
    "Mixed Precision Training" : 35
    "Data Augmentation" : 25
    "Learning Rate Scheduling" : 20
    "Weight Decay" : 15
    "Gradient Clipping" : 5
```

### Optimization Details:
- **Mixed Precision**: FP16 training for faster computation
- **Data Augmentation**: Mosaic, MixUp, HSV, and perspective transforms
- **Learning Rate**: Cosine annealing with warmup
- **Weight Decay**: 0.0005 for regularization

---

## Class Distribution

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#ffe3e3', 'primaryBorderColor': '#ff8787'}}}%%
xychart-beta
    title "Class Distribution in Dataset"
    x-axis "Class" ["Accident", "Call", "Doctor", "Help", "Hot", "Lose", "Pain", "Thief"]
    y-axis "Count" 0 --> 60
    bar [52, 52, 52, 52, 52, 50, 52, 50]
```

### Dataset Statistics:
- Total samples: 412 videos
- 8 balanced classes
- 5 frames extracted per video
- Train/Val/Test split: 70%/15%/15%

---

## Model Performance

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#d8f5a2', 'primaryBorderColor': '#82c91e'}}}%%
xychart-beta
    title "Model Comparison (mAP@0.5)"
    x-axis ["3D CNN", "VGG16+LSTM", "YOLOv5s"]
    y-axis "mAP" 0 --> 100
    bar [30.1, 90.4, 94.0]
```

### Performance Metrics:
- **YOLOv5s**: 94.0 mAP@0.5
- **Inference Speed**: 15 FPS on CPU, 45 FPS on GPU
- **Model Size**: 14.4 MB (FP32), 7.2 MB (FP16)

---

## Real-time Processing

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#d0ebff', 'primaryBorderColor': '#4dabf7'}}}%%
stateDiagram-v2
    [*] --> CaptureFrame
    CaptureFrame --> Preprocess
    Preprocess --> Inference
    Inference --> Postprocess
    Postprocess --> Display
    Display --> [*]
```

### Real-time Features:
- Multi-threaded frame capture
- Asynchronous processing
- FPS counter and performance metrics
- Real-time visualization

---

## Future Enhancements

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#ffec99', 'primaryBorderColor': '#ffd43b'}}}%%
journey
    title Roadmap
    section Phase 1
      Mobile Optimization: 5: Done
      Real-time Processing: 5: Done
    section Phase 2
      Multi-hand Detection: 3
      Extended Vocabulary: 2
    section Phase 3
      Mobile App: 1
      Cloud Deployment: 1
```

### Planned Features:
1. **Multi-hand Detection**
2. **Expanded Vocabulary**
3. **Mobile Application**
4. **Cloud API**
5. **User Authentication**

---

## Color Legend

```mermaid
%%{init: {'theme': 'base'}}%%
graph LR
    A[Blue: Input/Output] --> B[Green: Processing]
    B --> C[Yellow: Training]
    C --> D[Purple: Optimization]
    D --> E[Red: Data Analysis]
    E --> F[Teal: Inference]
```

### Diagram Colors:
- 🔵 **Blue**: Input/Output operations
- 🟢 **Green**: Data processing
- 🟡 **Yellow**: Training process
- 🟣 **Purple**: Optimization techniques
- 🔴 **Red**: Data analysis
- 🟢 **Teal**: Real-time inference

---

*Last Updated: November 14, 2025*
