# Architecture Documentation

This document provides detailed architecture diagrams for all three instance segmentation models compared in this project.

---

## Overview: Training Pipeline

```mermaid
graph TB
    subgraph Input["Input: YOLO Dataset"]
        A[YOLO Format<br/>Normalized Polygons]
    end

    subgraph Conversion["Automatic Conversion"]
        B[Dataset Converter<br/>YOLO → COCO]
    end

    subgraph Models["Instance Segmentation Models"]
        direction TB
        C1[SOLOv2<br/>ResNet Backbone<br/>Grid-based]
        C2[RTMDet-Ins<br/>CSPNeXt Backbone<br/>One-stage]
        C3[YOLOv11-seg<br/>Custom Backbone<br/>Anchor-free]
    end

    subgraph Training["Training"]
        F[Matched Configuration<br/>YOLO Auto LR<br/>7-epoch Warmup<br/>Cosine Annealing]
    end

    subgraph Output["Output"]
        H[Trained Models<br/>.pth checkpoints]
        I[Metrics<br/>mAP50-95, mAP50, mAP75]
    end

    A --> B
    B --> C1
    B --> C2
    B --> C3
    C1 --> F
    C2 --> F
    C3 --> F
    F --> H
    F --> I

    style Input fill:#e1f5ff
    style Models fill:#fff4e1
    style Output fill:#e8f5e9
```

---

## SOLOv2 Architecture

**Type**: Grid-based instance segmentation
**Backbone**: ResNet (18/34/50/101)
**Key Feature**: Separate category and mask branches with spatial grid prediction

### High-Level Architecture

```mermaid
graph LR
    A[Input Image] --> B[ResNet Backbone]
    B --> C[FPN Neck]
    C --> D[Category Branch]
    C --> E[Mask Branch]
    C --> F[Mask Feature Head]
    D --> G[Category Predictions]
    E --> H[Kernel Predictions]
    F --> I[Mask Prototypes]
    H --> J[Instance Masks]
    I --> J
    G --> K[Final Output]
    J --> K

    style A fill:#e1f5ff
    style B fill:#fff4e1
    style C fill:#ffe1f5
    style K fill:#e8f5e9
```

### Detailed Internal Architecture

```mermaid
graph TB
    subgraph Input["Input"]
        IMG[Image<br/>H×W×3]
    end

    subgraph Backbone["ResNet Backbone"]
        direction TB
        C1[Conv1<br/>7×7, stride=2]
        C2[Stage 2<br/>C2: H/4×W/4]
        C3[Stage 3<br/>C3: H/8×W/8]
        C4[Stage 4<br/>C4: H/16×W/16]
        C5[Stage 5<br/>C5: H/32×W/32]

        C1 --> C2
        C2 --> C3
        C3 --> C4
        C4 --> C5
    end

    subgraph FPN["Feature Pyramid Network"]
        direction TB
        P2[P2: H/4×W/4<br/>256 channels]
        P3[P3: H/8×W/8<br/>256 channels]
        P4[P4: H/16×W/16<br/>256 channels]
        P5[P5: H/32×W/32<br/>256 channels]
        P6[P6: H/64×W/64<br/>256 channels]
    end

    subgraph Head["SOLOv2 Head"]
        direction TB

        subgraph Category["Category Branch"]
            CAT1[Conv Stack<br/>4 layers]
            CAT2[Grid System<br/>S×S cells]
            CAT3[Classification<br/>num_classes]
        end

        subgraph Mask["Mask Branch"]
            MASK1[Conv Stack<br/>4 layers]
            MASK2[Mask Features<br/>E channels]
            MASK3[Kernel Prediction<br/>S×S×E]
        end

        subgraph MaskFeat["Mask Feature Head"]
            MF1[Multi-level Features<br/>P2→P3→P4]
            MF2[Feature Fusion<br/>128 channels]
            MF3[Mask Prototype<br/>H/4×W/4×E]
        end
    end

    subgraph Output["Output"]
        direction TB
        OUT1[Category Scores<br/>S×S×num_classes]
        OUT2[Instance Masks<br/>N×H×W]
        OUT3[Final Predictions<br/>Class + Mask per instance]
    end

    IMG --> C1

    C2 --> P2
    C3 --> P3
    C4 --> P4
    C5 --> P5
    P5 --> P6

    P2 --> Category
    P3 --> Category
    P4 --> Category
    P5 --> Category

    P2 --> Mask
    P3 --> Mask
    P4 --> Mask
    P5 --> Mask

    P2 --> MaskFeat
    P3 --> MaskFeat
    P4 --> MaskFeat

    CAT1 --> CAT2
    CAT2 --> CAT3
    CAT3 --> OUT1

    MASK1 --> MASK2
    MASK2 --> MASK3

    MF1 --> MF2
    MF2 --> MF3

    MASK3 --> OUT2
    MF3 --> OUT2

    OUT1 --> OUT3
    OUT2 --> OUT3

    style Input fill:#e1f5ff
    style Backbone fill:#fff4e1
    style FPN fill:#ffe1f5
    style Head fill:#e1ffe1
    style Output fill:#e8f5e9
```

### SOLOv2 Key Components

1. **ResNet Backbone**: Extracts hierarchical features at multiple scales (C2-C5)
2. **FPN Neck**: Creates feature pyramid with 256 channels per level (P2-P6)
3. **Category Branch**:
   - 4-layer conv stack
   - Grid-based classification (S×S cells)
   - Predicts object category for each grid cell
4. **Mask Branch**:
   - 4-layer conv stack
   - Predicts dynamic convolution kernels (S×S×E)
   - Each kernel generates one instance mask
5. **Mask Feature Head**:
   - Fuses P2, P3, P4 features
   - Creates mask prototypes at H/4×W/4 resolution
   - High-resolution features for precise masks

**Strengths**:
- Separate branches allow specialized learning
- Grid-based prediction enables dense instance localization
- Multi-level mask features for precise boundaries

---

## RTMDet-Ins Architecture

**Type**: One-stage anchor-free instance segmentation
**Backbone**: CSPNeXt (modern, efficient)
**Key Feature**: Shared detection and segmentation head with dynamic mask prediction

### High-Level Architecture

```mermaid
graph LR
    A[Input Image] --> B[CSPNeXt Backbone]
    B --> C[CSPNeXtPAFPN Neck]
    C --> D[RTMDet-Ins Head]
    D --> E[Classification Branch]
    D --> F[BBox Regression Branch]
    D --> G[Mask Branch]
    E --> H[Class Scores]
    F --> I[Bounding Boxes]
    G --> J[Instance Masks]
    H --> K[Final Output]
    I --> K
    J --> K

    style A fill:#e1f5ff
    style B fill:#fff4e1
    style C fill:#ffe1f5
    style K fill:#e8f5e9
```

### Detailed Internal Architecture

```mermaid
graph TB
    subgraph Input["Input"]
        IMG[Image<br/>H×W×3]
    end

    subgraph Backbone["CSPNeXt Backbone"]
        direction TB
        STEM[Stem<br/>Focus + Conv]
        STAGE1[Stage 1<br/>CSPNeXt Block<br/>H/4×W/4]
        STAGE2[Stage 2<br/>CSPNeXt Block<br/>H/8×W/8]
        STAGE3[Stage 3<br/>CSPNeXt Block<br/>H/16×W/16]
        STAGE4[Stage 4<br/>CSPNeXt Block<br/>H/32×W/32]

        STEM --> STAGE1
        STAGE1 --> STAGE2
        STAGE2 --> STAGE3
        STAGE3 --> STAGE4
    end

    subgraph Neck["CSPNeXtPAFPN"]
        direction TB

        subgraph TopDown["Top-Down Path"]
            TD1[P5 → P4]
            TD2[P4 → P3]
        end

        subgraph BottomUp["Bottom-Up Path"]
            BU1[P3 → P4]
            BU2[P4 → P5]
        end

        OUT_P3[P3 Out<br/>H/8×W/8]
        OUT_P4[P4 Out<br/>H/16×W/16]
        OUT_P5[P5 Out<br/>H/32×W/32]
    end

    subgraph Head["RTMDet-Ins SepBN Head"]
        direction TB

        subgraph SharedConv["Shared Conv Layers"]
            CONV1[Conv Stack<br/>2 stacked convs<br/>Share across levels]
        end

        subgraph ClsBranch["Classification Branch"]
            CLS_CONV[Conv Layers]
            CLS_OUT[Class Prediction<br/>num_classes]
        end

        subgraph BBoxBranch["BBox Branch"]
            BBOX_CONV[Conv Layers]
            BBOX_OUT[BBox Prediction<br/>4 coords]
        end

        subgraph MaskBranch["Mask Branch"]
            MASK_CONV[Conv Layers]
            MASK_KERNEL[Kernel Prediction<br/>Dynamic convs]
        end

        subgraph MaskFeat["Mask Feature"]
            MASK_PROTO[Mask Prototypes<br/>H/4×W/4]
        end
    end

    subgraph Output["Output"]
        direction TB
        DET[Detections<br/>Class + BBox]
        MASKS[Instance Masks<br/>N×H×W]
        FINAL[Final Predictions<br/>Class + BBox + Mask]
    end

    IMG --> STEM

    STAGE1 --> OUT_P3
    STAGE2 --> OUT_P3
    STAGE3 --> OUT_P4
    STAGE4 --> OUT_P5

    OUT_P3 --> CONV1
    OUT_P4 --> CONV1
    OUT_P5 --> CONV1

    CONV1 --> CLS_CONV
    CONV1 --> BBOX_CONV
    CONV1 --> MASK_CONV

    CLS_CONV --> CLS_OUT
    BBOX_CONV --> BBOX_OUT
    MASK_CONV --> MASK_KERNEL

    OUT_P3 --> MASK_PROTO
    OUT_P4 --> MASK_PROTO

    CLS_OUT --> DET
    BBOX_OUT --> DET
    MASK_KERNEL --> MASKS
    MASK_PROTO --> MASKS

    DET --> FINAL
    MASKS --> FINAL

    style Input fill:#e1f5ff
    style Backbone fill:#fff4e1
    style Neck fill:#ffe1f5
    style Head fill:#e1ffe1
    style Output fill:#e8f5e9
```

### RTMDet-Ins Key Components

1. **CSPNeXt Backbone**:
   - Modern efficient architecture
   - Cross-Stage Partial connections
   - Channel attention for feature refinement
   - 4 stages with increasing receptive fields

2. **CSPNeXtPAFPN Neck**:
   - Path Aggregation FPN
   - Top-down + bottom-up fusion
   - Rich multi-scale features (P3, P4, P5)

3. **Shared Conv Layers**:
   - 2 stacked convolutions
   - Shared across all FPN levels
   - Efficient feature processing

4. **Three Prediction Branches**:
   - **Classification**: Object class scores
   - **BBox Regression**: Bounding box coordinates (DistancePointBBoxCoder)
   - **Mask Branch**: Dynamic convolution kernels for masks

5. **Mask Feature Generation**:
   - Fuses P3 and P4 features
   - Generates mask prototypes at H/4×W/4
   - Combined with dynamic kernels for final masks

6. **Dynamic Soft Label Assignment**:
   - DynamicSoftLabelAssigner with topk=13
   - Automatically assigns positive samples during training
   - No manual anchor design needed

**Strengths**:
- Modern, efficient backbone (CSPNeXt)
- One-stage design for fast inference
- Shared convolutions reduce parameters
- Dynamic label assignment for better training
- AdamW optimization for fast convergence

---

## YOLOv11-seg Architecture

**Type**: Anchor-free one-stage instance segmentation
**Backbone**: Custom CSPDarknet-based
**Key Feature**: Lightweight, ultra-fast inference

### High-Level Architecture

```mermaid
graph LR
    A[Input Image] --> B[YOLOv11 Backbone]
    B --> C[YOLO Neck<br/>PANet]
    C --> D[Detect Head]
    C --> E[Segment Head]
    D --> F[BBox + Class]
    E --> G[Mask Coefficients]
    G --> H[Prototype Masks]
    H --> I[Instance Masks]
    F --> J[Final Output]
    I --> J

    style A fill:#e1f5ff
    style B fill:#fff4e1
    style C fill:#ffe1f5
    style J fill:#e8f5e9
```

### Detailed Architecture

```mermaid
graph TB
    subgraph Input["Input"]
        IMG[Image<br/>H×W×3]
    end

    subgraph Backbone["YOLOv11 Backbone"]
        direction TB
        CONV1[Conv + SPPF<br/>H/2×W/2]
        C3K1[C3k2 Block<br/>H/4×W/4]
        C3K2[C3k2 Block<br/>H/8×W/8]
        C3K3[C3k2 Block<br/>H/16×W/16]
        C3K4[C3k2 Block<br/>H/32×W/32]

        CONV1 --> C3K1
        C3K1 --> C3K2
        C3K2 --> C3K3
        C3K3 --> C3K4
    end

    subgraph Neck["PANet Neck"]
        direction TB
        UP1[Upsample + Concat]
        UP2[Upsample + Concat]
        DOWN1[Downsample + Concat]
        DOWN2[Downsample + Concat]

        P3[P3: H/8×W/8]
        P4[P4: H/16×W/16]
        P5[P5: H/32×W/32]
    end

    subgraph DetHead["Detection Head"]
        direction TB
        DET1[Detect Layer<br/>P3, P4, P5]
        CLS[Classification]
        REG[Regression]
        DFL[Distribution Focal Loss]
    end

    subgraph SegHead["Segmentation Head"]
        direction TB
        PROTO[Prototype Head<br/>32 prototypes<br/>H/4×W/4]
        COEF[Mask Coefficients<br/>Per detection]
        COMBINE[Linear Combination<br/>Coef × Prototypes]
    end

    subgraph Output["Output"]
        direction TB
        BBOX[BBoxes + Classes]
        MASKS[Instance Masks]
        OUT[Final Predictions]
    end

    IMG --> CONV1

    C3K2 --> UP2
    C3K3 --> UP1
    C3K4 --> UP1

    UP2 --> P3
    UP1 --> P4
    C3K4 --> P5

    P3 --> DET1
    P4 --> DET1
    P5 --> DET1

    DET1 --> CLS
    DET1 --> REG
    REG --> DFL

    P3 --> PROTO
    DET1 --> COEF

    CLS --> BBOX
    DFL --> BBOX

    PROTO --> COMBINE
    COEF --> COMBINE
    COMBINE --> MASKS

    BBOX --> OUT
    MASKS --> OUT

    style Input fill:#e1f5ff
    style Backbone fill:#fff4e1
    style Neck fill:#ffe1f5
    style DetHead fill:#e1ffe1
    style SegHead fill:#ffe1e1
    style Output fill:#e8f5e9
```

### YOLOv11-seg Key Components

1. **YOLOv11 Backbone**:
   - Custom C3k2 blocks (evolved from CSPDarknet)
   - SPPF for multi-scale receptive fields
   - Very lightweight (2.9M params for nano)

2. **PANet Neck**:
   - Path Aggregation Network
   - Bidirectional feature fusion
   - 3 output scales (P3, P4, P5)

3. **Detection Head**:
   - Anchor-free detection
   - Distribution Focal Loss (DFL) for bbox regression
   - Shared across 3 scales

4. **Segmentation Head**:
   - **Prototype-based masks**: 32 prototype masks at H/4×W/4
   - **Mask coefficients**: Each detection predicts coefficients
   - **Linear combination**: Final mask = Σ(coef_i × prototype_i)
   - Fast but less precise than kernel-based approaches

**Strengths**:
- Extremely fast inference (7x faster training)
- Smallest model (2.9M params)
- Good for real-time applications
- Prototype-based masks are efficient

**Limitations**:
- Prototype masks less precise at high IoU thresholds
- Lower mAP75 compared to SOLOv2/RTMDet-Ins
- Fixed number of prototypes (32) limits mask expressiveness

---

## Architecture Comparison

| Feature | SOLOv2 | RTMDet-Ins | YOLOv11-seg |
|---------|--------|------------|-------------|
| **Backbone** | ResNet (18/34/50/101) | CSPNeXt | Custom C3k2 |
| **Neck** | FPN | CSPNeXtPAFPN | PANet |
| **Detection** | Grid-based | Anchor-free | Anchor-free |
| **Mask Strategy** | Dynamic kernels | Dynamic kernels | Prototypes |
| **Mask Resolution** | 56×56 kernels | Dynamic convs | 32 prototypes |
| **Branches** | Separate (Cat + Mask) | Shared base + 3 heads | Detection + Seg |
| **Optimizer** | SGD | AdamW | SGD/AdamW |
| **Strengths** | Precise boundaries | Best overall | Fastest |
| **Best For** | High-precision masks | Balanced | Real-time |

### Mask Prediction Strategies

1. **SOLOv2 (Grid + Kernels)**:
   - Divides image into S×S grid
   - Each cell predicts category
   - Dynamic kernels (S×S×E) applied to mask features
   - Most precise boundaries (87.2% mAP75)

2. **RTMDet-Ins (Dynamic Kernels)**:
   - Anchor-free detection first
   - Each detection predicts dynamic conv kernel
   - Applied to shared mask features
   - Best overall consistency (75.3% mAP50-95)

3. **YOLOv11-seg (Prototypes)**:
   - Learns 32 fixed prototype masks
   - Each detection predicts coefficients
   - Linear combination of prototypes
   - Fastest but less precise (62.0% mAP50-95)

### Why SOLOv2 & RTMDet-Ins Excel

Both use **dynamic kernel prediction** which allows:
- Each instance generates custom convolution kernel
- Kernels adapted to specific object shape/size
- Higher mask expressiveness than fixed prototypes
- Better performance at high IoU thresholds

YOLOv11's prototype approach is:
- ✅ Very fast (linear combination vs. convolution)
- ✅ Efficient (only 32 prototypes to learn)
- ⚠️ Limited expressiveness (linear combination constraint)
- ⚠️ Struggles with complex shapes at high IoU

---

## Performance vs. Architecture

Results on Lingfield Racetrack (3 classes, 217 train, 99 val):

| Model | Architecture Type | mAP50-95 | mAP75 | Training Time |
|-------|------------------|----------|-------|---------------|
| **SOLOv2-nano** | Grid + Dynamic Kernels | **74.5%** | **87.2%** | 90 min |
| **RTMDet-Ins-tiny** | One-stage + Dynamic Kernels | **75.3%** | 80.6% | 85 min |
| **YOLOv11n-seg** | One-stage + Prototypes | 62.0% | ~80% | 12 min |

**Key Insight**: Dynamic kernel approaches (SOLOv2, RTMDet-Ins) achieve 12-13% higher mAP50-95 than prototype-based approach (YOLO), demonstrating the importance of mask prediction strategy.

---


