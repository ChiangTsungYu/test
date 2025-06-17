graph TD
    %% Encoder + Projection Head
    A[Input: (特徵數, 1)] --> B[Conv1D(128, kernel=3) + ReLU]
    B --> C[BatchNorm]
    C --> D[MaxPooling1D(2)]

    D --> E[Conv1D(256, kernel=3) + ReLU]
    E --> F[BatchNorm]
    F --> G[MaxPooling1D(2)]

    G --> H[Conv1D(512, kernel=3) + ReLU]
    H --> I[BatchNorm]
    I --> J[GlobalAveragePooling1D]

    J --> K[Dense(512) + ReLU]
    K --> L[Dropout(0.5)]
    L --> M[Projection Head: Dense(128)]

    %% Supervised Contrastive Loss 接這裡
    M --> N[Supervised Contrastive Loss]

    %% Classifier 微調路徑
    M --> O[Classifier Head: Dense(128) + ReLU]
    O --> P[Dropout(0.3)]
    P --> Q[Dense(num_classes, softmax)]

    %% 輸出預測
    Q --> R[分類結果]
