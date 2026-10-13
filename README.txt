RADAR
  ↓
SIGNAL PROCESSING
  ↓
DETECTIONS
  │
  │
  ├───────────────┐
  │               │
  ▼               ▼
Kalman         Detection
PREDICT        radar
  │               │
  └───────┬───────┘
          ▼
   DATA ASSOCIATION
   Mahalanobis
   Gating
   Hungarian
          │
          ▼
    Kalman UPDATE
          │
          ▼
    nuovo stato