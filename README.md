# Summer-project-surgical-robotics
A compilation of the coding work I have undertaken this summer relating to surgical robotics
## MAE implementation
The folder labelled MAE_implementation contains my code that utilises meta's masked auto encoder. The goal was to produce a representation vector (128-D) to utilise for downstream tasks. This is the research pipeline: 
```text
Raw 2-hour RARP Video
        │
        ▼
Clip Surgical Step 5
        │
        ▼
Sample Frames (1 FPS)
        │
        ▼
Resize Images (224 × 224)
        │
        ▼
ImageNet Normalization
        │
        ▼
Save Preprocessed Clips
(clip_0000.pt ... clip_0003.pt)
        │
        ▼
SurgicalFrameDataset
        │
        ▼
PyTorch DataLoader
        │
        ▼
Frozen ImageNet MAE Encoder
(models_mae.py)
        │
        ▼
CLS Token Embedding (768-D)
        │
        ▼
Trainable MLP Adapter
(mae_wrapper.py)
        │
        ▼
Trainable MAE Decoder
(models_mae.py)
        │
        ▼
Masked Patch Reconstruction Loss
        │
        ▼
Optimize Adapter + Decoder
        │
        ▼
Extract Adapted CLS Embeddings (768-D)
        │
        ▼
Projection Head (128-D)
        │
        ▼
Downstream Robotics Tasks
• Action Recognition
• Phase Recognition
• Surgical Planning
• Imitation Learning
• Reinforcement Learning
```
For reasons involving privacy, protection and copyright I have not included the RARP video or the json/txt annotations.
## License
This project is under the Apache 2.0 license. See [License](https://github.com/maygops/Summer-project-surgical-robotics/blob/main/LICENSE) for details.