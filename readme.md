# **UNC Landmark Classification**
***

```python
import torch
import torch.nn.functional as F
from src.model.resnet import ResNet152


N_CLASSES = 5

model     = ResNet152(num_classes=N_CLASSES)
weights   = torch.load("weights/rs-152-c5-best_params.pth", weights_only=True)

# init model weights from checkpoint
model._parameters = weights

# [B, C, H, W]
image  = torch.rand(5, 3, 224, 224)

# [B, N]
pred   = model(image)
logits = F.softmax(pred, dim=1)

# [B, ]
cls_preds = torch.argmax(logits, dim=1)
class_map = {
    0: 'bell_tower',
    1: 'gerrard_hall',
    2: 'graham_hall', 
    3: 'person_hall', 
    4: 'south_building'
}

pred_names = [class_map[cls_preds[i].item()] for i in range(cls_preds.shape[0])]
```

