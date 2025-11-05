# **UNC Landmark Classification**
***

```python
import torch
import torch.nn.functional as F
from src.model.resnet import ResNet152


model: ResNet152 = torch.load("...pth")

# [B, H, W, C]
image  = torch.rand(1, 224, 224, 3)

# [B, C]
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
print(f"Model predictions (B=0, ... B=n): {pred_names}")
```

