import os
import shutil
from sklearn.model_selection import train_test_split

data_dir = "../data/raw"
processed_dir = "../data/processed"

if os.path.exists(processed_dir):
    shutil.rmtree(processed_dir)
os.makedirs(processed_dir, exist_ok=True)

# Create processed directories
for split in ['train', 'val', 'test']:
    for cls in os.listdir(data_dir):
        if os.path.isdir(os.path.join(data_dir, cls)):
            os.makedirs(os.path.join(processed_dir, split, cls), exist_ok=True)

# Split and copy images
for cls in sorted(os.listdir(data_dir)):
    cls_path = os.path.join(data_dir, cls)
    if not os.path.isdir(cls_path):
        continue

    images = [f for f in os.listdir(cls_path) if not f.startswith('.')]

    train, temp = train_test_split(images, test_size=0.30, random_state=42)
    val, test = train_test_split(temp, test_size=0.50, random_state=42)

    for split, files in [('train', train), ('val', val), ('test', test)]:
        for f in files:
            src = os.path.join(cls_path, f)
            dst = os.path.join(processed_dir, split, cls, f)
            shutil.copy2(src, dst)

    print(f"{cls}: train={len(train)}, val={len(val)}, test={len(test)}")