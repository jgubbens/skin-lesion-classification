import os
import pandas as pd
from sklearn.model_selection import train_test_split

data_dir = "../data/raw"
splits_dir = "../data/splits"
os.makedirs(splits_dir, exist_ok=True)

# Create dataframe with file path and label of all images
rows = []
for cls in sorted(os.listdir(data_dir)):
    cls_path = os.path.join(data_dir, cls)
    if not os.path.isdir(cls_path):
        continue
    for img in os.listdir(cls_path):
        if img.startswith('.'):
            continue
        rows.append({
            'filepath': os.path.join(cls_path, img),
            'label': cls
        })

df = pd.DataFrame(rows)

# Train/val/test split
train_df, temp_df = train_test_split(df, test_size=0.30, stratify=df['label'], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df['label'], random_state=42)

# Save splits
train_df.to_csv(os.path.join(splits_dir, 'train.csv'), index=False)
val_df.to_csv(os.path.join(splits_dir, 'val.csv'), index=False)
test_df.to_csv(os.path.join(splits_dir, 'test.csv'), index=False)

print(f"Train: {len(train_df)}")
print(f"Val: {len(val_df)}")
print(f"Test: {len(test_df)}")
print("\nClass distribution in train:")
print(train_df['label'].value_counts())