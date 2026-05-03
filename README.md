# Skin Lesion Classification

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/justingubbens/skin-lesion-classification.git
cd skin-lesion-classification
```

### 2. Create virtual environment
```bash
python -m venv venv
```

**Mac/Linux:**
```bash
source venv/bin/activate
```

**Windows:**
```bash
venv\Scripts\activate
```

```
pip install -r requirements.txt
```

### 3. Download the data
1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/salviohexia/isic-2019-skin-lesion-images-for-classification)
2. Unzip and place contents in `data/raw/`
3. Run `python scripts/prepare_data.py` to generate splits

## Dataset

https://www.kaggle.com/datasets/salviohexia/isic-2019-skin-lesion-images-for-classification

The ISIC (International Skin Imaging Collaboration) 2019 Skin Lesion Images for Classification dataset contains over 25,000 dermascopic images commonly used to train skin cancer detection models. It contains images across 8 diagnostic categories:

- Melanoma
- Melanocytic nevus
- Basal cell carcinoma
- Actinic keratosis
- Benign keratosis
- Dermatofibroma
- Vascular lesion
- Squamous cell carcinoma