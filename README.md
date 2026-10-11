# Fruit Ripeness Classifier

Upload a photo of an apple, banana or mango; the model predicts the fruit and whether it is ripe or unripe.
Trained by me in Google Colab with Ultralytics YOLOv8n-cls.

**Live demo:** https://phantom-fruit-ripeness-classifier.streamlit.app/

## Results (145 held-out test photos, 6 classes)
| Model | Val accuracy | Test accuracy |
|---|---|---|
| From scratch (100 epochs) | 83.9% | 73.1% |
| Fine-tuned from pretrained (30 epochs) | 96.8% | **95.2%** |

With only 364 training photos, starting from pretrained weights beat training from scratch by ~22 points,
even though the scratch model trained 3x longer.

## What I learned along the way
1. I first trained an **object detector** (YOLOv8n) on a Roboflow fruit dataset: 98.6% mAP50 on its own validation set.
2. Tested on an **independent 571-photo dataset**, it dropped to 44% fruit-type / 39% fruit+ripeness accuracy
   (mostly predicting "banana") - the training data was too narrow (domain shift).
3. I switched to an **image classifier** trained on the independent dataset's train split and evaluated on its test split.
4. **Error analysis** of the 7 test mistakes: 4 fruit-type confusions (mostly green apple vs green mango),
   3 ripeness errors; at least 2 involved ambiguous labels or photos with many fruits.

## Limitations
- Test photos come from the same source as the training photos, so real-world accuracy may be lower.
- 145 test photos: results are accurate to roughly +/- 3-4 points.
- Three fruits only, ripe/unripe only (no overripe), one fruit per photo works best.

## Run locally
```bash
pip install -r requirements.txt
# put best.pt in the model/ folder
python -m streamlit run app.py
```
