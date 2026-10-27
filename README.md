# Video Violence Detection (MobileNetV2 + BiLSTM)

Classifies a short video clip as **Violence** or **NonViolence**. MobileNetV2 extracts features from each frame and a Bidirectional LSTM models how they change over time. The repository also contains a FastAPI web app (video upload and live webcam), TFLite quantization experiments, and a PyTorch re-implementation.

## Results

Dataset: [Real Life Violence Situations](https://www.kaggle.com/datasets/mohamedmustafa/real-life-violence-situations-dataset) (1,000 violence + 1,000 non-violence videos), stratified 80 / 20 split (1,600 train / 400 validation).

**v3** (MobileNetV2 + 2-layer BiLSTM, 32 frames per clip) on the 400-video validation split:

| Class | Precision | Recall | F1 |
|---|---|---|---|
| NonViolence | 0.99 | 0.97 | 0.98 |
| Violence | 0.98 | 0.99 | 0.99 |
| **Accuracy** | | | **0.98** |

Notebook with the code and evaluation: [kaggle.com/code/basanttyasser/mobilstmv3](https://www.kaggle.com/code/basanttyasser/mobilstmv3)

Caveats:
- The validation split was also used for early stopping and checkpoint selection, and there is no separate test set, so treat 98% as a validation result.
- The evaluation script samples one contiguous high-motion window of frames, while training samples the highest-motion frame in each time segment. Use the same sampling at inference time as the one you evaluated.
- Trained on public YouTube clips only. Performance on CCTV or other footage has not been measured.

## Repository layout

| Path | What it is |
|---|---|
| `app/` | FastAPI app: `POST /predict` for uploaded videos and a WebSocket `/ws/live` for live webcam detection, with a small HTML/JS front end in `app/static/` |
| `gradio_demo/` | Simple Gradio demo for an uploaded clip |
| `notebooks/` | TFLite quantization (dynamic range, float16, full INT8) and quantization-aware training |
| `pytorch_version/` | PyTorch version of the pipeline (dataset, preprocessing, training with AMP, inference) |

## The web app

- **Upload mode:** the clip is sampled two ways (uniform and highest-motion window). If either flags violence, the clip is flagged, and the higher score is reported.
- **Live mode:** a rolling buffer of frames from the browser webcam is sent over a WebSocket and scored continuously.
- The decision threshold is lowered to 0.35 to reduce missed violent clips (at the cost of more false alarms).

```bash
cd app
pip install -r ../requirements.txt
# put your trained Keras weights next to main.py and set MODEL_PATH / SEQUENCE_LENGTH to match them
python main.py     # serves on http://localhost:8000
```

Model weights are not stored in this repository. Published weights and model cards:
- v1 (16 frames): [huggingface.co/basant18/MoBiLSTM-violence-detection](https://huggingface.co/basant18/MoBiLSTM-violence-detection)
- Quantized TFLite experiments: [huggingface.co/basant18/Post-Training-Quantization](https://huggingface.co/basant18/Post-Training-Quantization)

## Quantization experiments

Sizes of the TFLite files converted from the 16-frame `modelv2.keras`:

| Format | Size |
|---|---|
| Keras (`modelv2.keras`, saved with optimizer state) | 48.4 MB |
| Dynamic-range quantization | 7.16 MB |
| Float16 | 7.11 MB |
| Quantization-aware training (INT8) | 5.55 MB |
| Full INT8 | 4.45 MB |

The Keras file was saved with the Adam optimizer state, so it is larger than the weights alone; compare against a float32 TFLite export for a like-for-like reduction. The full-INT8 conversion used random calibration data, and accuracy of the quantized models has not been evaluated here.

## Tech

Python, TensorFlow/Keras, OpenCV, FastAPI, WebSockets, TFLite, PyTorch, Hugging Face Hub.

## Intended use

Educational and research use. A person should review predictions before any action is taken.
