# GENREation — Music Genre Classification Using CNN

GENREation is an educational deep-learning project that predicts one of ten
music genres from an uploaded audio file. It uses MFCC audio features, a
TensorFlow/Keras CNN, and a Streamlit interface.

## Objective

Predict the genre of an audio file and display the predicted genre, the
model's softmax score, the top-three predictions, and an optional log-mel
spectrogram.

## Supported genres

Blues, Classical, Country, Disco, Hip-Hop, Jazz, Metal, Pop, Reggae, and Rock.

## Technology stack

- Python
- Librosa — audio loading and MFCC/spectrogram extraction
- TensorFlow / Keras — CNN model
- Scikit-learn — stratified split and evaluation metrics
- Streamlit — interactive prediction interface
- Matplotlib / Seaborn — visualizations
- NumPy / Pandas — feature and result handling

## Project structure

```text
music_genre_classification/
├── Data/                       # GTZAN data; do not commit
│   └── genres_original/        # or genre folders directly under Data/
├── artifacts/                  # selected model and evaluation files
├── experiments/
│   └── v2/                      # isolated baseline-restoration experiment
├── notebooks/
│   └── Music_Genre_Classification.ipynb
├── src/
│   ├── __init__.py
│   └── audio_features.py
├── app.py
├── train.py                    # original training script
├── train_v2.py                 # restored baseline, isolated outputs
├── requirements.txt
├── .gitignore
└── README.md
```

## Dataset

Use the GTZAN Genre Collection from the source supplied by your course or a
reputable dataset mirror. Arrange the data in either of these layouts:

```text
Data/
├── blues/
├── classical/
├── country/
├── disco/
├── hiphop/
├── jazz/
├── metal/
├── pop/
├── reggae/
└── rock/
```

or:

```text
Data/
└── genres_original/
    ├── blues/
    ├── classical/
    └── ... (the remaining genre folders)
```

Each genre folder should contain its audio files. Dataset mirrors can differ
in structure and quality. Follow the dataset's license and terms. Do not
commit the dataset to GitHub.

## Environment setup (Windows PowerShell)

Use the Python environment in which TensorFlow is already installed. The
project has been tested in the user's workflow with a Python 3.12 virtual
environment; confirm your installed TensorFlow/Python compatibility if
creating a fresh environment.

```powershell
py -3.12 -m venv .venv312
.\.venv312\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, run this for the current terminal:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv312\Scripts\Activate.ps1
```

## Train the original baseline

From the project root:

```powershell
python train.py --data_dir Data --epochs 30 --batch_size 8
```

The original script writes its selected model and evaluation files into
`artifacts/`. Preserve a copy of the existing baseline artifacts before
retraining because these filenames are overwritten.

## Run the isolated baseline-restoration experiment

The supplied `train_v2.py` restores the simpler original MFCC CNN architecture
and writes its results into `experiments/v2/`, leaving `artifacts/` untouched.

Optional small-subset diagnostic:

```powershell
python train_v2.py --data_dir Data --epochs 30 --batch_size 8 --diagnostic
```

The diagnostic trains a separate model on a small subset of training songs.
It is a pipeline sanity check only, not a validation result or a metric to
report in the project.

Full experiment:

```powershell
python train_v2.py --data_dir Data --epochs 30 --batch_size 8
```

Review these files under `experiments/v2/`:

- `music_genre_cnn.keras`
- `labels.json`
- `evaluation.json`
- `confusion_matrix.png`
- `training_history.png`
- `training_history.csv`
- `data_split.json`
- `skipped_files.json`

Only after reviewing the evaluation should you decide whether to promote the
V2 model into `artifacts/`. To use it in the existing app, copy the selected
`experiments/v2/music_genre_cnn.keras` and `experiments/v2/labels.json` into
`artifacts/` as `music_genre_cnn.keras` and `labels.json`, respectively.
Keep a backup of the prior model.

## Run the Streamlit application

```powershell
streamlit run app.py
```

The application supports WAV, MP3, OGG, FLAC, and AU uploads (subject to the
installed audio codecs). It displays the predicted genre, softmax score,
top-three genre scores, and optional log-mel spectrogram.

The app and training pipeline both use `src/audio_features.py`. This shared
preprocessing is important: inference features must use the same sample rate,
duration, MFCC count, frame count, and normalization as training.

## Notebook

Open `notebooks/Music_Genre_Classification.ipynb` in VS Code or Jupyter.
Run the cells from top to bottom after placing the GTZAN data in `Data/`.
The notebook documents dataset inspection, MFCC extraction, the stratified
80/20 split, CNN training, evaluation, and artifact saving.

## Evaluation and reporting

Report the metrics produced by your own final training run. Do not copy
historical numbers into the final report unless they belong to the model
being submitted.

The original run supplied during development reported 58.5% validation
accuracy and 54.47% macro F1. A later experimental run reported 10.5%
validation accuracy and 4.77% macro F1. These results show why experiments
must be compared and why the later model should not automatically replace
the earlier baseline. They are historical development results, not a claim
about the final model.

The validation split is a stratified random track-level split. GTZAN has
known dataset-quality limitations, including possible duplicates and
overlap. Treat the result as an educational benchmark, not a definitive
measure of real-world generalization. For stronger evaluation, investigate
duplicates and use artist-independent splitting if reliable metadata is
available.

## Deliverables

- Jupyter notebook documenting the workflow
- Trained CNN model (`.keras`)
- Genre label mapping (`labels.json`)
- Streamlit audio prediction demo
- Evaluation report and confusion matrix
- README documentation

## Disclaimer

This is an educational music-classification project. Predictions may be inaccurate for mixed genres, noisy recordings, short clips, or music outside the GTZAN training distribution. Softmax scores are not calibrated probabilities and do not guarantee correctness.