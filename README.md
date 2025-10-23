# Arabic Sentiment Analysis Notebook (Training MARBERT)

This repository contains the Jupyter Notebook (`.ipynb`) used to train the Arabic Sentiment Analysis model. The notebook details the entire process, from data loading and cleaning to fine-tuning and evaluating the `MARBERT` model.

The final trained model is hosted on Hugging Face.

* **🚀 Live Demo (Gradio):** **[https://huggingface.co/spaces/iMeshal/arabic-sentiment-app](https://huggingface.co/spaces/iMeshal/arabic-sentiment-app)**
* **📦 Trained Model:** **[iMeshal/arabic-sentiment-classifier-marbert](https://huggingface.co/iMeshal/arabic-sentiment-classifier-marbert)**

---

## 📖 Methodology (The Notebook's Process)

This notebook implements a full fine-tuning pipeline for a transformer model using Hugging Face `transformers` and `datasets`.

1.  **Data Loading:** The [Arabic Sentiment Twitter Corpus](https://www.kaggle.com/datasets/mksaad/arabic-sentiment-twitter-corpus) dataset is loaded from Kaggle.
2.  **Exploratory Data Analysis (EDA):** Analysis of class balance (perfectly 50/50) and text length (identifying noise/long tweets).
3.  **Data Cleaning:**
    * Identified and processed overly long tweets (>300 chars) which appeared to be concatenated records.
    * Split these records into individual, clean tweets.
    * Consolidated the dataset, resulting in a clean, balanced corpus.
4.  **Data Preparation:**
    * Split the data into Training (80%) and Validation (20%) sets using `stratify` to maintain balance.
    * Converted the data into a Hugging Face `DatasetDict`.
5.  **Model & Tokenizer:**
    * Loaded the `UBC-NLP/MARBERTv2` pre-trained model and its tokenizer, as it's optimized for Arabic social media text.
    * Due to library version conflicts, the environment was pinned to a compatible set of `transformers (4.30.0)` and related libraries.
6.  **Tokenization:**
    * Text labels (`'neg'`, `'pos'`) were converted to integers (`0`, `1`).
    * A tokenization function was applied to pre-pad and truncate all texts to `max_length=512`.
7.  **Training:**
    * The model was fine-tuned using the `transformers.Trainer`.
    * Key settings: `learning_rate=2e-5`, `batch_size=16`, `load_best_model_at_end=True`, and `EarlyStoppingCallback`.
    * The model achieved its best validation accuracy (93.4%) at Epoch 2.

---

## 📊 Final Performance

After training, the best model (from Epoch 2) was evaluated on a separate, unseen **Test Set**.

**Final Test Set Results:**

| Metric | Score |
| :--- | :---: |
| **Accuracy** | **94.40%** |
| F1 (Macro) | 94.40% |
| Precision | 94.40% |
| Recall | 94.40% |

The strong performance on the test set (exceeding the validation score) confirms the model generalizes well to new data.

---

### 📞 Contact

* **Name:** Meshal AL-Qushaym
* **Email:** meshalqushim@outlook.com
* **Kaggle:** [kaggle.com/meshalfalah](https://www.kaggle.com/meshalfalah)
