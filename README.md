# Service-Desk-Ticket-Classification-with-Deep-Learning

## 📌 Overview

This project builds a **text classification system** capable of categorizing customer complaints into predefined topics such as:

* Mortgage
* Credit Card
* Money Transfers
* Debt Collection
* And more...

Using a **PyTorch-based neural network**, the system is trained on tokenized and preprocessed customer text data, with metrics like accuracy, precision, and recall used to evaluate performance.

---

## 📁 File Structure

```
.
├── notebook.ipynb        # Main Jupyter notebook (training pipeline, evaluation, etc.)
├── words.json            # Vocabulary list
├── text.json             # Tokenized text data
├── labels.npy            # Encoded category labels
├── servicedesk.png       # Project banner image
```

---

## ⚙️ Features

* 📑 **Text preprocessing** using NLTK
* 🧠 **Neural network model** built with PyTorch
* 📊 **Training pipeline** with performance metrics
* 🧪 **Evaluation** with accuracy, precision, recall
* 🧹 **Input padding** and token-to-index mapping

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/cleversupport.git
cd cleversupport
```

### 2. Install Dependencies

```bash
pip install numpy pandas torch torchmetrics scikit-learn nltk
```

### 3. Download NLTK Data

```python
import nltk
nltk.download('punkt')
```

### 4. Run the Notebook

Open `notebook.ipynb` in Jupyter and execute the cells in order.

---

## 🔍 Model Highlights

* **Sequence Length**: Fixed padding/truncation to 50 tokens
* **Embedding + Dense Layers**: Model learns word representations and classifies complaints
* **Custom Metrics**: Powered by `torchmetrics` for detailed evaluation

---

## 🧪 Sample Code Snippet

```python
def pad_input(sentences, seq_len):
    features = np.zeros((len(sentences), seq_len), dtype=int)
    for i, sentence in enumerate(sentences):
        if len(sentence) != 0:
            features[i, -len(sentence):] = np.array(sentence)[:seq_len]
    return features
```


