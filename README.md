
# 🧠 K-Nearest Neighbours Classifier (k-NN)

This project implements a simple version of the **k-Nearest Neighbours (k-NN)** classification algorithm using pure Python – no external machine learning libraries like `sklearn`.

It supports command-line arguments, accuracy evaluation, prediction display, and an optional interactive mode.

---

## 🚀 Features

- ✅ Pure Python implementation (no ML libraries)
- ✅ Works with CSV files (semicolon-delimited)
- ✅ Calculates prediction accuracy
- ✅ Displays expected vs predicted labels
- ✅ Interactive mode for classifying new data
- ✅ Optional accuracy plot vs. `k` (with `matplotlib`)

---

## ⚙️ Usage

### 📌 Run accuracy evaluation:
```bash
python main.py <k> <train_file.csv> <test_file.csv>
```


### 💬 Run interactive mode:
```bash
python main.py <k> <train_file.csv> <test_file.csv> --interactive
```
You’ll be prompted to enter a vector (semicolon-separated). Example:
```
Enter vector: 5.1;3.5;1.4;0.2
Predicted class: Iris-setosa
```

---

## 📈 Plot accuracy vs `k`

Inside your code:
```python
knn.plot_accuracy_vs_k(max_k=50)
```

This shows how accuracy changes for different values of `k`.

---

## 📄 CSV Format

- Files must be `;` (semicolon) separated
- The **last column** should be the **class label**
- The **first row** should contain a header

**Example `train.csv`:**
```
SepalLength;SepalWidth;PetalLength;PetalWidth;Species
5.1;3.5;1.4;0.2;Iris-setosa
6.2;3.4;5.4;2.3;Iris-virginica
```

---

## 📂 File Structure

```
.
├── main.py                  # Program entry point (CLI interface)
├── KNearestNeighbours.py    # k-NN classifier implementation
├── train.csv                # Example training dataset
├── test.csv                 # Example test dataset
└── README.md                # Project description
```

---

## 📦 Dependencies

Only the standard library is used, except for optional plotting:

- `argparse` ✅ (built-in)
- `matplotlib` (for plotting)

📌 To install plotting support:
```bash
pip install matplotlib
```

---

## 📚 Example Output

```bash
Accuracy of model: 96.67%
Iris-setosa     Iris-setosa
Iris-versicolor Iris-versicolor
Iris-virginica  Iris-virginica
...
```

With `--interactive`:
```
Enter vector: 6.1;2.8;4.7;1.2
Predicted class: Iris-versicolor
```

---

Made for the PJATK NAI course.

---
