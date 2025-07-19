# Phishing Email Detector

A smart, Python-powered project by **Brendan Lambrecht** and **Zach Ydunate** for identifying phishing emails using machine learning. The goal of this repository is to help individuals and organizations protect their inboxes from malicious email attacks by flagging and classifying suspicious messages.

## 📝 Description

**Phishing Email Detector** is a software tool designed to detect and classify phishing emails with high accuracy. Leveraging state-of-the-art machine learning techniques and natural language processing (NLP), this project analyzes the content and metadata of emails to distinguish between legitimate and potentially harmful messages.

- **Easy to Use**: Simple command-line interface for scanning emails locally.
- **Customizable**: Train and evaluate your own models with your dataset.
- **Educational**: Designed for students and developers interested in cybersecurity, AI, and NLP.

## 🚀 Features

- Detects phishing and legitimate emails with machine learning models
- Preprocessing and feature extraction from email content and headers
- Supports multiple algorithms (Logistic Regression, Random Forest, etc.)
- Clear output with detailed report for each scanned email
- Easily extendable with new features or classifiers

## ⚙️ Installation

```bash
git clone https://github.com/your-username/phishing-email-detector.git
cd phishing-email-detector
pip install -r requirements.txt
```

## 🛠️ Usage

1. **Scan Email File**

   ```bash
   python detector.py --file your_email.eml
   ```

2. **Batch Scan Directory**

   ```bash
   python detector.py --dir email_directory/
   ```

3. **Train Model**

   ```bash
   python train.py --dataset path/to/dataset.csv
   ```

## 📊 Dataset

The project supports any CSV-formatted dataset with columns for email text and labels (`phishing`, `legitimate`). Sample datasets and data preparation scripts are included in the repository.

## 🧑‍💻 Model Training

You can train your own model with our scripts:

- Preprocessing and vectorization using NLP
- Training with several classifiers
- Evaluation and accuracy metrics
- Model saving and loading for future use

Modify `config.yaml` to experiment with different parameters.

## 🤝 Contributing

Contributions are welcome! Open an issue or submit a pull request.

- Fork and clone this repository
- Create a new branch for your feature or bugfix
- Submit a detailed pull request

## 📃 License

This project is licensed under the MIT License.

## 📧 Contact

**Brendan Lambrecht** & **Zach Ydunate**  
For questions, suggestions, or collaborations:  
Blambrecht04@gmail.com
`Zach Contact Info Here`
