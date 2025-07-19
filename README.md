# 🕵️‍♂️ PhishDetector

![PhishDetector Banner](https://img.icons8.com/ios-filled/100/000000/phishing.png)

> **A Machine Learning-powered web app to detect phishing websites from URLs and page features.**

---

## 🖤 Features

- ⚡ **Fast**: Real-time phishing detection via a Flask API
- 🤖 **Smart**: Decision Tree model trained on real phishing and legitimate website data
- 🔍 **Feature Extraction**: Analyzes both URL structure and HTML content
- 📊 **Visuals**: Model evaluation, feature importance, and confusion matrix plots
- 🛡️ **Customizable**: Easily adjust detection threshold for stricter or looser detection

---

## 🚀 Quickstart

### 1. **Clone the Repository**

```bash
git clone https://github.com/yourusername/phishDetector.git
cd phishDetector
```

### 2. **Install Requirements**

```bash
pip install -r requirements.txt
```

### 3. **Train the Model (Optional)**

If you want to retrain the model:

```bash
python phishML.py
```

This will generate:
- `phish_detector_model.pkl`
- `feature_list.pkl`
- Evaluation plots in the project folder

### 4. **Run the Flask App**

```bash
python app.py
```

The app will be available at [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## 🖥️ How to Use

### **Web UI**

1. Open your browser and go to [http://127.0.0.1:5000](http://127.0.0.1:5000)
2. Enter a website URL (e.g., `https://www.microsoft.com` or `http://paypal-login-verification.com`)
3. Click **Check** or **Submit**
4. See the prediction:  
   - 🟢 **Legitimate**  
   - 🔴 **Phishing**

### **API Usage**

Send a POST request to `/predict` with JSON:

```json
{
  "url": "https://suspicious-site.com"
}
```

**Example using `curl`:**

```bash
curl -X POST -H "Content-Type: application/json" \
     -d '{"url": "http://paypal-login-verification.com"}' \
     http://127.0.0.1:5000/predict
```

**Response:**
```json
{
  "prediction": 1,
  "probability": 0.87,
  "features": { ... }
}
```
- `prediction: 1` = Phishing
- `prediction: 0` = Legitimate

---

## ⚙️ Configuration

- **Detection Threshold:**  
  Change the `threshold` value in `app.py` to make detection stricter or looser.

---

## 📝 Notes

- For best results, ensure your feature extraction matches the training data.
- This app is for educational/demo purposes. For production, use HTTPS and a production WSGI server.

---

## 🧑‍💻 Contributing

Pull requests and issues are welcome!

---

## 📄 License

MIT License

---

<div align="center">
  <img src="https://img.icons8.com/ios-filled/50/000000/phishing.png" width="40"/>
  <br>
  <b>Stay safe online!</b>
