# 🕵️‍♂️ PhishGuard™️


  


> **A Machine Learning-powered web app to detect phishing websites from URLs and page features.**

## ✨ Features

- ⚡ **Fast:** Real-time phishing detection via a Flask API
- 🤖 **Smart:** Decision Tree model trained on real phishing and legitimate website data
- 🔍 **Feature Extraction:** Analyzes both URL structure and HTML content
- 📊 **Visuals:** Model evaluation, feature importance, and confusion matrix plots
- 🛡️ **Customizable:** Detection threshold easily adjustable for stricter or looser inspection

## 🚀 Quickstart

**1. Clone the Repository**
```bash
git clone https://github.com/yourusername/phishDetector.git
cd phishDetector
```

**2. Install Requirements**
```bash
pip install -r requirements.txt
```

**3. Train the Model (Optional)**
```bash
python phish1_MAIN.py
```
*Generates:*
- `phish_detector_model.pkl`
- `feature_list.pkl`
- Evaluation plots in the project folder
- Etc.

**4. Run the Flask App**
```bash
python serverHTML.py
```
The app will be available at this address: [http://127.0.0.1:4200](http://127.0.0.1:4200)

## 🖥️ How to Use

### Web UI

1. Open your browser at [http://127.0.0.1:5000](http://127.0.0.1:5000)
2. Enter a website URL (e.g., `https://www.microsoft.com` or `http://paypal-login-verification.com`)
3. Click **Check** or **Submit**
4. View prediction:
   - 🟢 **Legitimate**
   - 🔴 **Phishing**

### API Usage

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
- `prediction: 1` = **Phishing**
- `prediction: 0` = **Legitimate**

## ⚙️ Configuration

- **Detection Threshold:**  
  Adjust the `threshold` value in `app.py` to make detection stricter or looser.

## 📚 Documentation

- [CHANGELOG.md](CHANGELOG.md)
- [CONTRIBUTING.md](CONTRIBUTING.md)
- [DATASET.md](DATASET.md)

## 📝 Notes

- Match your feature extraction to the training data for optimal results.
- *Educational/demo app only.* For production, use HTTPS and a production-grade WSGI server.

## 🧑‍💻 Contributing

- [Submit issues or pull requests!](CONTRIBUTING.md)

## 📄 License

- MIT License
  
  Stay safe online!

