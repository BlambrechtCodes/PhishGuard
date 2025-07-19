from flask import Flask, request, jsonify, send_from_directory
import joblib
import pandas as pd
from flask_cors import CORS
from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup

app = Flask(__name__)
CORS(app)

# Load your trained model and feature list
model = joblib.load('phish_detector_model.pkl')
feature_list = joblib.load('feature_list.pkl')

def extract_features(url):
    parsed = urlparse(url if url.startswith('http') else 'http://' + url)
    features = {}

    # URL-based features
    features['URLLength'] = len(url)
    features['DomainLength'] = len(parsed.netloc)
    features['IsDomainIP'] = 1 if parsed.netloc.replace('.', '').isdigit() else 0
    features['TLDLength'] = len(parsed.netloc.split('.')[-1]) if '.' in parsed.netloc else 0
    features['NoOfSubDomain'] = parsed.netloc.count('.') - 1 if '.' in parsed.netloc else 0
    features['NoOfLettersInURL'] = sum(c.isalpha() for c in url)
    features['LetterRatioInURL'] = features['NoOfLettersInURL'] / len(url) if len(url) > 0 else 0
    features['NoOfDegitsInURL'] = sum(c.isdigit() for c in url)
    features['DegitRatioInURL'] = features['NoOfDegitsInURL'] / len(url) if len(url) > 0 else 0
    features['NoOfEqualsInURL'] = url.count('=')
    features['NoOfQMarkInURL'] = url.count('?')
    features['NoOfAmpersandInURL'] = url.count('&')
    features['NoOfOtherSpecialCharsInURL'] = sum(not c.isalnum() for c in url)

    # Defaults for features that require HTML
    features['HasTitle'] = 0
    features['HasFavicon'] = 0
    features['HasPasswordField'] = 0
    features['HasExternalFormSubmit'] = 0
    features['NoOfImage'] = 0
    features['NoOfCSS'] = 0
    features['NoOfJS'] = 0

    # Try to fetch the page and parse HTML-based features
    try:
        resp = requests.get(url, timeout=5)
        html = resp.text
        soup = BeautifulSoup(html, 'html.parser')

        # Title
        features['HasTitle'] = 1 if soup.title and soup.title.string else 0

        # Favicon
        features['HasFavicon'] = 1 if soup.find('link', rel=lambda x: x and 'icon' in x.lower()) else 0

        # Password field
        features['HasPasswordField'] = 1 if soup.find('input', {'type': 'password'}) else 0

        # External form submit
        forms = soup.find_all('form')
        features['HasExternalFormSubmit'] = 0
        for form in forms:
            action = form.get('action', '')
            if action and not action.startswith('/') and parsed.netloc not in action:
                features['HasExternalFormSubmit'] = 1
                break

        # Images, CSS, JS counts
        features['NoOfImage'] = len(soup.find_all('img'))
        features['NoOfCSS'] = len(soup.find_all('link', rel='stylesheet'))
        features['NoOfJS'] = len(soup.find_all('script'))

    except Exception as e:
        # If fetch fails, leave HTML-based features as default (0)
        pass

    # Fill in any missing features with 0
    for f in feature_list:
        if f not in features:
            features[f] = 0

    return features

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    url = data.get('url', '')
    features = extract_features(url)
    # Ensure DataFrame columns match feature_list exactly
    X = pd.DataFrame([[features[f] for f in feature_list]], columns=feature_list)
    proba = model.predict_proba(X)[0][1]  # Probability of phishing
    threshold = 0.3  # Lower = stricter
    pred = int(proba > threshold)
    return jsonify({'prediction': pred, 'probability': proba, 'features': features})

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/<path:path>')
def static_proxy(path):
    return send_from_directory('.', path)

if __name__ == '__main__':
    app.run(debug=True)