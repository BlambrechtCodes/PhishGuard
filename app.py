from flask import Flask, request, jsonify, send_from_directory
import joblib
import pandas as pd
from flask_cors import CORS
from urllib.parse import urlparse

app = Flask(__name__)
CORS(app)

# Load your trained model and feature list
model = joblib.load('phish_detector_model.pkl')
feature_list = joblib.load('feature_list.pkl')

def extract_features(url):
    parsed = urlparse(url if url.startswith('http') else 'http://' + url)
    features = {}

    # Basic features you can extract from the URL
    features['URLLength'] = len(url)
    features['DomainLength'] = len(parsed.netloc)
    features['IsDomainIP'] = 1 if parsed.netloc.replace('.', '').isdigit() else 0

    # The rest of the features are set to 0 or a sensible default
    features['CharContinuationRate'] = 0
    features['TLDLegitimateProb'] = 0
    features['URLCharProb'] = 0
    features['TLDLength'] = len(parsed.netloc.split('.')[-1]) if '.' in parsed.netloc else 0
    features['NoOfSubDomain'] = parsed.netloc.count('.') - 1 if '.' in parsed.netloc else 0
    features['HasObfuscation'] = 0
    features['NoOfObfuscatedChar'] = 0
    features['ObfuscationRatio'] = 0
    features['NoOfLettersInURL'] = sum(c.isalpha() for c in url)
    features['LetterRatioInURL'] = features['NoOfLettersInURL'] / len(url) if len(url) > 0 else 0
    features['NoOfDegitsInURL'] = sum(c.isdigit() for c in url)
    features['DegitRatioInURL'] = features['NoOfDegitsInURL'] / len(url) if len(url) > 0 else 0
    features['NoOfEqualsInURL'] = url.count('=')
    features['NoOfQMarkInURL'] = url.count('?')
    features['NoOfAmpersandInURL'] = url.count('&')
    features['NoOfOtherSpecialCharsInURL'] = sum(not c.isalnum() for c in url)
    features['LineOfCode'] = 0
    features['LargestLineLength'] = 0
    features['HasTitle'] = 0
    features['HasFavicon'] = 0
    features['Robots'] = 0
    features['NoOfURLRedirect'] = 0
    features['NoOfSelfRedirect'] = 0
    features['NoOfPopup'] = 0
    features['NoOfiFrame'] = 0
    features['HasExternalFormSubmit'] = 0
    features['HasPasswordField'] = 0
    features['Bank'] = 0
    features['Pay'] = 0
    features['Crypto'] = 0
    features['NoOfImage'] = 0
    features['NoOfCSS'] = 0
    features['NoOfJS'] = 0
    features['NoOfSelfRef'] = 0
    features['NoOfEmptyRef'] = 0
    features['NoOfExternalRef'] = 0

    # Remove 'label' if present
    if 'label' in features:
        del features['label']

    # Ensure all features in feature_list (except 'label') are present
    for f in feature_list:
        if f == 'label':
            continue
        if f not in features:
            features[f] = 0

    return features

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    url = data.get('url', '')
    features = extract_features(url)
    X = pd.DataFrame([features])
    pred = int(model.predict(X)[0])
    return jsonify({'prediction': pred})

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/<path:path>')
def static_proxy(path):
    return send_from_directory('.', path)

if __name__ == '__main__':
    app.run(debug=True)