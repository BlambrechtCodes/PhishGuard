from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import pandas as pd
import re
from urllib.parse import urlparse
from datetime import datetime
import os

app = Flask(__name__)
CORS(app)

# Load model once
try:
    model = joblib.load('phishing_detector_decision_tree.pkl')
    model_loaded = True
except Exception as e:
    print(f"Error loading model: {e}")
    model_loaded = False

SUSPICIOUS_KEYWORDS = [
    "secure", "account", "update", "confirm", "verify", "login", "signin", "bank",
    "paypal", "amazon", "microsoft", "apple", "google", "facebook", "twitter",
    "urgent", "suspended", "limited", "restricted", "temporary", "expire",
    "click", "free", "winner", "congratulations", "prize", "offer", "deal"
]

URL_SHORTENERS = [
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "short.link",
    "tiny.cc", "lnkd.in", "buff.ly", "ift.tt", "is.gd", "v.gd"
]

SUSPICIOUS_TLDS = [
    ".tk", ".ml", ".ga", ".cf", ".pw", ".top", ".click", ".download",
    ".stream", ".science", ".work", ".party", ".review"
]

def is_ip_address(domain):
    ipv4_pattern = r'^(\d{1,3}\.){3}\d{1,3}$'
    ipv6_pattern = r'^([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$'
    return bool(re.match(ipv4_pattern, domain) or re.match(ipv6_pattern, domain))

def count_suspicious_keywords(url):
    count = 0
    for keyword in SUSPICIOUS_KEYWORDS:
        count += url.count(keyword)
    return count

def estimate_domain_age(domain):
    if any(tld in domain for tld in SUSPICIOUS_TLDS):
        return 0  # recently registered (numeric feature)
    elif domain.endswith(('.com', '.org', '.net')):
        return 1  # established
    return 2  # unknown

def analyze_url_features(input_url):
    # Normalize URL
    normalized_url = input_url if input_url.startswith(('http://', 'https://')) else 'http://' + input_url
    parsed = urlparse(normalized_url)
    domain = parsed.netloc
    full_url = parsed.geturl()

    features = {
        'url_length': len(full_url),
        'domain_length': len(domain),
        'subdomain_count': max(len(domain.split('.')) - 2, 0) if domain else 0,
        'has_https': 1 if parsed.scheme == 'https' else 0,
        'has_ip_address': 1 if is_ip_address(domain) else 0,
        'suspicious_keywords': count_suspicious_keywords(full_url.lower()),
        'has_url_shortener': int(any(shortener in domain for shortener in URL_SHORTENERS)),
        'special_char_count': len(re.findall(r'[^a-zA-Z0-9\-._~:/?#[\]@!$&\'()*+,;=]', full_url)),
        'domain_age': estimate_domain_age(domain),
        'has_valid_cert': 1 if parsed.scheme == 'https' else 0
    }

    feature_df = pd.DataFrame([features])
    if model_loaded:
        try:
            prediction = model.predict(feature_df)[0]
            probability = model.predict_proba(feature_df)[0][1]
            is_phishing = bool(prediction)
            risk_score = int(probability * 100)
        except Exception as exc:
            print(f"Model prediction error: {exc}")
            # fallback rule-based scoring
            risk_score = 0
            if features['url_length'] > 100:
                risk_score += 15
            if features['has_ip_address']:
                risk_score += 25
            if features['subdomain_count'] > 3:
                risk_score += 20
            if not features['has_https']:
                risk_score += 15
            risk_score = min(risk_score, 100)
            is_phishing = risk_score >= 50
    else:
        # fallback rule-based scoring if model not loaded
        risk_score = 0
        if features['url_length'] > 100:
            risk_score += 15
        if features['has_ip_address']:
            risk_score += 25
        if features['subdomain_count'] > 3:
            risk_score += 20
        if not features['has_https']:
            risk_score += 15
        risk_score = min(risk_score, 100)
        is_phishing = risk_score >= 50

    risk_level = "LOW" if risk_score < 30 else "MEDIUM" if risk_score < 60 else "HIGH"

    return {
        'url': input_url,
        'is_phishing': is_phishing,
        'risk_score': risk_score,
        'risk_level': risk_level,
        'features': features,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'url' not in data:
        return jsonify({'error': 'URL is required'}), 400
    url = data['url'].strip()
    if not url:
        return jsonify({'error': 'URL is required'}), 400

    try:
        result = analyze_url_features(url)
        return jsonify({
            'prediction': 1 if result['is_phishing'] else 0,
            'risk_score': result['risk_score'],
            'risk_level': result['risk_level'],
            'features': result['features'],
            'url': result['url'],
            'timestamp': result['timestamp'],
            'model_loaded': model_loaded
        })
    except Exception as e:
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

@app.route('/')
def index():
    # Serve your index.html file from the same directory as this script
    current_dir = os.path.abspath(os.path.dirname(__file__))
    return send_from_directory(current_dir, 'index.html')

if __name__ == '__main__':
    # Run on all interfaces on port 5000
    app.run(host='0.0.0.0', port=5000)
