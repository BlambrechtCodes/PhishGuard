from flask import Flask, request, jsonify, send_from_directory
import joblib
import pandas as pd
from flask_cors import CORS
from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup
import re
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Load your machine learning model and scaler
try:
    model = joblib.load('phishing_detector_decision_tree.pkl')
    scaler = joblib.load('phishing_detector_scaler.pkl')
    feature_columns = joblib.load('phishing_detector_features.pkl')
    model_loaded = True
except Exception as e:
    print(f"Failed to load model: {e}")
    model_loaded = False

def extract_url_features(url):
    """Extract features that match the trained model's expectations"""
    parsed = urlparse(url)
    domain = parsed.netloc
    path = parsed.path
    query = parsed.query
    
    features = {
        # Basic URL features
        'URLLength': len(url),
        'DomainLength': len(domain),
        'IsDomainIP': 1 if is_ip_address(domain) else 0,
        
        # URL structure features
        'NoOfSubDomain': len(domain.split('.')) - 2,
        'HasHTTPS': 1 if parsed.scheme == 'https' else 0,
        'HasAtSymbol': 1 if '@' in url else 0,
        'HasRedirecting': 1 if '//' in url[7:] else 0,  # After http:// or https://
        
        # Character-based features
        'HasHyphen': 1 if '-' in domain else 0,
        'HasQuestionMark': 1 if '?' in url else 0,
        'HasEqualSign': 1 if '=' in query else 0,
        'HasUnderscore': 1 if '_' in domain else 0,
        'HasTilde': 1 if '~' in url else 0,
        'HasPercent': 1 if '%' in url else 0,
        'HasSlash': url.count('/'),
        'HasDot': url.count('.'),
        
        # Content features (from HTML analysis)
        'HasHTMLTag': 1 if has_html_tags(url) else 0,
        
        # Keyword-based features
        'Bank': 1 if contains_keywords(url, ['bank', 'paypal', 'chase']) else 0,
        'Crypto': 1 if contains_keywords(url, ['crypto', 'bitcoin', 'blockchain']) else 0,
        
        # Advanced features
        'DegitRatioInURL': digit_ratio(url),
        'CharContinuationRate': continuation_rate(url),
        'TLDLegitimateProb': tld_legitimacy(domain),
        'URLCharProb': url_character_prob(url),
        'TLDLength': len(domain.split('.')[-1]) if domain else 0
    }
    
    # Ensure all expected features are present (fill missing with 0)
    for col in feature_columns:
        if col not in features:
            features[col] = 0
    
    return features

# Helper functions for feature extraction
def is_ip_address(domain):
    """Check if domain is an IP address"""
    ipv4_pattern = r'^(\d{1,3}\.){3}\d{1,3}$'
    ipv6_pattern = r'^([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$'
    return bool(re.match(ipv4_pattern, domain) or re.match(ipv6_pattern, domain))

def has_html_tags(url):
    """Check if URL contains HTML tags (indicates direct HTML content)"""
    try:
        response = requests.get(url, timeout=5)
        return bool(BeautifulSoup(response.text, 'html.parser').find())
    except:
        return False

def contains_keywords(url, keywords):
    """Check if URL contains any of the given keywords"""
    return any(keyword in url.lower() for keyword in keywords)

def digit_ratio(url):
    """Calculate ratio of digits to total characters"""
    digits = sum(c.isdigit() for c in url)
    return digits / len(url) if url else 0

def continuation_rate(url):
    """Calculate character continuation rate (repeated characters)"""
    if len(url) < 2:
        return 0
    repeats = sum(1 for i in range(1, len(url)) if url[i] == url[i-1])
    return repeats / (len(url) - 1)

def tld_legitimacy(domain):
    """Estimate TLD legitimacy probability"""
    legit_tlds = ['.com', '.org', '.net', '.edu', '.gov']
    suspicious_tlds = ['.tk', '.ml', '.ga', '.cf', '.pw']
    
    tld = '.' + domain.split('.')[-1] if domain else ''
    if tld in legit_tlds:
        return 0.9
    elif tld in suspicious_tlds:
        return 0.1
    return 0.5

def url_character_prob(url):
    """Estimate character distribution probability"""
    # This is a simplified version - could be enhanced with actual character frequency analysis
    unusual_chars = ['@', '~', '%', '!', '*']
    return 1 - (sum(url.count(c) for c in unusual_chars) / len(url)) if url else 0.5


###! VERY IMPORTANT CONTENT BELOW IS THE PREDICTION ENDPOINT !!!!

###! THIS IS THE PREDICTION ENDPOINT ###
@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        url = data.get('url', '').strip()
        
        if not url:
            return jsonify({'error': 'URL is required'}), 400
        
        # Extract features in the format the model expects
        features = extract_url_features(url)
        
        # Create DataFrame with features in correct order
        feature_df = pd.DataFrame([features])[feature_columns]
        
        # Scale features
        scaled_features = scaler.transform(feature_df)
        
        # Make prediction
        prediction = model.predict(scaled_features)[0]
        probability = model.predict_proba(scaled_features)[0][1]
        
        # Determine risk level
        risk_score = int(probability * 100)
        if risk_score < 30:
            risk_level = "LOW"
        elif risk_score < 60:
            risk_level = "MEDIUM"
        else:
            risk_level = "HIGH"
        
        return jsonify({
            'url': url,
            'is_phishing': bool(prediction),
            'risk_score': risk_score,
            'risk_level': risk_level,
            'probability': probability,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        
    except Exception as e:
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

# ... (keep the rest of your server code the same)

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'detector_ready': True,
        'timestamp': datetime.now().isoformat(),
        'model_loaded': model_loaded
    })

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4200, debug=True)