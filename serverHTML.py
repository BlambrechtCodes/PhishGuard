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
    # Print expected features for debugging
    if hasattr(model, 'feature_names_in_'):
        print("Model expects features:", model.feature_names_in_)
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

LEGIT_TLDS = ['.com', '.org', '.net', '.edu', '.gov']

def is_ip_address(domain):
    ipv4_pattern = r'^(\d{1,3}\.){3}\d{1,3}$'
    ipv6_pattern = r'^([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$'
    return bool(re.match(ipv4_pattern, domain) or re.match(ipv6_pattern, domain))

def count_suspicious_keywords(url):
    count = 0
    for keyword in SUSPICIOUS_KEYWORDS:
        count += url.count(keyword)
    return count

def extract_url_features(input_url):
    # Normalize URL
    normalized_url = input_url if input_url.startswith(('http://', 'https://')) else 'http://' + input_url
    parsed = urlparse(normalized_url)
    domain = parsed.netloc
    full_url = parsed.geturl().lower()
    url_length = len(full_url)
    
    # Calculate basic character counts
    digit_count = sum(c.isdigit() for c in full_url)
    letter_count = sum(c.isalpha() for c in full_url)
    special_char_count = len(re.findall(r'[^a-zA-Z0-9\-._~:/?#[\]@!$&\'()*+,;=]', full_url))
    
    # Extract TLD (last part of domain)
    domain_parts = domain.split('.')
    tld = domain_parts[-1] if domain_parts else ""
    tld_length = len(tld)
    
    # Calculate URLCharProb - ratio of valid characters in URL
    valid_chars = re.findall(r'[a-zA-Z0-9\-._~:/?#[\]@!$&\'()*+,;=]', full_url)
    url_char_prob = len(valid_chars) / max(url_length, 1)
    
    # Calculate TLD Legitimacy Probability
    tld_with_dot = f".{tld}"
    if tld_with_dot in LEGIT_TLDS:
        tld_legitimate_prob = 0.9
    elif tld_with_dot in SUSPICIOUS_TLDS:
        tld_legitimate_prob = 0.1
    else:
        tld_legitimate_prob = 0.5
    
    # Initialize features with default values for HTML-based features (not available in URL analysis)
    features = {
        'URLLength': url_length,
        'DomainLength': len(domain),
        'IsDomainIP': 1 if is_ip_address(domain) else 0,
        'CharContinuationRate': special_char_count / max(url_length, 1),
        'TLDLegitimateProb': tld_legitimate_prob,
        'URLCharProb': url_char_prob,
        'TLDLength': tld_length,
        'NoOfSubDomain': max(len(domain_parts) - 2, 0) if domain else 0,
        'HasObfuscation': 1 if '%' in full_url else 0,
        'NoOfObfuscatedChar': full_url.count('%'),
        'ObfuscationRatio': full_url.count('%') / max(url_length, 1),
        'NoOfLettersInURL': letter_count,
        'LetterRatioInURL': letter_count / max(url_length, 1),
        'NoOfDegitsInURL': digit_count,
        'DegitRatioInURL': digit_count / max(url_length, 1),
        'NoOfEqualsInURL': full_url.count('='),
        'NoOfQMarkInURL': full_url.count('?'),
        'NoOfAmpersandInURL': full_url.count('&'),
        'NoOfOtherSpecialCharsInURL': special_char_count,
        'SpacialCharRatioInURL': special_char_count / max(url_length, 1),
        'IsHTTPS': 1 if parsed.scheme == 'https' else 0,
        # HTML-based features (set to 0 since we can't extract them from URL)
        'LineOfCode': 0,
        'LargestLineLength': 0,
        'HasTitle': 0,
        'DomainTitleMatchScore': 0,
        'URLTitleMatchScore': 0,
        'HasFavicon': 0,
        'Robots': 0,
        'IsResponsive': 0,
        'NoOfURLRedirect': 0,
        'NoOfSelfRedirect': 0,
        'HasDescription': 0,
        'NoOfPopup': 0,
        'NoOfiFrame': 0,
        'HasExternalFormSubmit': 0,
        'HasSubmitButton': 0,
        'HasHiddenFields': 0,
        'HasPasswordField': 0,
        'Bank': 1 if 'bank' in full_url else 0,
        'Pay': 1 if 'pay' in full_url else 0,
        'Crypto': 1 if 'crypto' in full_url else 0,
        'NoOfImage': 0,
        'NoOfCSS': 0,
        'NoOfJS': 0,
        'NoOfSelfRef': 0,
        'NoOfEmptyRef': 0,
        'NoOfExternalRef': 0
    }
    
    return features

def analyze_url_features(input_url):
    # Extract all possible features
    features = extract_url_features(input_url)
    
    # Create DataFrame with features in EXACT order expected by the model
    feature_columns = [
        'URLLength', 'DomainLength', 'IsDomainIP', 'CharContinuationRate',
        'TLDLegitimateProb', 'URLCharProb', 'TLDLength', 'NoOfSubDomain',
        'HasObfuscation', 'NoOfObfuscatedChar', 'ObfuscationRatio',
        'NoOfLettersInURL', 'LetterRatioInURL', 'NoOfDegitsInURL', 'DegitRatioInURL',
        'NoOfEqualsInURL', 'NoOfQMarkInURL', 'NoOfAmpersandInURL',
        'NoOfOtherSpecialCharsInURL', 'SpacialCharRatioInURL', 'IsHTTPS',
        'LineOfCode', 'LargestLineLength', 'HasTitle', 'DomainTitleMatchScore',
        'URLTitleMatchScore', 'HasFavicon', 'Robots', 'IsResponsive',
        'NoOfURLRedirect', 'NoOfSelfRedirect', 'HasDescription', 'NoOfPopup',
        'NoOfiFrame', 'HasExternalFormSubmit', 'HasSubmitButton', 'HasHiddenFields',
        'HasPasswordField', 'Bank', 'Pay', 'Crypto', 'NoOfImage', 'NoOfCSS', 'NoOfJS',
        'NoOfSelfRef', 'NoOfEmptyRef', 'NoOfExternalRef'
    ]
    
    # Feature validation
    missing = set(feature_columns) - set(features.keys())
    extra = set(features.keys()) - set(feature_columns)
    
    if missing:
        print(f"WARNING: Missing features: {missing}")
    if extra:
        print(f"WARNING: Extra features: {extra}")
    
    # Create DataFrame with features in exact order
    feature_df = pd.DataFrame([features])[feature_columns]
    
    if model_loaded:
        try:
            # Use ML model for prediction
            prediction = model.predict(feature_df)[0]
            probability = model.predict_proba(feature_df)[0][1]
            is_phishing = bool(prediction)
            risk_score = int(probability * 100)
            
            # Print exact ML confidence
            print(f"✅ ML Model Prediction: {probability*100:.1f}% confident of phishing")
        except Exception as exc:
            # Fallback to rule-based scoring
            risk_score = (
                0.3 * features['URLLength'] / 100 +
                0.4 * features['IsDomainIP'] +
                0.2 * features['NoOfSubDomain'] / 5 +
                0.1 * (1 - features['IsHTTPS'])
            ) * 100
            risk_score = min(risk_score, 100)
            is_phishing = risk_score >= 50
            
            # Debug print for fallback due to model error
            print(f"⚠️ Model error: {exc}. Using fallback rule-based scoring. Risk score: {risk_score}, Phishing: {is_phishing}")
    else:
        # Fallback if model not loaded
        risk_score = (
            0.3 * features['URLLength'] / 100 +
            0.4 * features['IsDomainIP'] +
            0.2 * features['NoOfSubDomain'] / 5 +
            0.1 * (1 - features['IsHTTPS'])
        ) * 100
        risk_score = min(risk_score, 100)
        is_phishing = risk_score >= 50
        
        # Debug print for model not loaded
        print(f"⚠️ Model not loaded. Using fallback rule-based scoring. Risk score: {risk_score}, Phishing: {is_phishing}")

    risk_level = "LOW" if risk_score <= 30 else "MEDIUM" if risk_score <= 60 else "HIGH"

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
        # Print the URL being analyzed
        print(f"🔍 Analyzing URL: {url}")
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
        print(f"❌ Analysis failed for URL: {url}. Error: {str(e)}")
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

@app.route('/')
def index():
    # Serve your index.html file from the same directory as this script
    current_dir = os.path.abspath(os.path.dirname(__file__))
    return send_from_directory(current_dir, 'index.html')

def test_specific_url():
    """Test function for a specific URL"""
    test_url = "http://www.campbellsautosport.com/InventoryDetails.aspx?id=3590&egSet=3584%7C3600%7C3603%7C3532%7C3589%7C3535%7C3570%7C3599%7C3594%7C3590%7C3602%7C3593%7C3577%7C3561%7C3512%7C3562%7C3598%7C3566%7C3568%7C3605%7C3592%7C3604%7C3596%7C3572%7C3575"
    print(f"\n{'='*50}")
    print(f"TESTING SPECIFIC URL: {test_url}")
    print(f"{'='*50}")
    
    # Extract features
    features = extract_url_features(test_url)
    print("\nExtracted Features:")
    for key, value in features.items():
        print(f"{key}: {value}")
    
    # Analyze URL
    result = analyze_url_features(test_url)
    
    print("\nAnalysis Result:")
    print(f"URL: {result['url']}")
    print(f"Phishing: {result['is_phishing']}")
    print(f"Risk Score: {result['risk_score']}%")
    print(f"Risk Level: {result['risk_level']}")
    print(f"Timestamp: {result['timestamp']}")
    
    # Highlight key features
    print("\nKey Features:")
    key_features = ['URLLength', 'HasObfuscation', 'NoOfQMarkInURL', 
                   'NoOfEqualsInURL', 'IsHTTPS', 'Bank', 'Pay', 'Crypto']
    for feat in key_features:
        print(f"{feat}: {features.get(feat, 'N/A')}")

if __name__ == '__main__':
    # Run the test on the specific URL
    test_specific_url()
    
    # Then start the server
    print("\nStarting server...")
    app.run(host='0.0.0.0', port=5000)