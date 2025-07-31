from flask import Flask, request, jsonify, send_from_directory
import joblib
import pandas as pd
import numpy as np
import tensorflow as tf
from flask_cors import CORS
from urllib.parse import urlparse, urljoin
import requests
from bs4 import BeautifulSoup
import re
from datetime import datetime
from warnings import simplefilter
import logging
import traceback
import sys
import tldextract
import threading


# Suppress warnings
simplefilter("ignore", category=UserWarning)


app = Flask(__name__)
CORS(app)


# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('phish_detector.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('PhishDetector')


logger.info("===== Starting PhishDetector Application =====")


# Thread lock for TensorFlow prediction
tf_lock = threading.Lock()


# Load models, scaler, and feature list
try:
    logger.info("Loading TensorFlow model and Decision Tree...")
    # Load TensorFlow model (primary)
    tf_model = tf.keras.models.load_model('phishing_detector_tensorflow_model.h5')
    # Load Decision Tree model (backup/ensemble)  
    dt_model = joblib.load('phishing_detector_decision_tree.pkl')
    scaler = joblib.load('phishing_detector_scaler.pkl')
    feature_columns = joblib.load('phishing_detector_features.pkl')
    model_loaded = True
    logger.info(f"Models loaded successfully with {len(feature_columns)} features")
except Exception as e:
    logger.critical(f"Failed to load models: {e}")
    logger.error(traceback.format_exc())
    model_loaded = False
    tf_model = None
    dt_model = None
    scaler = None
    feature_columns = []


def normalize_url(url: str) -> str:
    """Ensure URL has a scheme; default to https."""
    parsed = urlparse(url)
    if not parsed.scheme:
        logger.debug(f"URL missing scheme, prepending https:// to {url}")
        return 'https://' + url
    return url


def is_ip_address(domain):
    ipv4_pattern = r'^(\d{1,3}\.){3}\d{1,3}$'
    ipv6_pattern = r'^([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$'
    return bool(re.match(ipv4_pattern, domain) or re.match(ipv6_pattern, domain))


def contains_keywords(url, keywords):
    return any(keyword in url.lower() for keyword in keywords)


def continuation_rate(url):
    if len(url) < 2:
        return 0.0
    repeats = sum(1 for i in range(1, len(url)) if url[i] == url[i-1])
    return repeats / (len(url) - 1)


def tld_legitimacy(root_domain):
    legit_tlds = ['.com', '.org', '.net', '.edu', '.gov']
    suspicious_tlds = ['.tk', '.ml', '.ga', '.cf', '.pw']
    tld = '.' + root_domain.split('.')[-1] if '.' in root_domain else ''
    if tld in legit_tlds:
        return 0.9
    elif tld in suspicious_tlds:
        return 0.1
    else:
        return 0.5


def url_character_prob(url):
    unusual_chars = ['@', '~', '%', '!', '*']
    if not url:
        return 0.5
    count = sum(url.count(c) for c in unusual_chars)
    return 1 - (count / len(url))


def extract_url_features(url):
    """Extract all features exactly as done during training."""
    url = normalize_url(url)
    logger.debug(f"Normalized URL: {url}")

    start_time = datetime.now()
    parsed = urlparse(url)
    tld_meta = tldextract.extract(url)
    domain = parsed.netloc
    root_domain = f"{tld_meta.domain}.{tld_meta.suffix}" if tld_meta.suffix else tld_meta.domain
    query = parsed.query

    # Initialize HTML features with default values
    html_features = {
        'HasObfuscation': 0,
        'NoOfObfuscatedChar': 0,
        'ObfuscationRatio': 0,
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
        'NoOfImage': 0,
        'NoOfCSS': 0,
        'NoOfJS': 0,
        'NoOfSelfRef': 0,
        'NoOfEmptyRef': 0,
        'NoOfExternalRef': 0
    }

    try:
        fetch_start = datetime.now()
        response = requests.get(url, timeout=10, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        response.raise_for_status()

        html_content = response.text
        soup = BeautifulSoup(html_content, 'html.parser')
        logger.debug(f"HTML fetched and parsed in {(datetime.now() - fetch_start).total_seconds():.2f} seconds")

        # Extract HTML features exactly as in training
        lines = html_content.splitlines()
        html_features['LineOfCode'] = len(lines)
        html_features['LargestLineLength'] = max((len(line) for line in lines), default=0)

        # Title analysis
        title = soup.title.string.strip() if soup.title and soup.title.string else None
        html_features['HasTitle'] = int(bool(title))
        if title:
            domain_words = set(re.findall(r'\w+', tld_meta.domain.lower()))
            title_words = set(re.findall(r'\w+', title.lower()))
            if domain_words:
                html_features['DomainTitleMatchScore'] = len(domain_words & title_words) / len(domain_words)
            url_words = set(re.findall(r'\w+', url.lower()))
            if url_words:
                html_features['URLTitleMatchScore'] = len(url_words & title_words) / len(url_words)

        # Other HTML features
        html_features['HasFavicon'] = int(bool(soup.find('link', rel=lambda x: x and 'icon' in x.lower())))
        html_features['IsResponsive'] = int(bool(soup.find('meta', attrs={'name': 'viewport'})))
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        html_features['HasDescription'] = int(bool(meta_desc and meta_desc.get('content', '').strip()))
        html_features['NoOfiFrame'] = len(soup.find_all('iframe'))

        # Form analysis
        for form in soup.find_all('form'):
            if form.find(attrs={'type': 'submit'}):
                html_features['HasSubmitButton'] = 1
            if form.find(attrs={'type': 'hidden'}):
                html_features['HasHiddenFields'] = 1
            if form.find(attrs={'type': 'password'}):
                html_features['HasPasswordField'] = 1
            action = form.get('action')
            if action:
                action_url = urljoin(url, action)
                action_tld = tldextract.extract(action_url)
                action_domain = f"{action_tld.domain}.{action_tld.suffix}" if action_tld.suffix else action_tld.domain
                if action_domain.lower() != root_domain.lower():
                    html_features['HasExternalFormSubmit'] = 1

        # Asset counts
        html_features['NoOfImage'] = len(soup.find_all('img'))
        html_features['NoOfCSS'] = len(soup.find_all('link', rel='stylesheet')) + len(soup.find_all('style'))
        html_features['NoOfJS'] = len(soup.find_all('script'))

        # Link analysis
        for link in soup.find_all('a'):
            href = link.get('href', '')
            if not href or href.startswith(('#', 'javascript:')):
                html_features['NoOfEmptyRef'] += 1
            else:
                try:
                    abs_url = urlparse(urljoin(url, href))
                    link_tld = tldextract.extract(abs_url.netloc)
                    link_domain = f"{link_tld.domain}.{link_tld.suffix}" if link_tld.suffix else link_tld.domain
                    if link_domain.lower() == root_domain.lower():
                        html_features['NoOfSelfRef'] += 1
                    else:
                        html_features['NoOfExternalRef'] += 1
                except Exception:
                    html_features['NoOfExternalRef'] += 1

        # Obfuscation detection
        suspicious_encodings = re.findall(r'(%[0-9a-fA-F]{2}|\\x[0-9a-fA-F]{2}|&#x?[0-9a-fA-F]+;)', html_content)
        count_obf = len(suspicious_encodings)
        html_features['NoOfObfuscatedChar'] = count_obf
        html_features['HasObfuscation'] = int(count_obf > 0)
        html_features['ObfuscationRatio'] = count_obf / len(html_content) if html_content else 0

        # robots.txt check
        try:
            robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
            resp_robots = requests.get(robots_url, timeout=5)
            html_features['Robots'] = int(resp_robots.status_code == 200)
        except Exception:
            pass

        # Redirect info
        html_features['NoOfURLRedirect'] = len(response.history)
        html_features['NoOfSelfRedirect'] = sum(
            1 for r in response.history if tldextract.extract(urlparse(r.url).netloc).domain.lower() == tld_meta.domain.lower()
        )

        # Popup detection
        html_features['NoOfPopup'] = html_content.lower().count('window.open')

        logger.info("HTML analysis completed successfully")

    except Exception as e:
        logger.warning(f"HTML analysis failed for {url}: {e}")

    # Basic URL features - MUST match training exactly
    digits = sum(c.isdigit() for c in url)
    letters = sum(c.isalpha() for c in url)
    special_chars = len(url) - digits - letters - url.count(' ')

    subdomains = tld_meta.subdomain.split('.') if tld_meta.subdomain else []
    num_subdomains = len([sd for sd in subdomains if sd])

    features = {
        'URLLength': len(url),
        'DomainLength': len(domain),
        'IsDomainIP': int(is_ip_address(domain)),
        'NoOfSubDomain': num_subdomains,
        'HasHTTPS': int(parsed.scheme == 'https'),
        'HasAtSymbol': int('@' in url),
        'HasRedirecting': int('//' in url[7:]),
        'HasHyphen': int('-' in domain),
        'HasQuestionMark': int('?' in url),
        'HasEqualSign': int('=' in query),
        'HasUnderscore': int('_' in domain),
        'HasTilde': int('~' in url),
        'HasPercent': int('%' in url),
        'HasSlash': url.count('/'),
        'HasDot': url.count('.'),
        'HasHTMLTag': int(html_features['LineOfCode'] > 0),
        'Bank': int(contains_keywords(url, ['bank', 'paypal', 'chase'])),
        'Crypto': int(contains_keywords(url, ['crypto', 'bitcoin', 'blockchain'])),
        'Pay': int(contains_keywords(url, ['pay', 'payment', 'checkout'])),
        'DegitRatioInURL': digits / len(url) if len(url) > 0 else 0,
        'CharContinuationRate': continuation_rate(url),
        'TLDLegitimateProb': tld_legitimacy(root_domain),
        'URLCharProb': url_character_prob(url),
        'TLDLength': len(tld_meta.suffix) if tld_meta.suffix else 0,
        'NoOfLettersInURL': letters,
        'LetterRatioInURL': letters / len(url) if len(url) > 0 else 0,
        'NoOfDegitsInURL': digits,
        'NoOfEqualsInURL': url.count('='),
        'NoOfQMarkInURL': url.count('?'),
        'NoOfAmpersandInURL': url.count('&'),
        'IsHTTPS': int(parsed.scheme == 'https'),
        'NoOfOtherSpecialCharsInURL': special_chars,
        'SpacialCharRatioInURL': special_chars / len(url) if len(url) > 0 else 0,
    }

    # Merge HTML features
    features.update(html_features)

    # Ensure all expected features are present, fill missing with zero
    for col in feature_columns:
        if col not in features:
            features[col] = 0
            logger.warning(f"Feature '{col}' missing from extraction; defaulting to 0")

    logger.info(f"Feature extraction completed in {(datetime.now() - start_time).total_seconds():.3f}s")
    return features


@app.route('/api/predict', methods=['POST'])
def predict():
    start_time = datetime.now()
    logger.info("===== New Prediction Request =====")
    try:
        data = request.get_json(force=True)
        raw_url = data.get('url', '')
        if not raw_url or not isinstance(raw_url, str) or not raw_url.strip():
            logger.warning("Empty or invalid URL received.")
            return jsonify({'error': 'URL is required and must be a non-empty string'}), 400

        url = normalize_url(raw_url.strip())
        logger.info(f"Analyzing URL: {url}")

        if not (model_loaded and tf_model and dt_model and scaler):
            logger.error("Models not loaded.")
            return jsonify({'error': 'Models not loaded'}), 500

        features = extract_url_features(url)
        feature_df = pd.DataFrame([features], columns=feature_columns)

        # Scale features for TensorFlow model
        scaled_features = scaler.transform(feature_df)

        # Validate input shape matches model expected shape
        if scaled_features.shape[1] != tf_model.input_shape[1]:
            error_msg = f"Input features length ({scaled_features.shape[1]}) does not match model expected ({tf_model.input_shape[1]})"
            logger.error(error_msg)
            return jsonify({'error': error_msg}), 400

        with tf_lock:
            tf_proba = tf_model.predict(scaled_features, verbose=0)[0][0]

        dt_proba = dt_model.predict_proba(feature_df)[0][1]

        ensemble_proba = 0.7 * tf_proba + 0.3 * dt_proba
        prediction = int(ensemble_proba > 0.5)

        confidence_score = max(ensemble_proba, 1 - ensemble_proba) * 100

        risk_score = int(ensemble_proba * 100)
        if risk_score < 20:
            risk_level = "LOW"
        elif risk_score < 60:
            risk_level = "MEDIUM"
        else:
            risk_level = "HIGH"

        warnings = []
        if features.get('IsDomainIP', 0):
            warnings.append("URL uses an IP address instead of a domain name.")
        if not features.get('HasHTTPS', 1):
            warnings.append("URL does not use HTTPS encryption.")
        if features.get('TLDLegitimateProb', 1) < 0.5:
            warnings.append("Suspicious top-level domain detected.")
        if features.get('Bank', 0):
            warnings.append("Bank-related keywords detected in URL.")
        if features.get('Crypto', 0):
            warnings.append("Cryptocurrency-related keywords detected in URL.")

        response_data = {
            'url': url,
            'is_phishing': bool(prediction),
            'risk_score': risk_score,
            'risk_level': risk_level,
            'probability': float(ensemble_proba),
            'confidence': float(confidence_score),
            'tensorflow_prob': float(tf_proba),
            'decision_tree_prob': float(dt_proba),
            'ensemble_prob': float(ensemble_proba),
            'features': features,
            'warnings': warnings,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        logger.info(f"Prediction completed in {(datetime.now() - start_time).total_seconds():.3f}s")
        return jsonify(response_data)

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'error': f'Analysis failed: {e}'}), 500


@app.route('/api/health', methods=['GET'])
def health():
    status = {
        'status': 'healthy',
        'tensorflow_loaded': tf_model is not None,
        'decision_tree_loaded': dt_model is not None,
        'scaler_loaded': scaler is not None,
        'features_loaded': len(feature_columns) > 0,
        'timestamp': datetime.now().isoformat()
    }
    return jsonify(status)


@app.route('/')
def index():
    return send_from_directory('.', 'index.html')


if __name__ == '__main__':
    if not model_loaded:
        logger.critical("Models not loaded, cannot start server.")
        sys.exit(1)

    logger.info("Starting Flask application on http://127.0.0.1:5000")
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
