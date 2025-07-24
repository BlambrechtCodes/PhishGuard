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

# Load your machine learning model
try:
    model = joblib.load('phish_detector_model.pkl')  # Replace with your model path
    model_loaded = True
except Exception as e:
    print(f"Failed to load model: {e}")
    model_loaded = False

# Suspicious keywords commonly found in phishing URLs
SUSPICIOUS_KEYWORDS = [
    "secure", "account", "update", "confirm", "verify", "login", "signin", "bank",
    "paypal", "amazon", "microsoft", "apple", "google", "facebook", "twitter",
    "urgent", "suspended", "limited", "restricted", "temporary", "expire",
    "click", "free", "winner", "congratulations", "prize", "offer", "deal"
]

# Known URL shorteners
URL_SHORTENERS = [
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "short.link",
    "tiny.cc", "lnkd.in", "buff.ly", "ift.tt", "is.gd", "v.gd"
]

# Suspicious TLDs
SUSPICIOUS_TLDS = [
    ".tk", ".ml", ".ga", ".cf", ".pw", ".top", ".click", ".download",
    ".stream", ".science", ".work", ".party", ".review"
]

def extract_content_features(url):
    """Extract content features from URL using requests and BeautifulSoup"""
    features = {
        'has_title': 0, 'title_length': 0, 'has_favicon': 0, 'form_count': 0,
        'input_count': 0, 'external_links_count': 0, 'internal_links_count': 0,
        'image_count': 0, 'script_count': 0, 'iframe_count': 0
    }
    
    try:
        # Make request with timeout
        response = requests.get(url, timeout=5, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        # Parse HTML content
        soup = BeautifulSoup(response.text, 'html.parser')
        parsed_url = urlparse(url)
        
        # Title analysis
        title = soup.find('title')
        if title and title.string:
            features['has_title'] = 1
            features['title_length'] = len(title.string.strip())
        
        # Favicon check
        favicon_tags = soup.find_all('link', rel=lambda x: x and 'icon' in str(x).lower())
        features['has_favicon'] = 1 if favicon_tags else 0
        
        # Form analysis
        forms = soup.find_all('form')
        features['form_count'] = len(forms)
        
        # Input fields
        inputs = soup.find_all('input')
        features['input_count'] = len(inputs)
        
        # Link analysis
        links = soup.find_all('a', href=True)
        for link in links:
            href = link['href']
            if href.startswith('http'):
                if parsed_url.netloc in href:
                    features['internal_links_count'] += 1
                else:
                    features['external_links_count'] += 1
        
        # Media elements
        features['image_count'] = len(soup.find_all('img'))
        features['script_count'] = len(soup.find_all('script'))
        features['iframe_count'] = len(soup.find_all('iframe'))
        
    except Exception as e:
        print(f"Error extracting content features: {e}")
    
    return features

def analyze_url_features(input_url):
    """Analyze URL and return results with enhanced content analysis"""
    # Normalize URL
    normalized_url = input_url
    if not normalized_url.startswith(('http://', 'https://')):
        normalized_url = 'http://' + normalized_url
    
    try:
        parsed = urlparse(normalized_url)
        domain = parsed.netloc
        full_url = parsed.geturl()
    except:
        raise ValueError("Invalid URL format")
    
    # Extract features
    features = {
        'url_length': len(full_url),
        'domain_length': len(domain),
        'subdomain_count': len(domain.split('.')) - 2 if domain else 0,
        'has_https': parsed.scheme == 'https',
        'has_ip_address': is_ip_address(domain),
        'suspicious_keywords': count_suspicious_keywords(full_url.lower()),
        'has_url_shortener': any(shortener in domain for shortener in URL_SHORTENERS),
        'special_char_count': len(re.findall(r'[^a-zA-Z0-9\-._~:/?#[\]@!$&\'()*+,;=]', full_url)),
        'domain_age': estimate_domain_age(domain),
        'has_valid_cert': parsed.scheme == 'https'
    }
    
    # Extract content features
    content_features = extract_content_features(normalized_url)
    features.update(content_features)
    
    # If model is loaded, use it for prediction
    if model_loaded:
        try:
            # Convert features to DataFrame in the correct order expected by the model
            feature_df = pd.DataFrame([features])
            
            # Make prediction (adjust according to your model's requirements)
            prediction = model.predict(feature_df)[0]
            probability = model.predict_proba(feature_df)[0][1]  # Probability of being phishing
            
            is_phishing = bool(prediction)
            risk_score = int(probability * 100)
        except Exception as e:
            print(f"Model prediction failed: {e}")
            # Fall back to rule-based if model fails
            is_phishing, risk_score = rule_based_analysis(features)
    else:
        # Use rule-based analysis if model isn't loaded
        is_phishing, risk_score = rule_based_analysis(features)
    
    # Determine risk level
    if risk_score < 30:
        risk_level = "LOW"
    elif risk_score < 60:
        risk_level = "MEDIUM"
    else:
        risk_level = "HIGH"
    
    return {
        'url': input_url,
        'is_phishing': is_phishing,
        'risk_score': risk_score,
        'risk_level': risk_level,
        'features': features,
        'warnings': generate_warnings(features, risk_score),
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

def rule_based_analysis(features):
    """Fallback rule-based analysis if model isn't available"""
    risk_score = 0
    
    # Rule-based scoring logic
    if features['url_length'] > 100:
        risk_score += 15
    
    if features['has_ip_address']:
        risk_score += 25
    
    if features['subdomain_count'] > 3:
        risk_score += 20
    
    if not features['has_https']:
        risk_score += 15
    
    if features['suspicious_keywords'] > 0:
        risk_score += min(features['suspicious_keywords'] * 5, 25)
    
    if features['has_url_shortener']:
        risk_score += 20
    
    if features['special_char_count'] > 5:
        risk_score += 15
    
    risk_score = min(risk_score, 100)
    is_phishing = risk_score >= 50
    
    return is_phishing, risk_score

def generate_warnings(features, risk_score):
    """Generate warnings based on features"""
    warnings = []
    
    if features['url_length'] > 100:
        warnings.append("URL is unusually long (potential obfuscation)")
    
    if features['has_ip_address']:
        warnings.append("URL uses IP address instead of domain name")
    
    if features['subdomain_count'] > 3:
        warnings.append(f"URL has {features['subdomain_count']} subdomains (excessive nesting)")
    
    if not features['has_https']:
        warnings.append("URL does not use HTTPS encryption")
    
    if features['suspicious_keywords'] > 0:
        warnings.append(f"URL contains {features['suspicious_keywords']} suspicious keywords")
    
    if features['has_url_shortener']:
        warnings.append("URL uses a known URL shortening service")
    
    if features['special_char_count'] > 5:
        warnings.append(f"URL contains {features['special_char_count']} special characters")
    
    if features['form_count'] > 2:
        warnings.append(f"Page contains {features['form_count']} forms (potential data collection)")
    
    if features['input_count'] > 5:
        warnings.append(f"Page contains {features['input_count']} input fields (potential data collection)")
    
    return warnings

def is_ip_address(domain):
    """Check if domain is an IP address"""
    ipv4_pattern = r'^(\d{1,3}\.){3}\d{1,3}$'
    ipv6_pattern = r'^([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$'
    return bool(re.match(ipv4_pattern, domain) or re.match(ipv6_pattern, domain))

def count_suspicious_keywords(url):
    """Count suspicious keywords in URL"""
    count = 0
    for keyword in SUSPICIOUS_KEYWORDS:
        count += url.count(keyword)
    return count

def estimate_domain_age(domain):
    """Estimate domain age based on TLD"""
    if any(tld in domain for tld in SUSPICIOUS_TLDS):
        return "Recently registered"
    elif domain.endswith(('.com', '.org', '.net')):
        return "Established"
    return "Unknown"

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        url = data.get('url', '').strip()
        
        if not url:
            return jsonify({'error': 'URL is required'}), 400
        
        # Perform analysis
        analysis_result = analyze_url_features(url)
        
        # Extract URL breakdown
        parsed = urlparse(url if url.startswith(('http://', 'https://')) else 'http://' + url)
        url_breakdown = {
            'protocol': parsed.scheme or 'http',
            'domain': parsed.netloc,
            'path': parsed.path or '/',
            'parameters': url.count('='),
            'subdomains': analysis_result['features']['subdomain_count']
        }
        
        return jsonify({
            'prediction': 1 if analysis_result['is_phishing'] else 0,
            'probability': analysis_result['risk_score'] / 100.0,
            'features': analysis_result['features'],
            'url_breakdown': url_breakdown,
            'url': url,
            'analysis_timestamp': analysis_result['timestamp'],
            'model_used': model_loaded,  # Indicates if ML model was used
            'is_phishing': analysis_result['is_phishing'],
            'risk_score': analysis_result['risk_score'],
            'risk_level': analysis_result['risk_level'],
            'warnings': analysis_result['warnings']
        })
        
    except Exception as e:
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

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