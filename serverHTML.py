from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import pandas as pd
import re
from urllib.parse import urlparse, urljoin
from datetime import datetime
import os
import requests
from bs4 import BeautifulSoup
from PIL import Image
from io import BytesIO
import base64

app = Flask(__name__)
CORS(app)

# Add section to compare to Obvious working websites (reference a txt file list)
def load_obvious_websites(file_path):
    try:
        with open(file_path, 'r') as file:
            websites = {line.strip().lower() for line in file if line.strip()}
        print(f"[DEBUG] Loaded obvious websites: {websites}")
        return websites
    except Exception as e:
        print(f"[DEBUG] Error loading obvious websites: {e}")
        return set()

# CREATE FILE WITH LIST OF ALL VALID OBVIOUS WEBSITES


# Load model and artifacts ONCE with debug
try:
    print("[DEBUG] Loading model and preprocessing artifacts...")
    model = joblib.load('best_rf_model.pkl')
    scaler = joblib.load('scaler.pkl')
    top_k_features = joblib.load('top_features.pkl')
    training_columns = joblib.load('training_columns.pkl')
    model_loaded = True
    print("[DEBUG] Model/artifacts loaded successfully.")
    print("[DEBUG] Model type: {}".format(type(model)))
    print("[DEBUG] top_k_features: {}".format(top_k_features))
    print("[DEBUG] training_columns: {}".format(training_columns))
except Exception as e:
    print(f"[DEBUG] Error loading model/artifacts: {e}")
    model = scaler = top_k_features = training_columns = None
    model_loaded = False

TRUSTED_DOMAINS = {
    "microsoft.com","google.com","apple.com","amazon.com",
    "paypal.com","github.com","gov","edu"
}

# ----- constants unchanged -----

SUSPICIOUS_KEYWORDS = [
    "secure","account","update","confirm","verify","login","signin",
    "bank","paypal","amazon","microsoft","apple","google","facebook",
    "twitter","urgent","suspended","limited","restricted","temporary",
    "expire","click","free","winner","congratulations","prize","offer","deal"
]

URL_SHORTENERS = [
    "bit.ly","tinyurl.com","t.co","goo.gl","ow.ly","short.link",
    "tiny.cc","lnkd.in","buff.ly","ift.tt","is.gd","v.gd"
]

SUSPICIOUS_TLDS = [
    ".tk",".ml",".ga",".cf",".pw",".top",".click",".download",
    ".stream",".science",".work",".party",".review"
]

LEGIT_TLDS = ['.com','.org','.net','.edu','.gov']

class HTMLImageParser:
    def __init__(self, timeout=10, max_image_size=5*1024*1024):
        self.timeout = timeout
        self.max_image_size = max_image_size
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'Mozilla/5.0 ... Chrome'})

    def fetch_html_content(self, url):
        try:
            if not url.startswith(('http://', 'https://')):
                url = 'http://' + url
            print(f"[DEBUG] Fetching HTML content for: {url}")
            response = self.session.get(url, timeout=self.timeout, allow_redirects=True)
            response.raise_for_status()
            content_type = response.headers.get('content-type', '').lower()
            print(f"[DEBUG] Response content-type: {content_type}")
            if 'text/html' not in content_type:
                print("[DEBUG] Content is not HTML, skipping.")
                return None, None
            print("[DEBUG] HTML fetched successfully.")
            return response.text, response.url
        except requests.exceptions.RequestException as e:
            print(f"[DEBUG] Failed to fetch HTML: {str(e)}")
            return None, None

    def parse_images_from_html(self, html_content, base_url):
        print("[DEBUG] Parsing images from HTML.")
        if not html_content:
            print("[DEBUG] No HTML content to parse for images.")
            return {'NoOfImage': 0,'total_image_size': 0,'external_images': 0,'broken_images': 0,'suspicious_images': 0,'image_details': []}
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            images = soup.find_all('img')
            print(f"[DEBUG] Found {len(images)} images in HTML.")
            image_stats = {'NoOfImage': len(images),'total_image_size': 0,'external_images': 0,'broken_images': 0,'suspicious_images': 0,'image_details': []}
            for i, img in enumerate(images):
                img_data = self.analyze_single_image(img, base_url, i)
                image_stats['image_details'].append(img_data)
                if img_data['is_external']:
                    image_stats['external_images'] += 1
                if img_data['is_broken']:
                    image_stats['broken_images'] += 1
                if img_data['is_suspicious']:
                    image_stats['suspicious_images'] += 1
                if img_data['size']:
                    image_stats['total_image_size'] += img_data['size']
            print(f"[DEBUG] Image stats: {image_stats}")
            return image_stats
        except Exception as e:
            print(f"[DEBUG] Error parsing images: {str(e)}")
            return {'NoOfImage': 0,'total_image_size': 0,'external_images': 0,'broken_images': 0,'suspicious_images': 0,'image_details': []}

    def analyze_single_image(self, img_tag, base_url, index):
        img_data = {'index': index,'src': img_tag.get('src', ''),'alt': img_tag.get('alt', ''),'title': img_tag.get('title', ''),'width': img_tag.get('width'),'height': img_tag.get('height'),'is_external': False,'is_broken': False,'is_suspicious': False,'size': 0,'format': None,'absolute_url': ''}
        src = img_data['src']
        if not src:
            img_data['is_broken'] = True
            return img_data
        if src.startswith('data:'):
            img_data['format'] = 'data_url'
            img_data['size'] = len(src)
            return img_data
        elif src.startswith('//'):
            parsed_base = urlparse(base_url)
            img_data['absolute_url'] = f"{parsed_base.scheme}:{src}"
        elif src.startswith(('http://', 'https://')):
            img_data['absolute_url'] = src
        else:
            img_data['absolute_url'] = urljoin(base_url, src)
        base_domain = urlparse(base_url).netloc
        img_domain = urlparse(img_data['absolute_url']).netloc
        img_data['is_external'] = base_domain != img_domain
        img_data['is_suspicious'] = self.is_suspicious_image(img_data)
        if img_data['absolute_url']:
            img_data.update(self.get_image_metadata(img_data['absolute_url']))
        return img_data

    def is_suspicious_image(self, img_data):
        suspicious_patterns = [r'1x1\.gif',r'pixel\.gif',r'transparent\.gif',r'spacer\.gif',r'blank\.gif',r'clear\.gif',r'invisible\.png',r'track',r'analytics',r'beacon']
        src = img_data['src'].lower()
        alt = img_data['alt'].lower()
        for pattern in suspicious_patterns:
            if re.search(pattern, src) or re.search(pattern, alt):
                return True
        try:
            width = int(img_data['width']) if img_data['width'] else 0
            height = int(img_data['height']) if img_data['height'] else 0
            if width <= 1 and height <= 1:
                return True
        except (ValueError, TypeError):
            pass
        return False

    def get_image_metadata(self, img_url):
        metadata = {'size': 0,'format': None,'actual_width': None,'actual_height': None,'is_broken': False}
        try:
            head_response = self.session.head(img_url, timeout=5)
            if head_response.status_code == 200:
                content_length = head_response.headers.get('content-length')
                if content_length:
                    metadata['size'] = int(content_length)
                if metadata['size'] > self.max_image_size:
                    print(f"[DEBUG] Image {img_url} too large ({metadata['size']} bytes). Skip fetch.")
                    return metadata
            img_response = self.session.get(img_url, timeout=5, stream=True)
            if img_response.status_code == 200:
                chunk = next(img_response.iter_content(chunk_size=1024), b'')
                if chunk:
                    try:
                        img = Image.open(BytesIO(chunk))
                        metadata['format'] = img.format
                        metadata['actual_width'], metadata['actual_height'] = img.size
                    except Exception:
                        content_type = img_response.headers.get('content-type', '')
                        if 'image/' in content_type:
                            metadata['format'] = content_type.split('/')[-1]
                        else:
                            ext = img_url.split('.')[-1].lower()
                            if ext in ['jpg','jpeg','png','gif','webp','svg']:
                                metadata['format'] = ext
        except Exception as e:
            print(f"[DEBUG] Couldn't fetch image metadata for {img_url}: {str(e)}")
            metadata['is_broken'] = True
        return metadata

def is_trusted_domain(url):
    domain = urlparse(url).netloc.lower()
    base_domain = '.'.join(domain.split('.')[-2:])
    return any(trusted in base_domain for trusted in TRUSTED_DOMAINS)

def is_ip_address(domain):
    ipv4_pattern = r'^((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?){1,3}\.){3}(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?){1}$'
    ipv6_pattern = r'^(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$'
    return bool(re.match(ipv4_pattern, domain) or bool(re.match(ipv6_pattern, domain)))

def extract_html_features(html_content, base_url):
    print("[DEBUG] Extracting HTML features...")
    if not html_content:
        print("[DEBUG] No HTML content provided for extraction.")
        return {'LineOfCode': 0, 'LargestLineLength': 0, 'HasTitle': 0,'DomainTitleMatchScore': 0, 'URLTitleMatchScore': 0, 'HasFavicon': 0, 'Robots': 0, 'IsResponsive': 0, 'NoOfURLRedirect': 0,'NoOfSelfRedirect': 0,'HasDescription': 0,'NoOfPopup': 0,'NoOfiFrame': 0,'HasExternalFormSubmit': 0,'HasSubmitButton': 0,'HasHiddenFields': 0,'HasPasswordField': 0,'NoOfImage': 0,'NoOfCSS': 0,'NoOfJS': 0,'NoOfSelfRef': 0,'NoOfEmptyRef': 0,'NoOfExternalRef': 0}
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        base_domain = urlparse(base_url).netloc
        lines = html_content.split('\n')
        line_count = len(lines)
        largest_line = max(len(line) for line in lines) if lines else 0
        title_tag = soup.find('title')
        has_title = 1 if title_tag and title_tag.get_text().strip() else 0
        title_text = title_tag.get_text().strip().lower() if title_tag else ""
        domain_title_match = 0
        url_title_match = 0
        if title_text:
            domain_words = base_domain.lower().replace('.', ' ').split()
            url_words = base_url.lower().replace('/', ' ').replace('.', ' ').split()
            domain_matches = sum(1 for word in domain_words if word in title_text)
            url_matches = sum(1 for word in url_words if word in title_text)
            domain_title_match = domain_matches / max(len(domain_words), 1)
            url_title_match = url_matches / max(len(url_words), 1)
        has_description = 1 if soup.find('meta', attrs={'name': 'description'}) else 0
        has_favicon = 1 if soup.find('link', rel='icon') or soup.find('link', rel='shortcut icon') else 0
        robots_meta = 1 if soup.find('meta', attrs={'name': 'robots'}) else 0
        viewport_meta = soup.find('meta', attrs={'name': 'viewport'})
        is_responsive = 1 if viewport_meta else 0
        forms = soup.find_all('form')
        external_form_submit = 0
        has_submit_button = 0
        has_hidden_fields = 0
        has_password_field = 0
        for form in forms:
            action = form.get('action', '')
            if action and not action.startswith(('#', '/', '?')):
                form_domain = urlparse(urljoin(base_url, action)).netloc
                if form_domain != base_domain:
                    external_form_submit = 1
            if form.find('input', type='submit') or form.find('button', type='submit'):
                has_submit_button = 1
            if form.find('input', type='hidden'):
                has_hidden_fields = 1
            if form.find('input', type='password'):
                has_password_field = 1
        iframes = len(soup.find_all('iframe'))
        css_links = len(soup.find_all('link', rel='stylesheet'))
        js_scripts = len(soup.find_all('script'))
        links = soup.find_all('a', href=True)
        self_refs = 0
        empty_refs = 0
        external_refs = 0
        for link in links:
            href = link['href'].strip()
            if not href or href == '#':
                empty_refs += 1
            elif href.startswith(('http://', 'https://')):
                link_domain = urlparse(href).netloc
                if link_domain == base_domain:
                    self_refs += 1
                else:
                    external_refs += 1
            else:
                self_refs += 1
        popup_indicators = ['window.open','popup','alert(','confirm(']
        popup_count = sum(html_content.lower().count(indicator) for indicator in popup_indicators)
        return {'LineOfCode': line_count,'LargestLineLength': largest_line,'HasTitle': has_title,'DomainTitleMatchScore': domain_title_match,'URLTitleMatchScore': url_title_match,'HasFavicon': has_favicon,'Robots': robots_meta,'IsResponsive': is_responsive,'NoOfURLRedirect': 0,'NoOfSelfRedirect': 0,'HasDescription': has_description,'NoOfPopup': popup_count,'NoOfiFrame': iframes,'HasExternalFormSubmit': external_form_submit,'HasSubmitButton': has_submit_button,'HasHiddenFields': has_hidden_fields,'HasPasswordField': has_password_field,'NoOfCSS': css_links,'NoOfJS': js_scripts,'NoOfSelfRef': self_refs,'NoOfEmptyRef': empty_refs,'NoOfExternalRef': external_refs}
    except Exception as e:
        print(f"[DEBUG] Error extracting HTML features: {str(e)}")
        return {'LineOfCode': 0,'LargestLineLength': 0,'HasTitle': 0,'DomainTitleMatchScore': 0,'URLTitleMatchScore': 0,
                'HasFavicon': 0,'Robots': 0,'IsResponsive':0,'NoOfURLRedirect':0,'NoOfSelfRedirect':0,'HasDescription':0,
                'NoOfPopup':0,'NoOfiFrame':0,'HasExternalFormSubmit':0,'HasSubmitButton':0,'HasHiddenFields':0,
                'HasPasswordField':0,'NoOfImage':0,'NoOfCSS':0,'NoOfJS':0,'NoOfSelfRef':0,'NoOfEmptyRef':0,
                'NoOfExternalRef':0}

def extract_url_features(input_url):
    print(f"[DEBUG] Extracting URL features for: {input_url}")
    normalized_url = input_url if input_url.startswith(('http://', 'https://')) else 'http://' + input_url
    parsed = urlparse(normalized_url)
    domain = parsed.netloc
    full_url = parsed.geturl().lower()
    url_length = len(full_url)
    digit_count = sum(c.isdigit() for c in full_url)
    letter_count = sum(c.isalpha() for c in full_url)
    special_char_count = len(re.findall(r'[^a-zA-Z0-9\-._~:/?#[\]@!$&\'()*+,;=]', full_url))
    domain_parts = domain.split('.')
    tld = domain_parts[-1] if domain_parts else ""
    tld_length = len(tld)
    valid_chars = re.findall(r'[a-zA-Z0-9\-._~:/?#[\]@!$&\'()*+,;=]', full_url)
    url_char_prob = len(valid_chars) / max(url_length, 1)
    tld_with_dot = f".{tld}"
    if tld_with_dot in LEGIT_TLDS:
        tld_legitimate_prob = 1.0
    elif tld_with_dot in SUSPICIOUS_TLDS:
        tld_legitimate_prob = 0.1
    else:
        tld_legitimate_prob = 0.5
    feature_dict = {
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
        'Bank': 1 if 'bank' in full_url else 0,
        'Pay': 1 if 'pay' in full_url else 0,
        'Crypto': 1 if 'crypto' in full_url else 0,
    }
    print(f"[DEBUG] Extracted URL features: {feature_dict}")
    return feature_dict

def analyze_url_comprehensive(input_url, fetch_html=True):
    print(f"[DEBUG] Starting comprehensive analysis for: {input_url}")
    url_features = extract_url_features(input_url)
    html_features = {'LineOfCode': 0,'LargestLineLength': 0,'HasTitle': 0,'DomainTitleMatchScore': 0,'URLTitleMatchScore': 0,'HasFavicon': 0,'Robots': 0,'IsResponsive': 0,'NoOfURLRedirect': 0,'NoOfSelfRedirect': 0,'HasDescription': 0,'NoOfPopup': 0,'NoOfiFrame': 0,'HasExternalFormSubmit': 0,'HasSubmitButton': 0,'HasHiddenFields': 0,'HasPasswordField': 0,'NoOfImage': 0,'NoOfCSS': 0,'NoOfJS': 0,'NoOfSelfRef': 0,'NoOfEmptyRef': 0,'NoOfExternalRef': 0}
    image_stats = None
    if fetch_html:
        print(f"[DEBUG] Attempting HTML fetch/parse: fetch_html={fetch_html}")
        parser = HTMLImageParser()
        html_content, final_url = parser.fetch_html_content(input_url)
        if html_content:
            print("[DEBUG] HTML fetched, extracting HTML features.")
            html_features = extract_html_features(html_content, final_url or input_url)
            image_stats = parser.parse_images_from_html(html_content, final_url or input_url)
            html_features['NoOfImage'] = image_stats['NoOfImage']
        else:
            print("[DEBUG] Unable to fetch HTML, falling back to URL-only features.")
    features = {**url_features, **html_features}
    features['IsTrustedDomain'] = 1 if is_trusted_domain(input_url) else 0
    print(f"[DEBUG] Combined features for prediction: {features}")
    result = {
        'url': input_url,
        'features': features,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'html_analyzed': fetch_html and image_stats is not None
    }
    if image_stats:
        print("[DEBUG] Including image analysis in result.")
        result['image_analysis'] = image_stats
    return result

def preprocess_features_for_prediction(features_dict):
    print("[DEBUG] Preprocessing features for model input...")
    print(f"[DEBUG] Input features dict keys: {list(features_dict.keys())}")
    df = pd.DataFrame([features_dict])
    print(f"[DEBUG] Initial DataFrame columns: {list(df.columns)}")
    if training_columns is not None:
        for col in training_columns:
            if col not in df.columns:
                print(f"[DEBUG] Missing column '{col}'. Filling with 0.")
                df[col] = 0
        df = df[training_columns]
        print(f"[DEBUG] DataFrame after column alignment: {list(df.columns)}")
    else:
        print("[DEBUG] No training_columns loaded!")
    if scaler is not None:
        print("[DEBUG] Scaling DataFrame features.")
        scaled_data = scaler.transform(df)
        df_scaled = pd.DataFrame(scaled_data, columns=training_columns)
        print(f"[DEBUG] Scaled DataFrame:\n{df_scaled}")
    else:
        print("[DEBUG] No scaler loaded, skipping scaling.")
        df_scaled = df
    if top_k_features is not None:
        print(f"[DEBUG] Selecting top_k_features: {top_k_features}")
        df_scaled = df_scaled[top_k_features]
        print(f"[DEBUG] DataFrame for model input:\n{df_scaled}")
    else:
        print("[DEBUG] No top_k_features loaded! Using all columns.")
    return df_scaled

def calculate_fallback_score(features):
    print(f"[DEBUG] Running fallback scoring for features: {features}")
    score = 0
    score += min(features['URLLength'] / 100, 0.3) * 20
    score += features['IsDomainIP'] * 25
    score += min(features['NoOfSubDomain'] / 5, 0.2) * 15
    if features['IsHTTPS']:
        score -= 15
    else:
        score += 20
    score += features['HasObfuscation'] * 15
    if features.get('IsTrustedDomain', 0) == 1:
        score -= 30
    if features['LineOfCode'] > 0:
        score += (1 - features['HasTitle']) * 10
        score += features['HasExternalFormSubmit'] * 20
        score += min(features['NoOfPopup'] / 3, 1) * 15
        score += min(features['NoOfiFrame'] / 2, 1) * 10
        score += (1 - features['HasFavicon']) * 5
        if features['NoOfImage'] == 0:
            score += 10
        elif features['NoOfImage'] > 50:
            score += 15
    print(f"[DEBUG] Fallback score: {score}")
    return max(0, min(int(score), 100))

@app.route('/api/predict', methods=['POST'])
def predict():
    print("[DEBUG] /api/predict route entered")
    data = request.get_json()
    print("[DEBUG] Received request: {}".format(data))
    if not data or 'url' not in data:
        print("[DEBUG] Missing URL in request data.")
        return jsonify({'error': 'URL is required'}), 400
    url_in = data['url'].strip()
    print(f"[DEBUG] User input URL: {url_in}")
    if not url_in:
        print("[DEBUG] Input URL is blank.")
        return jsonify({'error': 'URL is required'}), 400
    fetch_html = data.get('fetch_html', True)
    print(f"[DEBUG] fetch_html param: {fetch_html}")
    try:
        result = analyze_url_comprehensive(url_in, fetch_html=fetch_html)
        features = result['features']
        print(f"[DEBUG] Features extracted for prediction: {features}")
        if model_loaded:
            print("[DEBUG] ML model loaded. Preparing features for prediction.")
            try:
                processed_features = preprocess_features_for_prediction(features)
                print(f"[DEBUG] Model input DataFrame:\n{processed_features}")
                prediction = int(model.predict(processed_features)[0])
                print(f"[DEBUG] ML prediction: {prediction}")
                probability = float(model.predict_proba(processed_features)[0][1])
                print(f"[DEBUG] ML probability (legitimate): {probability}")
                risk_score = int(probability * 100)
                print(f"[DEBUG] Risk score from model: {risk_score}")
            except Exception as exc:
                print(f"[DEBUG] Error during model prediction: {exc}")
                print("[DEBUG] Using fallback scoring instead.")
                risk_score = calculate_fallback_score(features)
                prediction = 0 if risk_score >= 50 else 1
        else:
            print("[DEBUG] ML model NOT loaded. Using fallback scoring.")
            risk_score = calculate_fallback_score(features)
            prediction = 0 if risk_score >= 50 else 1
        risk_level = "LOW" if risk_score <= 30 else "MEDIUM" if risk_score <= 60 else "HIGH"
        print(f"[DEBUG] Final prediction: {prediction} | risk_level: {risk_level}")
        response_data = {
            'prediction': prediction,
            'risk_score': risk_score,
            'risk_level': risk_level,
            'features': features,
            'url': result['url'],
            'timestamp': result['timestamp'],
            'model_loaded': model_loaded,
            'html_analyzed': result['html_analyzed']
        }
        if 'image_analysis' in result:
            print("[DEBUG] Adding image analysis to API response.")
            response_data['image_analysis'] = result['image_analysis']
        print(f"[DEBUG] Returning response: {response_data}")
        return jsonify(response_data)
    except Exception as e:
        print(f"[DEBUG] Top-level error in prediction endpoint: {str(e)}")
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

@app.route('/api/analyze-images', methods=['POST'])
def analyze_images_only():
    print("[DEBUG] /api/analyze-images route entered")
    data = request.get_json()
    print(f"[DEBUG] Request data: {data}")
    if not data or 'url' not in data:
        print("[DEBUG] Missing URL in image analysis request.")
        return jsonify({'error': 'URL is required'}), 400
    url_in = data['url'].strip()
    print(f"[DEBUG] User input URL: {url_in}")
    if not url_in:
        print("[DEBUG] Input URL is blank.")
        return jsonify({'error': 'URL is required'}), 400
    try:
        parser = HTMLImageParser()
        print("[DEBUG] Fetching HTML for image analysis...")
        html_content, final_url = parser.fetch_html_content(url_in)
        if not html_content:
            print("[DEBUG] Could not fetch HTML for image analysis.")
            return jsonify({'error': 'Could not fetch HTML content'}), 400
        print("[DEBUG] Parsing images in HTML...")
        image_stats = parser.parse_images_from_html(html_content, final_url or url_in)
        print(f"[DEBUG] Returning image analysis: {image_stats}")
        return jsonify({
            'url': url_in,
            'final_url': final_url,
            'image_analysis': image_stats,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
    except Exception as e:
        print(f"[DEBUG] Error in analyze-images endpoint: {str(e)}")
        return jsonify({'error': f'Image analysis failed: {str(e)}'}), 500

@app.route('/')
def index():
    print("[DEBUG] index route called")
    current_dir = os.path.abspath(os.path.dirname(__file__))
    print(f"[DEBUG] Serving index.html from {current_dir}")
    return send_from_directory(current_dir, 'index.html')

if __name__ == '__main__':
    print("[DEBUG] Starting Phishing Detector")
    print("[DEBUG] 🌐 Starting Flask server...")
    app.run(host='0.0.0.0', port=5000, debug=True)
