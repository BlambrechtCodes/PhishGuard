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
import time
from PIL import Image
from io import BytesIO
import base64

app = Flask(__name__)
CORS(app)

# Load model once
try:
    model = joblib.load('phishing_detector_decision_tree.pkl')
    model_loaded = True
    if hasattr(model, 'feature_names_in_'):
        print("Model expects features:", model.feature_names_in_)
except Exception as e:
    print(f"Error loading model: {e}")
    model_loaded = False

SUSPICIOUS_KEYWORDS = [
    "secure", "account", "update", "confirm", "verify", "login", "signin",
    "bank", "paypal", "amazon", "microsoft", "apple", "google", "facebook",
    "twitter", "urgent", "suspended", "limited", "restricted", "temporary",
    "expire", "click", "free", "winner", "congratulations", "prize", "offer", "deal"
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

class HTMLImageParser:
    def __init__(self, timeout=10, max_image_size=5*1024*1024):  # 5MB max
        self.timeout = timeout
        self.max_image_size = max_image_size
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })

    def fetch_html_content(self, url):
        """Fetch HTML content from URL with error handling"""
        try:
            # Normalize URL
            if not url.startswith(('http://', 'https://')):
                url = 'http://' + url
            
            print(f"Fetching HTML content from: {url}")
            response = self.session.get(url, timeout=self.timeout, allow_redirects=True)
            response.raise_for_status()
            
            # Check content type
            content_type = response.headers.get('content-type', '').lower()
            if 'text/html' not in content_type:
                print(f"Non-HTML content type: {content_type}")
                return None, None
            
            return response.text, response.url
        except requests.exceptions.RequestException as e:
            print(f"❌ Failed to fetch HTML: {str(e)}")
            return None, None

    def parse_images_from_html(self, html_content, base_url):
        """Parse and analyze images from HTML content"""
        if not html_content:
            return {
                'NoOfImage': 0,
                'total_image_size': 0,
                'external_images': 0,
                'broken_images': 0,
                'suspicious_images': 0,
                'image_details': []
            }

        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            images = soup.find_all('img')
            
            image_stats = {
                'NoOfImage': len(images),
                'total_image_size': 0,
                'external_images': 0,
                'broken_images': 0,
                'suspicious_images': 0,
                'image_details': []
            }

            print(f"🖼️ Found {len(images)} images in HTML")

            for i, img in enumerate(images):
                img_data = self.analyze_single_image(img, base_url, i)
                image_stats['image_details'].append(img_data)
                
                # Update counters
                if img_data['is_external']:
                    image_stats['external_images'] += 1
                if img_data['is_broken']:
                    image_stats['broken_images'] += 1
                if img_data['is_suspicious']:
                    image_stats['suspicious_images'] += 1
                if img_data['size']:
                    image_stats['total_image_size'] += img_data['size']

            return image_stats

        except Exception as e:
            print(f"❌ Error parsing images: {str(e)}")
            return {
                'NoOfImage': 0,
                'total_image_size': 0,
                'external_images': 0,
                'broken_images': 0,
                'suspicious_images': 0,
                'image_details': []
            }

    def analyze_single_image(self, img_tag, base_url, index):
        """Analyze a single image tag"""
        img_data = {
            'index': index,
            'src': img_tag.get('src', ''),
            'alt': img_tag.get('alt', ''),
            'title': img_tag.get('title', ''),
            'width': img_tag.get('width'),
            'height': img_tag.get('height'),
            'is_external': False,
            'is_broken': False,
            'is_suspicious': False,
            'size': 0,
            'format': None,
            'absolute_url': ''
        }

        src = img_data['src']
        if not src:
            img_data['is_broken'] = True
            return img_data

        # Handle different URL formats
        if src.startswith('data:'):
            # Data URL (base64 encoded image)
            img_data['format'] = 'data_url'
            img_data['size'] = len(src)
            return img_data
        elif src.startswith('//'):
            # Protocol-relative URL
            parsed_base = urlparse(base_url)
            img_data['absolute_url'] = f"{parsed_base.scheme}:{src}"
        elif src.startswith(('http://', 'https://')):
            # Absolute URL
            img_data['absolute_url'] = src
        else:
            # Relative URL
            img_data['absolute_url'] = urljoin(base_url, src)

        # Check if image is external
        base_domain = urlparse(base_url).netloc
        img_domain = urlparse(img_data['absolute_url']).netloc
        img_data['is_external'] = base_domain != img_domain

        # Check for suspicious patterns
        img_data['is_suspicious'] = self.is_suspicious_image(img_data)

        # Try to get image metadata
        if img_data['absolute_url']:
            img_data.update(self.get_image_metadata(img_data['absolute_url']))

        return img_data

    def is_suspicious_image(self, img_data):
        """Check if image has suspicious characteristics"""
        suspicious_patterns = [
            r'1x1\.gif',  # Tracking pixels
            r'pixel\.gif',
            r'transparent\.gif',
            r'spacer\.gif',
            r'blank\.gif',
            r'clear\.gif',
            r'invisible\.png',
            r'track',
            r'analytics',
            r'beacon'
        ]

        src = img_data['src'].lower()
        alt = img_data['alt'].lower()
        
        # Check for suspicious patterns in src or alt
        for pattern in suspicious_patterns:
            if re.search(pattern, src) or re.search(pattern, alt):
                return True

        # Check for very small dimensions (likely tracking pixels)
        try:
            width = int(img_data['width']) if img_data['width'] else 0
            height = int(img_data['height']) if img_data['height'] else 0
            if width <= 1 and height <= 1:
                return True
        except (ValueError, TypeError):
            pass

        return False

    def get_image_metadata(self, img_url):
        """Get image metadata (size, format, dimensions)"""
        metadata = {
            'size': 0,
            'format': None,
            'actual_width': None,
            'actual_height': None,
            'is_broken': False
        }

        try:
            # Use HEAD request first to get content length
            head_response = self.session.head(img_url, timeout=5)
            if head_response.status_code == 200:
                content_length = head_response.headers.get('content-length')
                if content_length:
                    metadata['size'] = int(content_length)
                    
                # If image is too large, don't download it
                if metadata['size'] > self.max_image_size:
                    print(f"Image too large ({metadata['size']} bytes): {img_url}")
                    return metadata

            # Get actual image data for format and dimensions
            img_response = self.session.get(img_url, timeout=5, stream=True)
            if img_response.status_code == 200:
                # Read only first chunk to determine format
                chunk = next(img_response.iter_content(chunk_size=1024), b'')
                if chunk:
                    try:
                        # Use PIL to get image info
                        img = Image.open(BytesIO(chunk))
                        metadata['format'] = img.format
                        metadata['actual_width'], metadata['actual_height'] = img.size
                    except Exception:
                        # Fallback: determine format from content type or URL
                        content_type = img_response.headers.get('content-type', '')
                        if 'image/' in content_type:
                            metadata['format'] = content_type.split('/')[-1]
                        else:
                            # Guess from URL extension
                            ext = img_url.split('.')[-1].lower()
                            if ext in ['jpg', 'jpeg', 'png', 'gif', 'webp', 'svg']:
                                metadata['format'] = ext

        except Exception as e:
            print(f"Could not fetch image metadata for {img_url}: {str(e)}")
            metadata['is_broken'] = True

        return metadata

def is_ip_address(domain):
    ipv4_pattern = r'^(\d{1,3}\.){3}\d{1,3}$'
    ipv6_pattern = r'^([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$'
    return bool(re.match(ipv4_pattern, domain) or re.match(ipv6_pattern, domain))

def count_suspicious_keywords(url):
    count = 0
    for keyword in SUSPICIOUS_KEYWORDS:
        count += url.count(keyword)
    return count

def extract_html_features(html_content, base_url):
    """Extract features from HTML content"""
    if not html_content:
        return {
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
        soup = BeautifulSoup(html_content, 'html.parser')
        base_domain = urlparse(base_url).netloc
        
        # Basic HTML metrics
        lines = html_content.split('\n')
        line_count = len(lines)
        largest_line = max(len(line) for line in lines) if lines else 0
        
        # Title analysis
        title_tag = soup.find('title')
        has_title = 1 if title_tag and title_tag.get_text().strip() else 0
        title_text = title_tag.get_text().strip().lower() if title_tag else ""
        
        # Domain-title matching
        domain_title_match = 0
        url_title_match = 0
        if title_text:
            domain_words = base_domain.lower().replace('.', ' ').split()
            url_words = base_url.lower().replace('/', ' ').replace('.', ' ').split()
            
            domain_matches = sum(1 for word in domain_words if word in title_text)
            url_matches = sum(1 for word in url_words if word in title_text)
            
            domain_title_match = domain_matches / max(len(domain_words), 1)
            url_title_match = url_matches / max(len(url_words), 1)

        # Meta tags
        has_description = 1 if soup.find('meta', attrs={'name': 'description'}) else 0
        has_favicon = 1 if soup.find('link', rel='icon') or soup.find('link', rel='shortcut icon') else 0
        robots_meta = 1 if soup.find('meta', attrs={'name': 'robots'}) else 0
        
        # Responsive design indicators
        viewport_meta = soup.find('meta', attrs={'name': 'viewport'})
        is_responsive = 1 if viewport_meta else 0
        
        # Forms analysis
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

        # Count various elements
        iframes = len(soup.find_all('iframe'))
        css_links = len(soup.find_all('link', rel='stylesheet'))
        js_scripts = len(soup.find_all('script'))
        
        # Analyze links
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
                self_refs += 1  # Relative links are self-references

        # Popup indicators (basic heuristics)
        popup_indicators = ['window.open', 'popup', 'alert(', 'confirm(']
        popup_count = sum(html_content.lower().count(indicator) for indicator in popup_indicators)

        return {
            'LineOfCode': line_count,
            'LargestLineLength': largest_line,
            'HasTitle': has_title,
            'DomainTitleMatchScore': domain_title_match,
            'URLTitleMatchScore': url_title_match,
            'HasFavicon': has_favicon,
            'Robots': robots_meta,
            'IsResponsive': is_responsive,
            'NoOfURLRedirect': 0,  # Would need to track redirects during fetch
            'NoOfSelfRedirect': 0,  # Would need to analyze redirect chain
            'HasDescription': has_description,
            'NoOfPopup': popup_count,
            'NoOfiFrame': iframes,
            'HasExternalFormSubmit': external_form_submit,
            'HasSubmitButton': has_submit_button,
            'HasHiddenFields': has_hidden_fields,
            'HasPasswordField': has_password_field,
            'NoOfCSS': css_links,
            'NoOfJS': js_scripts,
            'NoOfSelfRef': self_refs,
            'NoOfEmptyRef': empty_refs,
            'NoOfExternalRef': external_refs
        }

    except Exception as e:
        print(f"❌ Error extracting HTML features: {str(e)}")
        return {
            'LineOfCode': 0, 'LargestLineLength': 0, 'HasTitle': 0,
            'DomainTitleMatchScore': 0, 'URLTitleMatchScore': 0, 'HasFavicon': 0,
            'Robots': 0, 'IsResponsive': 0, 'NoOfURLRedirect': 0,
            'NoOfSelfRedirect': 0, 'HasDescription': 0, 'NoOfPopup': 0,
            'NoOfiFrame': 0, 'HasExternalFormSubmit': 0, 'HasSubmitButton': 0,
            'HasHiddenFields': 0, 'HasPasswordField': 0, 'NoOfImage': 0,
            'NoOfCSS': 0, 'NoOfJS': 0, 'NoOfSelfRef': 0,
            'NoOfEmptyRef': 0, 'NoOfExternalRef': 0
        }

def extract_url_features(input_url):
    """Extract features from URL only"""
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
        tld_legitimate_prob = 0.9
    elif tld_with_dot in SUSPICIOUS_TLDS:
        tld_legitimate_prob = 0.1
    else:
        tld_legitimate_prob = 0.5

    return {
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

def analyze_url_comprehensive(input_url, fetch_html=True):
    """Comprehensive URL analysis with optional HTML parsing"""
    print(f"Starting comprehensive analysis of: {input_url}")
    
    # Extract URL-based features
    url_features = extract_url_features(input_url)
    
    # Initialize HTML features with defaults
    html_features = {
        'LineOfCode': 0, 'LargestLineLength': 0, 'HasTitle': 0,
        'DomainTitleMatchScore': 0, 'URLTitleMatchScore': 0, 'HasFavicon': 0,
        'Robots': 0, 'IsResponsive': 0, 'NoOfURLRedirect': 0,
        'NoOfSelfRedirect': 0, 'HasDescription': 0, 'NoOfPopup': 0,
        'NoOfiFrame': 0, 'HasExternalFormSubmit': 0, 'HasSubmitButton': 0,
        'HasHiddenFields': 0, 'HasPasswordField': 0, 'NoOfImage': 0,
        'NoOfCSS': 0, 'NoOfJS': 0, 'NoOfSelfRef': 0,
        'NoOfEmptyRef': 0, 'NoOfExternalRef': 0
    }
    
    image_stats = None
    
    if fetch_html:
        # Initialize HTML parser
        parser = HTMLImageParser()
        
        # Fetch HTML content
        html_content, final_url = parser.fetch_html_content(input_url)
        
        if html_content:
            print("✅ Successfully fetched HTML content")
            
            # Extract HTML features
            html_features = extract_html_features(html_content, final_url or input_url)
            
            # Parse images
            image_stats = parser.parse_images_from_html(html_content, final_url or input_url)
            html_features['NoOfImage'] = image_stats['NoOfImage']
            
            print(f"HTML Analysis Complete:")
            print(f"   - Lines of code: {html_features['LineOfCode']}")
            print(f"   - Images found: {html_features['NoOfImage']}")
            print(f"   - External images: {image_stats['external_images']}")
            print(f"   - Suspicious images: {image_stats['suspicious_images']}")
            print(f"   - CSS files: {html_features['NoOfCSS']}")
            print(f"   - JS files: {html_features['NoOfJS']}")
        else:
            print("⚠️ Could not fetch HTML content, using URL-only analysis")

    # Combine all features
    features = {**url_features, **html_features}
    
    # Create feature DataFrame in exact order
    feature_columns = [
        'URLLength', 'DomainLength', 'IsDomainIP', 'CharContinuationRate',
        'TLDLegitimateProb', 'URLCharProb', 'TLDLength', 'NoOfSubDomain',
        'HasObfuscation', 'NoOfObfuscatedChar', 'ObfuscationRatio',
        'NoOfLettersInURL', 'LetterRatioInURL', 'NoOfDegitsInURL',
        'DegitRatioInURL', 'NoOfEqualsInURL', 'NoOfQMarkInURL',
        'NoOfAmpersandInURL', 'NoOfOtherSpecialCharsInURL', 'SpacialCharRatioInURL',
        'IsHTTPS', 'LineOfCode', 'LargestLineLength', 'HasTitle',
        'DomainTitleMatchScore', 'URLTitleMatchScore', 'HasFavicon', 'Robots',
        'IsResponsive', 'NoOfURLRedirect', 'NoOfSelfRedirect', 'HasDescription',
        'NoOfPopup', 'NoOfiFrame', 'HasExternalFormSubmit', 'HasSubmitButton',
        'HasHiddenFields', 'HasPasswordField', 'Bank', 'Pay', 'Crypto',
        'NoOfImage', 'NoOfCSS', 'NoOfJS', 'NoOfSelfRef', 'NoOfEmptyRef',
        'NoOfExternalRef'
    ]
    
    feature_df = pd.DataFrame([features])[feature_columns]
    
    # Make prediction
    if model_loaded:
        try:
            prediction = model.predict(feature_df)[0]
            probability = model.predict_proba(feature_df)[0][1]
            is_phishing = bool(prediction)
            risk_score = int(probability * 100)
            print(f"ML Model Prediction: {probability*100:.1f}% confident of phishing")
        except Exception as exc:
            risk_score = calculate_fallback_score(features)
            is_phishing = risk_score >= 50
            print(f"Model error: {exc}. Using fallback scoring. Risk: {risk_score}%")
    else:
        risk_score = calculate_fallback_score(features)
        is_phishing = risk_score >= 50
        print(f"Model not loaded. Using fallback scoring. Risk: {risk_score}%")

    risk_level = "LOW" if risk_score <= 30 else "MEDIUM" if risk_score <= 60 else "HIGH"

    result = {
        'url': input_url,
        'is_phishing': is_phishing,
        'risk_score': risk_score,
        'risk_level': risk_level,
        'features': features,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'html_analyzed': fetch_html and html_content is not None
    }
    
    if image_stats:
        result['image_analysis'] = image_stats
    
    return result

def calculate_fallback_score(features):
    """Calculate fallback risk score using rule-based approach"""
    score = 0
    
    # URL-based scoring
    score += min(features['URLLength'] / 100, 0.3) * 20
    score += features['IsDomainIP'] * 25
    score += min(features['NoOfSubDomain'] / 5, 0.2) * 15
    score += (1 - features['IsHTTPS']) * 10
    score += features['HasObfuscation'] * 15
    
    # HTML-based scoring
    if features['LineOfCode'] > 0:  # HTML was analyzed
        score += (1 - features['HasTitle']) * 10
        score += features['HasExternalFormSubmit'] * 20
        score += min(features['NoOfPopup'] / 3, 1) * 15
        score += min(features['NoOfiFrame'] / 2, 1) * 10
        score += (1 - features['HasFavicon']) * 5
        
        # Image-based scoring
        if features['NoOfImage'] == 0:
            score += 10  # No images might be suspicious
        elif features['NoOfImage'] > 50:
            score += 15  # Too many images might be suspicious
    
    return min(int(score), 100)

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'url' not in data:
        return jsonify({'error': 'URL is required'}), 400

    url = data['url'].strip()
    if not url:
        return jsonify({'error': 'URL is required'}), 400

    # Option to disable HTML fetching for faster analysis
    fetch_html = data.get('fetch_html', True)

    try:
        result = analyze_url_comprehensive(url, fetch_html=fetch_html)
        
        response_data = {
            'prediction': 1 if result['is_phishing'] else 0,
            'risk_score': result['risk_score'],
            'risk_level': result['risk_level'],
            'features': result['features'],
            'url': result['url'],
            'timestamp': result['timestamp'],
            'model_loaded': model_loaded,
            'html_analyzed': result['html_analyzed']
        }
        
        if 'image_analysis' in result:
            response_data['image_analysis'] = result['image_analysis']
        
        return jsonify(response_data)

    except Exception as e:
        print(f"Analysis failed for URL: {url}. Error: {str(e)}")
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

@app.route('/api/analyze-images', methods=['POST'])
def analyze_images_only():
    """Endpoint specifically for image analysis"""
    data = request.get_json()
    if not data or 'url' not in data:
        return jsonify({'error': 'URL is required'}), 400

    url = data['url'].strip()
    if not url:
        return jsonify({'error': 'URL is required'}), 400

    try:
        parser = HTMLImageParser()
        html_content, final_url = parser.fetch_html_content(url)
        
        if not html_content:
            return jsonify({'error': 'Could not fetch HTML content'}), 400
        
        image_stats = parser.parse_images_from_html(html_content, final_url or url)
        
        return jsonify({
            'url': url,
            'final_url': final_url,
            'image_analysis': image_stats,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

    except Exception as e:
        return jsonify({'error': f'Image analysis failed: {str(e)}'}), 500

@app.route('/')
def index():
    current_dir = os.path.abspath(os.path.dirname(__file__))
    return send_from_directory(current_dir, 'index.html')

def test_comprehensive_analysis():
    """Test function with comprehensive analysis"""
    test_urls = [
        "http://www.campbellsautosport.com/InventoryDetails.aspx?id=3590&egSet=3584%7C3600%7C3603%7C3532%7C3589%7C3535%7C3570%7C3599%7C3594%7C3590%7C3602%7C3593%7C3577%7C3561%7C3512%7C3562%7C3598%7C3566%7C3568%7C3605%7C3592%7C3604%7C3596%7C3572%7C3575",
        "https://www.google.com",
        "https://github.com"
    ]
    
    for test_url in test_urls:
        print(f"\n{'='*80}")
        print(f"TESTING: {test_url}")
        print(f"{'='*80}")
        
        result = analyze_url_comprehensive(test_url, fetch_html=True)
        
        print(f"\nANALYSIS RESULTS:")
        print(f"   URL: {result['url']}")
        print(f"   Phishing: {result['is_phishing']}")
        print(f"   Risk Score: {result['risk_score']}%")
        print(f"   Risk Level: {result['risk_level']}")
        print(f"   HTML Analyzed: {result['html_analyzed']}")
        
        if 'image_analysis' in result:
            img_stats = result['image_analysis']
            print(f"\nIMAGE ANALYSIS:")
            print(f"   Total Images: {img_stats['NoOfImage']}")
            print(f"   External Images: {img_stats['external_images']}")
            print(f"   Broken Images: {img_stats['broken_images']}")
            print(f"   Suspicious Images: {img_stats['suspicious_images']}")
            print(f"   Total Size: {img_stats['total_image_size']} bytes")
            
            if img_stats['image_details']:
                print(f"\n Image Details:")
                for img in img_stats['image_details'][:5]:  # Show first 5 images
                    print(f"      {img['index']}: {img['src'][:50]}...")
                    print(f"         External: {img['is_external']}, Suspicious: {img['is_suspicious']}")
                    if img['size']:
                        print(f"         Size: {img['size']} bytes, Format: {img['format']}")

if __name__ == '__main__':
    print("Starting Phishing Detector")
    test_comprehensive_analysis()
    print("\n🌐 Starting Flask server...")
    app.run(host='0.0.0.0', port=5000, debug=True)
