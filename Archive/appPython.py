from flask import Flask, request, jsonify, send_from_directory
import joblib
import pandas as pd
from flask_cors import CORS
from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import re
import threading
import time
from datetime import datetime

class PhishingDetector:
    def __init__(self):
        # Initialize Flask app
        self.app = Flask(__name__)
        CORS(self.app)
        
            # Load your machine learning model
        try:
            self.model = joblib.load('phishing_detector_decision_tree.pkl')  # Replace with your model path
            self.model_loaded = True
        except Exception as e:
            print(f"Failed to load model: {e}")
            self.model_loaded = False
        
        # Suspicious keywords commonly found in phishing URLs
        self.SUSPICIOUS_KEYWORDS = [
            "secure", "account", "update", "confirm", "verify", "login", "signin", "bank",
            "paypal", "amazon", "microsoft", "apple", "google", "facebook", "twitter",
            "urgent", "suspended", "limited", "restricted", "temporary", "expire",
            "click", "free", "winner", "congratulations", "prize", "offer", "deal"
        ]
        
        # Known URL shorteners
        self.URL_SHORTENERS = [
            "bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "short.link",
            "tiny.cc", "lnkd.in", "buff.ly", "ift.tt", "is.gd", "v.gd"
        ]
        
        # Suspicious TLDs
        self.SUSPICIOUS_TLDS = [
            ".tk", ".ml", ".ga", ".cf", ".pw", ".top", ".click", ".download",
            ".stream", ".science", ".work", ".party", ".review"
        ]
        
        # Setup Flask routes
        self.setup_flask_routes()
        
        # Setup GUI
        self.setup_gui()
    
    def setup_flask_routes(self):
        """Setup Flask API routes"""
        @self.app.route('/api/predict', methods=['POST'])
        @self.app.route('/api/predict', methods=['POST'])
    
        def predict():
            try:
                data = request.get_json()
                url = data.get('url', '').strip()
                
                if not url:
                    return jsonify({'error': 'URL is required'}), 400
                
                # Perform analysis
                analysis_result = self.analyze_url_features(url)
                
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
                    'model_used': self.model_loaded  # Indicates if ML model was used
                })
                
            except Exception as e:
                return jsonify({'error': f'Analysis failed: {str(e)}'}), 500
        
        @self.app.route('/api/health', methods=['GET'])
        def health():
            return jsonify({
                'status': 'healthy',
                'detector_ready': True,
                'timestamp': datetime.now().isoformat()
            })
        
        @self.app.route('/')
        def index():
            return '''
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>PhishGuard API</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 40px; background: linear-gradient(135deg, #1e3a8a, #1e40af); color: white; }
                    .container { max-width: 800px; margin: 0 auto; padding: 20px; }
                    .card { background: rgba(30, 64, 175, 0.8); padding: 20px; border-radius: 10px; margin: 20px 0; }
                    h1 { color: #60a5fa; text-align: center; }
                    .endpoint { background: rgba(0, 0, 0, 0.2); padding: 15px; border-radius: 5px; margin: 10px 0; }
                    code { background: rgba(0, 0, 0, 0.3); padding: 2px 5px; border-radius: 3px; }
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>🛡️ PhishGuard API</h1>
                    <div class="card">
                        <h2>Available Endpoints</h2>
                        <div class="endpoint">
                            <h3>POST /api/predict</h3>
                            <p>Analyze a URL for phishing threats</p>
                            <p><strong>Body:</strong> <code>{"url": "https://example.com"}</code></p>
                        </div>
                        <div class="endpoint">
                            <h3>GET /api/health</h3>
                            <p>Check API health status</p>
                        </div>
                    </div>
                    <div class="card">
                        <h2>Usage Example</h2>
                        <pre><code>curl -X POST http://localhost:5000/api/predict \\
     -H "Content-Type: application/json" \\
     -d '{"url": "https://suspicious-site.com"}'</code></pre>
                    </div>
                </div>
            </body>
            </html>
            '''
    
    def setup_gui(self):
        """Setup the main GUI window"""
        self.root = tk.Tk()
        self.root.title("PhishGuard - URL Security Analysis")
        self.root.geometry("900x700")
        self.root.configure(bg='#1e3a8a')  # Blue background
        
        # Configure styles
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure custom styles
        style.configure('Title.TLabel', 
                       font=('Arial', 24, 'bold'),
                       background='#1e3a8a',
                       foreground='#60a5fa')
        
        style.configure('Subtitle.TLabel',
                       font=('Arial', 12),
                       background='#1e3a8a',
                       foreground='#93c5fd')
        
        style.configure('Section.TLabel',
                       font=('Arial', 14, 'bold'),
                       background='#1e40af',
                       foreground='black')
        
        style.configure('Info.TLabel',
                       font=('Arial', 10),
                       background='#1e40af',
                       foreground='black')
        
        style.configure('Custom.TFrame',
                       background='#1e40af',
                       relief='raised',
                       borderwidth=2)
        
        style.configure('Analysis.TButton',
                       font=('Arial', 12, 'bold'),
                       background='#0ea5e9',
                       foreground='black')
        
        self.create_widgets()
    
    def create_widgets(self):
        """Create all GUI widgets"""
        # Main container
        main_frame = ttk.Frame(self.root, style='Custom.TFrame')
        main_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Header
        header_frame = ttk.Frame(main_frame, style='Custom.TFrame')
        header_frame.pack(fill='x', pady=(0, 20))
        
        title_label = ttk.Label(header_frame, text="🛡️ PhishGuard", style='Title.TLabel')
        title_label.pack()
        
        subtitle_label = ttk.Label(header_frame, 
                                  text="Advanced AI-powered phishing detection and URL security analysis",
                                  style='Subtitle.TLabel')
        subtitle_label.pack(pady=(5, 0))
        
        # Flask server status
        status_frame = ttk.Frame(header_frame, style='Custom.TFrame')
        status_frame.pack(pady=(10, 0))
        
        self.server_status = ttk.Label(status_frame, text="Flask Server: Not Running", 
                                      font=('Arial', 10), background='#1e3a8a', foreground='#fbbf24')
        self.server_status.pack(side='left')
        
        self.server_btn = ttk.Button(status_frame, text="Start Server", 
                                    command=self.toggle_server, style='Analysis.TButton')
        self.server_btn.pack(side='right', padx=(10, 0))
        
        # Input section
        input_frame = ttk.Frame(main_frame, style='Custom.TFrame')
        input_frame.pack(fill='x', pady=(0, 20))
        
        input_title = ttk.Label(input_frame, text="URL Security Analysis", style='Section.TLabel')
        input_title.pack(anchor='w', pady=(0, 5))
        
        input_desc = ttk.Label(input_frame, 
                              text="Enter a URL to check if it's safe or potentially malicious",
                              style='Info.TLabel')
        input_desc.pack(anchor='w', pady=(0, 10))
        
        # URL input
        url_frame = ttk.Frame(input_frame)
        url_frame.pack(fill='x', pady=(0, 10))
        
        self.url_entry = ttk.Entry(url_frame, font=('Arial', 12))
        self.url_entry.pack(side='left', fill='x', expand=True, padx=(0, 10))
        self.url_entry.insert(0, "Enter URL to analyze (e.g., https://example.com)")
        self.url_entry.bind('<FocusIn>', self.clear_placeholder)
        self.url_entry.bind('<Return>', lambda e: self.analyze_url())
        
        self.analyze_btn = ttk.Button(url_frame, text="🛡️ Analyze URL", 
                                     command=self.analyze_url, style='Analysis.TButton')
        self.analyze_btn.pack(side='right')
        
        # Progress bar
        self.progress = ttk.Progressbar(input_frame, mode='indeterminate')
        self.progress.pack(fill='x', pady=(0, 10))
        self.progress.pack_forget()  # Hide initially
        
        # Results section
        self.results_frame = ttk.Frame(main_frame, style='Custom.TFrame')
        self.results_frame.pack(fill='both', expand=True)
        self.results_frame.pack_forget()  # Hide initially
        
        # Info cards section
        info_frame = ttk.Frame(main_frame, style='Custom.TFrame')
        info_frame.pack(fill='x', pady=(20, 0))
        
        self.create_info_cards(info_frame)
        
        # Flask server thread
        self.server_thread = None
        self.server_running = False
    
    def toggle_server(self):
        """Toggle Flask server on/off"""
        if not self.server_running:
            self.start_flask_server()
        else:
            self.stop_flask_server()
    
    def start_flask_server(self):
        """Start Flask server in background thread"""
        if not self.server_running:
            self.server_thread = threading.Thread(target=self.run_flask_server, daemon=True)
            self.server_thread.start()
            self.server_running = True
            self.server_status.configure(text="Flask Server: Running on http://localhost:5000", 
                                        foreground='#22c55e')
            self.server_btn.configure(text="Stop Server")
    
    def stop_flask_server(self):
        """Stop Flask server"""
        self.server_running = False
        self.server_status.configure(text="Flask Server: Stopped", foreground='#ef4444')
        self.server_btn.configure(text="Start Server")
    
    def run_flask_server(self):
        """Run Flask server"""
        try:
            self.app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
        except Exception as e:
            print(f"Flask server error: {e}")
            self.server_running = False
    
    def create_info_cards(self, parent):
        """Create information cards at the bottom"""
        cards_frame = ttk.Frame(parent)
        cards_frame.pack(fill='x')
        
        # Configure grid weights
        cards_frame.columnconfigure(0, weight=1)
        cards_frame.columnconfigure(1, weight=1)
        cards_frame.columnconfigure(2, weight=1)
        
        # Fast Analysis card
        card1 = ttk.Frame(cards_frame, style='Custom.TFrame', relief='raised', borderwidth=1)
        card1.grid(row=0, column=0, padx=5, pady=5, sticky='ew')
        
        ttk.Label(card1, text="📊 Fast Analysis", style='Section.TLabel').pack(pady=(10, 5))
        ttk.Label(card1, text="Get instant results on URL safety and security status\nusing advanced rule-based detection.",
                 style='Info.TLabel', justify='center').pack(pady=(0, 10))
        
        # Detailed Insights card
        card2 = ttk.Frame(cards_frame, style='Custom.TFrame', relief='raised', borderwidth=1)
        card2.grid(row=0, column=1, padx=5, pady=5, sticky='ew')
        
        ttk.Label(card2, text="🛡️ Detailed Insights", style='Section.TLabel').pack(pady=(10, 5))
        ttk.Label(card2, text="View comprehensive analysis of URL structure,\nsecurity features, and potential threats.",
                 style='Info.TLabel', justify='center').pack(pady=(0, 10))
        
        # Clear Reports card
        card3 = ttk.Frame(cards_frame, style='Custom.TFrame', relief='raised', borderwidth=1)
        card3.grid(row=0, column=2, padx=5, pady=5, sticky='ew')
        
        ttk.Label(card3, text="✅ Clear Reports", style='Section.TLabel').pack(pady=(10, 5))
        ttk.Label(card3, text="Easy-to-understand security reports with\nvisual risk indicators and actionable warnings.",
                 style='Info.TLabel', justify='center').pack(pady=(0, 10))
    
    def clear_placeholder(self, event):
        """Clear placeholder text when entry is focused"""
        if self.url_entry.get() == "Enter URL to analyze (e.g., https://example.com)":
            self.url_entry.delete(0, tk.END)
    
    def analyze_url(self):
        """Analyze the entered URL"""
        url = self.url_entry.get().strip()
        
        if not url or url == "Enter URL to analyze (e.g., https://example.com)":
            messagebox.showerror("Error", "Please enter a URL to analyze")
            return
        
        # Show progress bar and disable button
        self.progress.pack(fill='x', pady=(0, 10))
        self.progress.start()
        self.analyze_btn.configure(state='disabled', text="Analyzing...")
        
        # Run analysis in separate thread
        thread = threading.Thread(target=self.perform_analysis, args=(url,))
        thread.daemon = True
        thread.start()
    
    def perform_analysis(self, url):
        """Perform URL analysis in background thread"""
        try:
            # Simulate processing time
            time.sleep(2)
            
            analysis_result = self.analyze_url_features(url)
            
            # Update GUI in main thread
            self.root.after(0, self.display_results, analysis_result)
            
        except Exception as e:
            self.root.after(0, self.show_error, f"Analysis failed: {str(e)}")
    
    def extract_content_features(self, url):
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
    
    def analyze_url_features(self, input_url):
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
        
        # Extract features (same as before)
        features = {
            'url_length': len(full_url),
            'domain_length': len(domain),
            'subdomain_count': len(domain.split('.')) - 2 if domain else 0,
            'has_https': parsed.scheme == 'https',
            'has_ip_address': self.is_ip_address(domain),
            'suspicious_keywords': self.count_suspicious_keywords(full_url.lower()),
            'has_url_shortener': any(shortener in domain for shortener in self.URL_SHORTENERS),
            'special_char_count': len(re.findall(r'[^a-zA-Z0-9\-._~:/?#[\]@!$&\'()*+,;=]', full_url)),
            'domain_age': self.estimate_domain_age(domain),
            'has_valid_cert': parsed.scheme == 'https'
        }
        
        # Extract content features
        content_features = self.extract_content_features(normalized_url)
        features.update(content_features)
        
        # If model is loaded, use it for prediction
        if self.model_loaded:
            try:
                # Convert features to DataFrame in the correct order expected by the model
                feature_df = pd.DataFrame([features])
                
                # Make prediction (adjust according to your model's requirements)
                prediction = self.model.predict(feature_df)[0]
                probability = self.model.predict_proba(feature_df)[0][1]  # Probability of being phishing
                
                is_phishing = bool(prediction)
                risk_score = int(probability * 100)
            except Exception as e:
                print(f"Model prediction failed: {e}")
                # Fall back to rule-based if model fails
                is_phishing, risk_score = self.rule_based_analysis(features)
        else:
            # Use rule-based analysis if model isn't loaded
            is_phishing, risk_score = self.rule_based_analysis(features)
        
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
            'warnings': self.generate_warnings(features, risk_score),
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    
    def rule_based_analysis(self, features):
        """Fallback rule-based analysis if model isn't available"""
        risk_score = 0
        
        # Your existing rule-based scoring logic
        if features['url_length'] > 100:
            risk_score += 15
        
        if features['has_ip_address']:
            risk_score += 25
        
        if features['subdomain_count'] > 3:
            risk_score += 20
        
        if not features['has_https']:
            risk_score += 15
        
        # Add more rules as needed...
        
        risk_score = min(risk_score, 100)
        is_phishing = risk_score >= 50
        
        return is_phishing, risk_score

    def generate_warnings(self, features, risk_score):
        """Generate warnings based on features"""
        warnings = []
        
        if features['url_length'] > 100:
            warnings.append("URL is unusually long")
        
        if features['has_ip_address']:
            warnings.append("URL uses IP address instead of domain name")
        
        if features['subdomain_count'] > 3:
            warnings.append("URL has excessive subdomains")
        
        if not features['has_https']:
            warnings.append("URL does not use HTTPS encryption")
        
        # Add more warning conditions as needed...
        
        return warnings

    def is_ip_address(self, domain):
        """Check if domain is an IP address"""
        ipv4_pattern = r'^(\d{1,3}\.){3}\d{1,3}$'
        ipv6_pattern = r'^([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$'
        return bool(re.match(ipv4_pattern, domain) or re.match(ipv6_pattern, domain))
    
    def count_suspicious_keywords(self, url):
        """Count suspicious keywords in URL"""
        count = 0
        for keyword in self.SUSPICIOUS_KEYWORDS:
            count += url.count(keyword)
        return count
    
    def estimate_domain_age(self, domain):
        """Estimate domain age based on TLD"""
        if any(tld in domain for tld in self.SUSPICIOUS_TLDS):
            return "Recently registered"
        elif domain.endswith(('.com', '.org', '.net')):
            return "Established"
        return "Unknown"
    
    def display_results(self, analysis):
        """Display analysis results"""
        # Hide progress bar and re-enable button
        self.progress.stop()
        self.progress.pack_forget()
        self.analyze_btn.configure(state='normal', text="🛡️ Analyze URL")
        
        # Clear previous results
        for widget in self.results_frame.winfo_children():
            widget.destroy()
        
        # Show results frame
        self.results_frame.pack(fill='both', expand=True)
        
        # Main result header
        result_header = ttk.Frame(self.results_frame, style='Custom.TFrame')
        result_header.pack(fill='x', pady=(0, 10))
        
        # Risk indicator
        risk_color = self.get_risk_color(analysis['risk_level'])
        status_text = "⚠️ Potentially Malicious" if analysis['is_phishing'] else "✅ Likely Safe"
        
        ttk.Label(result_header, text=f"Security Analysis Result - {status_text}",
                 style='Section.TLabel').pack(side='left')
        
        ttk.Label(result_header, text=f"{analysis['risk_level']} RISK ({analysis['risk_score']}%)",
                 foreground=risk_color, font=('Arial', 12, 'bold'),
                 background='#1e40af').pack(side='right')
        
        # Create notebook for tabbed results
        notebook = ttk.Notebook(self.results_frame)
        notebook.pack(fill='both', expand=True)
        
        # Overview tab
        overview_frame = ttk.Frame(notebook)
        notebook.add(overview_frame, text="Overview")
        self.create_overview_tab(overview_frame, analysis)
        
        # Details tab
        details_frame = ttk.Frame(notebook)
        notebook.add(details_frame, text="URL Analysis")
        self.create_details_tab(details_frame, analysis)
        
        # Content Analysis tab
        content_frame = ttk.Frame(notebook)
        notebook.add(content_frame, text="Content Analysis")
        self.create_content_tab(content_frame, analysis)
        
        # Warnings tab
        if analysis['warnings']:
            warnings_frame = ttk.Frame(notebook)
            notebook.add(warnings_frame, text="Security Warnings")
            self.create_warnings_tab(warnings_frame, analysis)
    
    def create_overview_tab(self, parent, analysis):
        """Create overview tab content"""
        # Risk meter
        risk_frame = ttk.LabelFrame(parent, text="Threat Assessment", padding=10)
        risk_frame.pack(fill='x', pady=(0, 10))
        
        # Risk percentage display
        risk_display = ttk.Frame(risk_frame)
        risk_display.pack(fill='x')
        
        ttk.Label(risk_display, text="Threat Probability:",
                 font=('Arial', 12)).pack(side='left')
        
        risk_color = self.get_risk_color(analysis['risk_level'])
        ttk.Label(risk_display, text=f"{analysis['risk_score']}%",
                 font=('Arial', 12, 'bold'), foreground=risk_color).pack(side='right')
        
        # Progress bar for risk
        risk_progress = ttk.Progressbar(risk_frame, length=400, mode='determinate')
        risk_progress.pack(fill='x', pady=(10, 0))
        risk_progress['value'] = analysis['risk_score']
        
        # Risk level indicators
        indicators_frame = ttk.Frame(risk_frame)
        indicators_frame.pack(fill='x', pady=(5, 0))
        
        ttk.Label(indicators_frame, text="🛡️ Safe", font=('Arial', 8)).pack(side='left')
        ttk.Label(indicators_frame, text="⚠️ Suspicious", font=('Arial', 8)).pack()
        ttk.Label(indicators_frame, text="🚨 Malicious", font=('Arial', 8)).pack(side='right')
        
        # Analysis summary
        summary_frame = ttk.LabelFrame(parent, text="Analysis Summary", padding=10)
        summary_frame.pack(fill='both', expand=True)
        
        summary_text = scrolledtext.ScrolledText(summary_frame, height=8, wrap=tk.WORD)
        summary_text.pack(fill='both', expand=True)
        
        # Generate summary
        summary = f"URL: {analysis['url']}\n"
        summary += f"Analysis Time: {analysis['timestamp']}\n\n"
        summary += f"Risk Level: {analysis['risk_level']} ({analysis['risk_score']}%)\n"
        summary += f"Classification: {'Potentially Malicious' if analysis['is_phishing'] else 'Likely Safe'}\n\n"
        
        if analysis['is_phishing']:
            summary += "🚨 WARNING: This URL exhibits characteristics commonly found in phishing attacks. Exercise extreme caution.\n\n"
        else:
            summary += "✅ This URL appears legitimate based on our analysis. However, always remain vigilant.\n\n"
        
        summary += "Key Findings:\n"
        for warning in analysis['warnings'][:5]:  # Show top 5 warnings
            summary += f"• {warning}\n"
        
        summary_text.insert(tk.END, summary)
        summary_text.configure(state='disabled')
    
    def create_details_tab(self, parent, analysis):
        """Create details tab content"""
        # URL structure analysis
        structure_frame = ttk.LabelFrame(parent, text="URL Structure Analysis", padding=10)
        structure_frame.pack(fill='x', pady=(0, 10))
        
        # Create grid for details
        details = [
            ("Protocol", "HTTPS" if analysis['features']['has_https'] else "HTTP"),
            ("URL Length", f"{analysis['features']['url_length']} characters"),
            ("Domain Length", f"{analysis['features']['domain_length']} characters"),
            ("Subdomains", f"{analysis['features']['subdomain_count']} subdomains"),
            ("Suspicious Keywords", f"{analysis['features']['suspicious_keywords']} found"),
            ("Domain Age", analysis['features']['domain_age']),
            ("IP Address", "Uses IP" if analysis['features']['has_ip_address'] else "Uses Domain"),
            ("URL Shortener", "Yes" if analysis['features']['has_url_shortener'] else "No"),
            ("Special Characters", f"{analysis['features']['special_char_count']} found")
        ]
        
        for i, (label, value) in enumerate(details):
            row = i // 3
            col = i % 3
            
            detail_frame = ttk.Frame(structure_frame)
            detail_frame.grid(row=row, column=col, padx=5, pady=5, sticky='w')
            
            ttk.Label(detail_frame, text=f"{label}:", font=('Arial', 9, 'bold')).pack(anchor='w')
            ttk.Label(detail_frame, text=value, font=('Arial', 9)).pack(anchor='w')
        
        # Configure grid weights
        for i in range(3):
            structure_frame.columnconfigure(i, weight=1)
    
    def create_content_tab(self, parent, analysis):
        """Create content analysis tab"""
        content_frame = ttk.LabelFrame(parent, text="Website Content Analysis", padding=10)
        content_frame.pack(fill='x', pady=(0, 10))
        
        # Content details
        content_details = [
            ("Page Title", "Present" if analysis['features'].get('has_title', 0) else "Missing"),
            ("Title Length", f"{analysis['features'].get('title_length', 0)} characters"),
            ("Favicon", "Present" if analysis['features'].get('has_favicon', 0) else "Missing"),
            ("Forms", f"{analysis['features'].get('form_count', 0)} found"),
            ("Input Fields", f"{analysis['features'].get('input_count', 0)} found"),
            ("Images", f"{analysis['features'].get('image_count', 0)} found"),
            ("Scripts", f"{analysis['features'].get('script_count', 0)} found"),
            ("External Links", f"{analysis['features'].get('external_links_count', 0)} found"),
            ("Internal Links", f"{analysis['features'].get('internal_links_count', 0)} found")
        ]
        
        for i, (label, value) in enumerate(content_details):
            row = i // 3
            col = i % 3
            
            detail_frame = ttk.Frame(content_frame)
            detail_frame.grid(row=row, column=col, padx=5, pady=5, sticky='w')
            
            ttk.Label(detail_frame, text=f"{label}:", font=('Arial', 9, 'bold')).pack(anchor='w')
            ttk.Label(detail_frame, text=value, font=('Arial', 9)).pack(anchor='w')
        
        # Configure grid weights
        for i in range(3):
            content_frame.columnconfigure(i, weight=1)
    
    def create_warnings_tab(self, parent, analysis):
        """Create warnings tab content"""
        warnings_frame = ttk.LabelFrame(parent, text="Security Warnings", padding=10)
        warnings_frame.pack(fill='both', expand=True)
        
        warnings_text = scrolledtext.ScrolledText(warnings_frame, height=10, wrap=tk.WORD)
        warnings_text.pack(fill='both', expand=True)
        
        warnings_content = "Security Issues Detected:\n\n"
        for i, warning in enumerate(analysis['warnings'], 1):
            warnings_content += f"{i}. {warning}\n"
        
        warnings_content += "\nRecommendations:\n"
        warnings_content += "• Verify the URL with the official website\n"
        warnings_content += "• Check for spelling errors in the domain name\n"
        warnings_content += "• Look for secure connection indicators (HTTPS)\n"
        warnings_content += "• Be cautious of urgent or threatening language\n"
        warnings_content += "• When in doubt, don't click or enter personal information\n"
        
        warnings_text.insert(tk.END, warnings_content)
        warnings_text.configure(state='disabled')
    
    def get_risk_color(self, risk_level):
        """Get color based on risk level"""
        colors = {
            'LOW': '#22c55e',    # Green
            'MEDIUM': '#f59e0b',  # Yellow
            'HIGH': '#ef4444'     # Red
        }
        return colors.get(risk_level, '#6b7280')
    
    def show_error(self, error_message):
        """Show error message"""
        self.progress.stop()
        self.progress.pack_forget()
        self.analyze_btn.configure(state='normal', text="🛡️ Analyze URL")
        messagebox.showerror("Analysis Error", error_message)
    
    def run(self):
        """Start the application"""
        print("Starting PhishGuard...")
        print("GUI application ready!")
        print("Flask API will be available at http://localhost:5000 when started")
        self.root.mainloop()

if __name__ == "__main__":
    app = PhishingDetector()
    app.run()