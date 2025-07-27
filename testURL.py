
#! This is just for testing purposes and should not be used in production without further validation.

#! This is phish1_MAIN but HEAVILY SIMPLIFIED for Debugging and Testing Purposes

# Enhanced Phishing Detector with TensorFlow Neural Network
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from urllib.parse import urlparse
import re
import warnings
warnings.filterwarnings('ignore')

class PhishingDetector:
    def __init__(self):
        self.tf_model = None
        self.tree_model = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.history = None
        
    def load_and_preprocess_data(self, filepath):
        """Load and preprocess the phishing dataset"""
        print("\n=== Loading Dataset ===")
        df = pd.read_csv(filepath)
        print("Label distribution:\n", df['label'].value_counts())
    
        # Verify features
        print("Sample features:\n", df.iloc[0])
        print("Dataset shape:", df.shape)
        
        # Feature engineering
        print("\n=== Feature Engineering ===")
        X = df.select_dtypes(include=[np.number])
        y = df['label']
        
        # Remove problematic features
        features_to_drop = ['URLSimilarityIndex', 'HasSocialNet', 'HasCopyrightInfo']
        X = X.drop(columns=[col for col in features_to_drop if col in X.columns])
        
        # Remove highly correlated features with target
        correlations = df.corr(numeric_only=True)['label'].abs()
        high_corr_features = correlations[correlations > 0.8].index.tolist()
        high_corr_features = [f for f in high_corr_features if f != 'label' and f in X.columns]
        if high_corr_features:
            print("Dropping highly correlated features:", high_corr_features)
            X = X.drop(columns=high_corr_features)
        
        if 'label' in X.columns:
            X = X.drop(columns=['label'])
            
        self.feature_columns = list(X.columns)
        print("Final features:", len(self.feature_columns))
        
        return X, y
    
    def build_tensorflow_model(self, input_dim):
        """Build TensorFlow neural network model"""
        print("\n=== Building TensorFlow Model ===")
        
        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(input_dim,)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(128, activation='relu', 
                                 kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, activation='relu',
                                kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(32, activation='relu',
                                kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(name='precision'),
                     tf.keras.metrics.Recall(name='recall')]
        )
        
        model.summary()
        return model
    
    def train_models(self, X, y, test_size=0.2, validation_split=0.2):
        """Train both TensorFlow and Decision Tree models"""
        print("\n=== Splitting Data ===")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Build and train TensorFlow model
        self.tf_model = self.build_tensorflow_model(X_train.shape[1])
        
        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-6)
        ]
        
        print("\n=== Training TensorFlow Model ===")
        class_weights = {0: 1., 1: len(y[y==0]) / len(y[y==1])}  # Inverse ratio
    
        # Add to model.fit()
        self.history = self.tf_model.fit(
            X_train_scaled, y_train,
            class_weight=class_weights,  # Add this
            epochs=100,
            batch_size=32,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
    )
        
        # Train Decision Tree
        self.tree_model = DecisionTreeClassifier(
            class_weight='balanced',
            max_depth=10,
            min_samples_split=10,
            random_state=42
        )
        self.tree_model.fit(X_train, y_train)
        
        return X_train_scaled, X_test_scaled, X_test, y_train, y_test
    
    def evaluate_models(self, X_test_scaled, X_test, y_test):
        """Evaluate both models"""
        tf_pred_proba = self.tf_model.predict(X_test_scaled).flatten()
        tf_pred = (tf_pred_proba > 0.5).astype(int)
        dt_pred = self.tree_model.predict(X_test)
        dt_pred_proba = self.tree_model.predict_proba(X_test)[:, 1]
        
        print("\n--- TensorFlow Neural Network ---")
        print("Accuracy:", round(metrics.accuracy_score(y_test, tf_pred) * 100, 2), "%")
        print(classification_report(y_test, tf_pred, target_names=['Legitimate', 'Phishing']))
        
        print("\n--- Decision Tree ---")
        print("Accuracy:", round(metrics.accuracy_score(y_test, dt_pred) * 100, 2), "%")
        print(classification_report(y_test, dt_pred, target_names=['Legitimate', 'Phishing']))
        
        return tf_pred, dt_pred, tf_pred_proba, dt_pred_proba
    
    # Modify your feature extraction to ensure it matches training data
    def extract_url_features(self, url):
        parsed = urlparse(url)
        domain = parsed.netloc
        path = parsed.path.lower()
        query = parsed.query.lower()
        
        features = {
            # Structural features
            'URLLength': len(url),
            'DomainLength': len(domain),
            'PathLength': len(path),
            'NumSubdomains': domain.count('.'),
            'IsIP': int(bool(re.match(r'\d+\.\d+\.\d+\.\d+', domain))),
            'IsHTTPS': int(parsed.scheme == 'https'),
            
            # Suspicious patterns
            'HasLogin': int('login' in path or 'signin' in path),
            'HasAccount': int('account' in path or 'verify' in path),
            'HasSecure': int('secure' in domain or 'secure' in path),
            'HasBanking': int('bank' in path or 'pay' in path or 'account' in path),
            'HasSensitive': int('password' in path or 'reset' in path or 'confirm' in path),
            
            # Query characteristics
            'NumParams': query.count('&') + 1 if query else 0,
            'HasSuspiciousParam': int('redirect' in query or 'return' in query),
            
            # Domain characteristics
            'TLD': len(domain.split('.')[-1]) if '.' in domain else 0,
            'IsCommonTLD': int(domain.endswith(('.com','.org','.net','.gov'))),
            'IsSuspiciousTLD': int(domain.endswith(('.xyz','.top','.gq','.tk'))),
            
            # Character analysis
            'SpecialCharRatio': len(re.findall(r'[^\w\s]', url)) / max(1, len(url)),
            'DigitRatio': sum(c.isdigit() for c in url) / max(1, len(url)),
        }
        
        # Ensure all expected features are present
        for col in self.feature_columns:
            if col not in features:
                features[col] = 0
                
        return features
    
    def predict_url(self, url_or_features, threshold=0.3, use_ensemble=True):
        """Predict if a URL is phishing"""
        if isinstance(url_or_features, str):
            features = self.extract_url_features(url_or_features)
        else:
            features = url_or_features
        
        # Ensure all features are present
        for col in self.feature_columns:
            if col not in features:
                features[col] = 0
                
        features_df = pd.DataFrame([features], columns=self.feature_columns)
        features_scaled = self.scaler.transform(features_df)
        
        tf_proba = self.tf_model.predict(features_scaled)[0][0]
        dt_proba = self.tree_model.predict_proba(features_df)[0][1]
        
        if use_ensemble:
        # Give more weight to TF model since DT isn't performing
            ensemble_proba = 0.9 * tf_proba + 0.1 * dt_proba
            prediction = int(ensemble_proba > threshold)
            return {
                'prediction': 'Phishing' if prediction else 'Legitimate',
                'probability': ensemble_proba,
                'confidence': 'High' if ensemble_proba > 0.7 or ensemble_proba < 0.3 else 'Medium',
                'tensorflow_prob': tf_proba,
                'decision_tree_prob': dt_proba
        }
        else:
            prediction = int(tf_proba > threshold)
            return {
                'prediction': 'Phishing' if prediction else 'Legitimate',
                'probability': tf_proba,
                'confidence': 'High' if tf_proba > 0.8 or tf_proba < 0.2 else 'Medium',
                'tensorflow_prob': tf_proba,
                'decision_tree_prob': dt_proba
            }
    
    def save_models(self, prefix='phishing_detector'):
        """Save trained models"""
        self.tf_model.save(f'{prefix}_tensorflow_model.h5')
        joblib.dump(self.tree_model, f'{prefix}_decision_tree.pkl')
        joblib.dump(self.scaler, f'{prefix}_scaler.pkl')
        joblib.dump(self.feature_columns, f'{prefix}_features.pkl')
        print(f"Models saved with prefix: {prefix}")
    
    def load_models(self, prefix='phishing_detector'):
        """Load trained models"""
        try:
            self.tf_model = tf.keras.models.load_model(f'{prefix}_tensorflow_model.h5')
            self.tree_model = joblib.load(f'{prefix}_decision_tree.pkl')
            self.scaler = joblib.load(f'{prefix}_scaler.pkl')
            self.feature_columns = joblib.load(f'{prefix}_features.pkl')
            print(f"Models loaded successfully from prefix: {prefix}")
            return True
        except Exception as e:
            print(f"Error loading models: {e}")
            return False

if __name__ == "__main__":
    detector = PhishingDetector()
    
    # Example usage
    X, y = detector.load_and_preprocess_data('PhiUSIIL_Phishing_URL_Dataset.csv')
    X_train_scaled, X_test_scaled, X_test, y_train, y_test = detector.train_models(X, y)
    detector.evaluate_models(X_test_scaled, X_test, y_test)
    detector.save_models()
    
    # Example test predictions
    print("\n=== Phishing URL Detection Test Cases ===")

    # 1. Clearly legitimate URLs
    legit_urls = [
        "https://www.google.com",
        "https://www.microsoft.com/en-us",
        "https://www.apple.com",
        "https://www.wikipedia.org",
        "https://www.linkedin.com"
    ]

    print("\n--- Legitimate URLs ---")
    for url in legit_urls:
        result = detector.predict_url(url)
        print(f"\nURL: {url}")
        print(f"Prediction: {result['prediction']}")
        print(f"Probability: {result['probability']:.4f} (0=legit, 1=phish)")
        print(f"Confidence: {result['confidence']}")
        print(f"TF Prob: {result['tensorflow_prob']:.4f}, DT Prob: {result['decision_tree_prob']:.4f}")

    # 2. Known phishing patterns
    phishing_urls = [
        "http://secure-google-login.com",
        "https://facebook.verify-account.com",
        "http://appleid.apple.com.verify-account.net/login",
        "https://paypal-confirm-account.com",
        "http://netflix-renew-subscription.xyz"
    ]

    print("\n--- Phishing URLs ---")
    for url in phishing_urls:
        result = detector.predict_url(url)
        print(f"\nURL: {url}")
        print(f"Prediction: {result['prediction']}")
        print(f"Probability: {result['probability']:.4f} (0=legit, 1=phish)")
        print(f"Confidence: {result['confidence']}")
        print(f"TF Prob: {result['tensorflow_prob']:.4f}, DT Prob: {result['decision_tree_prob']:.4f}")

    # 3. Suspicious but ambiguous cases
    ambiguous_urls = [
        "https://account-update.com",
        "http://secure-login.net",
        "https://payment-confirmation.org",
        "http://profile-verification.xyz",
        "https://password-reset.net"
    ]

    print("\n--- Ambiguous URLs ---")
    for url in ambiguous_urls:
        result = detector.predict_url(url)
        print(f"\nURL: {url}")
        print(f"Prediction: {result['prediction']}")
        print(f"Probability: {result['probability']:.4f} (0=legit, 1=phish)")
        print(f"Confidence: {result['confidence']}")
        print(f"TF Prob: {result['tensorflow_prob']:.4f}, DT Prob: {result['decision_tree_prob']:.4f}")

    # 4. IP address URLs
    ip_urls = [
        "http://192.168.1.1/login",
        "https://10.0.0.1/admin",
        "http://172.16.254.1/verify"
    ]

    print("\n--- IP Address URLs ---")
    for url in ip_urls:
        result = detector.predict_url(url)
        print(f"\nURL: {url}")
        print(f"Prediction: {result['prediction']}")
        print(f"Probability: {result['probability']:.4f} (0=legit, 1=phish)")
        print(f"Confidence: {result['confidence']}")
        print(f"TF Prob: {result['tensorflow_prob']:.4f}, DT Prob: {result['decision_tree_prob']:.4f}")

    # 5. Long, complex URLs
    complex_urls = [
        "https://www.amazon.com/gp/product/B07PGL2Z5J/ref=ppx_yo_dt_b_asin_title_o00_s00?ie=UTF8&psc=1",
        "http://secure-payment-gateway.com/process.php?session=df8s7df6s7d6f&user=12345&token=sd87f6sd87f6sd"
    ]

    print("\n--- Complex URLs ---")
    for url in complex_urls:
        result = detector.predict_url(url)
        print(f"\nURL: {url}")
        print(f"Prediction: {result['prediction']}")
        print(f"Probability: {result['probability']:.4f} (0=legit, 1=phish)")
        print(f"Confidence: {result['confidence']}")
        print(f"TF Prob: {result['tensorflow_prob']:.4f}, DT Prob: {result['decision_tree_prob']:.4f}")

    # 6. Test with feature dictionaries
    print("\n--- Testing with Feature Dictionaries ---")

    # Legitimate features
    legit_features = {
        'URLLength': 22,
        'DomainLength': 14,
        'IsDomainIP': 0,
        'CharContinuationRate': 0.05,
        'TLDLegitimateProb': 0.9,
        'URLCharProb': 0.05,
        'TLDLength': 3,
        'NoOfSubDomain': 1,
        'IsHTTPS': 1
    }

    result = detector.predict_url(legit_features)
    print("\nLegitimate Features:")
    print(f"Prediction: {result['prediction']}")
    print(f"Probability: {result['probability']:.4f}")
    print(f"Confidence: {result['confidence']}")

    # Phishing features
    phish_features = {
        'URLLength': 68,
        'DomainLength': 32,
        'IsDomainIP': 0,
        'CharContinuationRate': 0.35,
        'TLDLegitimateProb': 0.1,
        'URLCharProb': 0.35,
        'TLDLength': 3,
        'NoOfSubDomain': 3,
        'IsHTTPS': 0,
        'HasObfuscation': 1,
        'Bank': 1
    }

    result = detector.predict_url(phish_features)
    print("\nPhishing Features:")
    print(f"Prediction: {result['prediction']}")
    print(f"Probability: {result['probability']:.4f}")
    print(f"Confidence: {result['confidence']}")