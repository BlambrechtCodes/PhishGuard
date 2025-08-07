# Enhanced Phishing Detector with TensorFlow Neural Network
# This script is designed to train a phishing URL detection model using a dataset of URLs.
#! This is an IMPROVED version of the original phishML.py.
#! This code DOES NOT WORK with the Application "app.py" in the same directory.

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
import warnings
warnings.filterwarnings('ignore')

print(tf.__version__)
print("\n=== Enhanced Phishing Detector with TensorFlow ===")

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
        print("Dataset shape:", df.shape)
        
        # Data quality checks
        print("\n=== Data Quality Checks ===")
        print(f"Duplicate rows: {df.duplicated().sum()}")
        print("Label distribution:\n", df['label'].value_counts())
        
        # Feature engineering
        X = df.select_dtypes(include=[np.number])
        y = df['label']
        
        # Remove problematic features
        features_to_drop = ['URLSimilarityIndex', 'HasSocialNet', 'HasCopyrightInfo']
        X = X.drop(columns=[col for col in features_to_drop if col in X.columns], errors='ignore')
        
        # Remove highly correlated features with target (aggressive removal)
        correlations = X.corrwith(y).abs()
        high_corr_features = correlations[correlations > 0.3].index.tolist()  # Lower threshold
        if high_corr_features:
            print("Dropping highly correlated features:", high_corr_features)
            X = X.drop(columns=high_corr_features)
        
        # Remove label if present
        if 'label' in X.columns:
            X = X.drop(columns=['label'])
            
        self.feature_columns = list(X.columns)
        print("Final features:", len(self.feature_columns), ":", self.feature_columns)
        
        return X, y
    
    def build_tensorflow_model(self, input_dim):
        """Build TensorFlow neural network model with balanced complexity"""
        print("\n=== Building TensorFlow Model ===")
        
        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(input_dim,)),
            tf.keras.layers.BatchNormalization(),
            
            # Balanced architecture
            tf.keras.layers.Dense(96, activation='relu', 
                                kernel_regularizer=tf.keras.regularizers.l2(0.005)),
            tf.keras.layers.Dropout(0.4),
            
            tf.keras.layers.Dense(48, activation='relu',
                                kernel_regularizer=tf.keras.regularizers.l2(0.005)),
            tf.keras.layers.Dropout(0.3),
            
            tf.keras.layers.Dense(24, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        # Compile with custom optimizer
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0008),
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.AUC(name='auc')
            ]
        )
        
        model.summary()
        return model
    
    def train_models(self, X, y, validation_split=0.25):  # No test split here
        """Train both TensorFlow and Decision Tree models"""
        print("\n=== Scaling Features ===")
        X_scaled = self.scaler.fit_transform(X)
        
        # Build and train TensorFlow model
        self.tf_model = self.build_tensorflow_model(X.shape[1])
        
        # Enhanced callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', 
                patience=10,
                restore_best_weights=True,
                min_delta=0.001
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', 
                factor=0.6, 
                patience=5,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        print("\n=== Training TensorFlow Model ===")
        self.history = self.tf_model.fit(
            X_scaled, y,
            epochs=80,
            batch_size=128,  # Larger batch size
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )
        
        # Train Decision Tree with balanced parameters
        print("\n=== Training Decision Tree Model ===")
        self.tree_model = DecisionTreeClassifier(
            class_weight='balanced',
            max_depth=7,
            min_samples_split=15,
            min_samples_leaf=5,
            random_state=42
        )
        self.tree_model.fit(X, y)
        
    def evaluate_models(self, X_test, y_test):
        """Evaluate both models on test data"""
        print("\n=== Model Evaluation ===")
        
        # Scale test features
        X_test_scaled = self.scaler.transform(X_test)
        
        # TensorFlow predictions
        tf_pred_proba = self.tf_model.predict(X_test_scaled).flatten()
        tf_pred = (tf_pred_proba > 0.5).astype(int)
        
        # Decision Tree predictions
        dt_pred = self.tree_model.predict(X_test)
        dt_pred_proba = self.tree_model.predict_proba(X_test)[:, 1]
        
        # Print evaluation metrics
        print("\n--- TensorFlow Neural Network ---")
        print("Accuracy:", round(metrics.accuracy_score(y_test, tf_pred) * 100, 2), "%")
        print(classification_report(y_test, tf_pred, target_names=['Legitimate', 'Phishing']))
        
        print("\n--- Decision Tree ---")
        print("Accuracy:", round(metrics.accuracy_score(y_test, dt_pred) * 100, 2), "%")
        print(classification_report(y_test, dt_pred, target_names=['Legitimate', 'Phishing']))
        
        # Plot training history
        self.plot_training_history()
        
        # Plot confusion matrices
        self.plot_confusion_matrices(y_test, tf_pred, dt_pred)
        
        # Plot ROC curves
        self.plot_roc_curves(y_test, tf_pred_proba, dt_pred_proba)
        
        return tf_pred, dt_pred, tf_pred_proba, dt_pred_proba
    
    def plot_training_history(self):
        """Plot training history"""
        if self.history is None:
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Loss
        axes[0, 0].plot(self.history.history['loss'], label='Training Loss')
        axes[0, 0].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[0, 0].set_title('Model Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        
        # Accuracy
        axes[0, 1].plot(self.history.history['accuracy'], label='Training Accuracy')
        axes[0, 1].plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        axes[0, 1].set_title('Model Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        
        # Precision
        axes[1, 0].plot(self.history.history['precision'], label='Training Precision')
        axes[1, 0].plot(self.history.history['val_precision'], label='Validation Precision')
        axes[1, 0].set_title('Model Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        
        # Recall
        axes[1, 1].plot(self.history.history['recall'], label='Training Recall')
        axes[1, 1].plot(self.history.history['val_recall'], label='Validation Recall')
        axes[1, 1].set_title('Model Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Training history saved as training_history.png")
    
    def plot_confusion_matrices(self, y_test, tf_pred, dt_pred):
        """Plot confusion matrices for both models"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # TensorFlow confusion matrix
        tf_cm = confusion_matrix(y_test, tf_pred)
        sns.heatmap(tf_cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
        axes[0].set_title('TensorFlow Neural Network')
        axes[0].set_xlabel('Predicted')
        axes[0].set_ylabel('Actual')
        
        # Decision Tree confusion matrix
        dt_cm = confusion_matrix(y_test, dt_pred)
        sns.heatmap(dt_cm, annot=True, fmt='d', cmap='Greens', ax=axes[1])
        axes[1].set_title('Decision Tree')
        axes[1].set_xlabel('Predicted')
        axes[1].set_ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig('confusion_matrices_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Confusion matrices saved as confusion_matrices_comparison.png")
    
    def plot_roc_curves(self, y_test, tf_proba, dt_proba):
        """Plot ROC curves for both models"""
        plt.figure(figsize=(8, 6))
        
        # TensorFlow ROC
        tf_fpr, tf_tpr, _ = roc_curve(y_test, tf_proba)
        tf_auc = auc(tf_fpr, tf_tpr)
        plt.plot(tf_fpr, tf_tpr, label=f'TensorFlow NN (AUC = {tf_auc:.3f})', linewidth=2)
        
        # Decision Tree ROC
        dt_fpr, dt_tpr, _ = roc_curve(y_test, dt_proba)
        dt_auc = auc(dt_fpr, dt_tpr)
        plt.plot(dt_fpr, dt_tpr, label=f'Decision Tree (AUC = {dt_auc:.3f})', linewidth=2)
        
        # Random classifier
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('roc_curves_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("ROC curves saved as roc_curves_comparison.png")
    
    def predict_url(self, features, threshold=0.5, use_ensemble=True):
        """Predict if a URL is phishing"""
        # Ensure features are in correct order
        if isinstance(features, dict):
            # Fill missing features with 0
            for col in self.feature_columns:
                if col not in features:
                    features[col] = 0
            # Convert to DataFrame in correct order
            features_df = pd.DataFrame([features])[self.feature_columns]
        else:
            features_df = features
        
        # Scale features for TensorFlow model
        features_scaled = self.scaler.transform(features_df)
        
        # Get predictions from both models
        tf_proba = self.tf_model.predict(features_scaled)[0][0]
        dt_proba = self.tree_model.predict_proba(features_df)[0][1]
        
        if use_ensemble:
            # Ensemble prediction (weighted average)
            ensemble_proba = 0.7 * tf_proba + 0.3 * dt_proba  # Weight NN more
            prediction = int(ensemble_proba > threshold)
            return {
                'prediction': 'Phishing' if prediction else 'Legitimate',
                'confidence': ensemble_proba,
                'tensorflow_prob': tf_proba,
                'decision_tree_prob': dt_proba,
                'ensemble_prob': ensemble_proba
            }
        else:
            tf_prediction = int(tf_proba > threshold)
            return {
                'prediction': 'Phishing' if tf_prediction else 'Legitimate',
                'confidence': tf_proba,
                'tensorflow_prob': tf_proba,
                'decision_tree_prob': dt_proba
            }
    
    def save_models(self, prefix='phishing_detector'):
        """Save trained models"""
        # Save TensorFlow model
        self.tf_model.save(f'{prefix}_tensorflow_model.h5')
        
        # Save Decision Tree model
        joblib.dump(self.tree_model, f'{prefix}_decision_tree.pkl')
        
        # Save scaler and feature columns
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

# Usage Example
if __name__ == "__main__":
    # Initialize detector
    detector = PhishingDetector()
    
    # Load and preprocess data
    X, y = detector.load_and_preprocess_data('PhiUSIIL_Phishing_URL_Dataset.csv')
    
    # Train models
    X_train_scaled, X_test_scaled, X_test, y_train, y_test = detector.train_models(X, y)
    
    # Evaluate models
    tf_pred, dt_pred, tf_proba, dt_proba = detector.evaluate_models(X_test_scaled, X_test, y_test)
    
    # Save models
    detector.save_models()
    
    # Example prediction
    print("\n=== Example Prediction ===")
    example_features = {
        'URLLength': 85,
        'DomainLength': 22,
        'IsDomainIP': 0,
        'CharContinuationRate': 0.8,
        'TLDLegitimateProb': 0.2,
        'URLCharProb': 0.15,
        'TLDLength': 3,
        'NoOfSubDomain': 2,
    }
    
    result = detector.predict_url(example_features, threshold=0.3, use_ensemble=True)
    print(f"Prediction: {result['prediction']}")
    print(f"Ensemble Confidence: {result['ensemble_prob']:.3f}")
    print(f"TensorFlow Probability: {result['tensorflow_prob']:.3f}")
    print(f"Decision Tree Probability: {result['decision_tree_prob']:.3f}")
    
    print("\n=== Cross-Validation ===")
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []

    for fold_num, (train_idx, val_idx) in enumerate(kfold.split(X, y), 1):
        print(f"\n--- Cross-Validation Fold {fold_num}/{kfold.n_splits} ---")
        X_cv_train, X_cv_val = X.iloc[train_idx], X.iloc[val_idx]
        y_cv_train, y_cv_val = y.iloc[train_idx], y.iloc[val_idx]
        
        scaler_cv = StandardScaler()
        X_cv_train_scaled = scaler_cv.fit_transform(X_cv_train)
        X_cv_val_scaled = scaler_cv.transform(X_cv_val)
        
        print(f"  Training TensorFlow model for fold {fold_num}...")
        model_cv = detector.build_tensorflow_model(X_cv_train.shape[1])
        model_cv.fit(
            X_cv_train_scaled, y_cv_train,
            epochs=50, verbose=1,
            validation_split=0.2,
            callbacks=[tf.keras.callbacks.EarlyStopping(patience=10)]
        )
        print(f"  Training completed for fold {fold_num}. Evaluating...")
        
        val_pred = (model_cv.predict(X_cv_val_scaled) > 0.5).astype(int)
        score = metrics.accuracy_score(y_cv_val, val_pred)
        cv_scores.append(score)
        print(f"  Fold {fold_num} accuracy: {score:.4f}")

    print("\nCross-validation scores:", cv_scores)
    print(f"Mean CV accuracy: {np.mean(cv_scores):.3f} (+/- {np.std(cv_scores)*2:.3f})")
    