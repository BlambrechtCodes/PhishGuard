# Enhanced Phishing Detector with TensorFlow Neural Network
# This script is designed to train a phishing URL detection model using a dataset of URLs.
#! This is an WORSE version of the original phish1.py.
#! This code DOES NOT WORK with the Application "app.py" in the same directory.

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from urllib.parse import urlparse
import warnings
import shap
import eli5
from eli5.sklearn import PermutationImportance
from sklearn.inspection import permutation_importance
import time
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline

warnings.filterwarnings('ignore')

print("TensorFlow version:", tf.__version__)
print("\n=== Enhanced Phishing Detector with TensorFlow ===")

class PhishingDetector:
    def __init__(self):
        self.tf_model = None
        self.tree_model = None
        self.scaler = RobustScaler()  # Changed to RobustScaler for better handling of outliers
        self.feature_columns = None
        self.history = None
        self.feature_selector = None
        self.best_threshold = 0.5  # Will be optimized during training
        self.class_weights = None
        
    def load_and_preprocess_data(self, filepath):
        """Load and preprocess the phishing dataset with enhanced feature engineering"""
        print("\n=== Loading Dataset ===")
        df = pd.read_csv(filepath)
        print("Dataset shape:", df.shape)
        
        # Data quality checks
        print("\n=== Data Quality Checks ===")
        print(f"Duplicate rows: {df.duplicated().sum()}")
        print("Label distribution:\n", df['label'].value_counts())
        
        # Remove duplicates if any
        df = df.drop_duplicates()
        
        # Enhanced feature engineering
        print("\n=== Enhanced Feature Engineering ===")
        X = df.select_dtypes(include=[np.number])
        y = df['label']
        
        # Remove problematic features
        features_to_drop = ['URLSimilarityIndex', 'HasSocialNet', 'HasCopyrightInfo']
        X = X.drop(columns=[col for col in features_to_drop if col in X.columns], errors='ignore')
        
        # Advanced feature selection
        print("\n=== Feature Selection ===")
        selector = SelectKBest(f_classif, k=min(30, X.shape[1]))  # Select top 30 features
        X_selected = selector.fit_transform(X, y)
        self.feature_selector = selector
        
        # Get selected feature names
        selected_mask = selector.get_support()
        self.feature_columns = X.columns[selected_mask].tolist()
        X = pd.DataFrame(X_selected, columns=self.feature_columns)
        
        print(f"Selected {len(self.feature_columns)} features:")
        print(self.feature_columns)
        
        # Calculate class weights for imbalanced data
        class_counts = y.value_counts()
        self.class_weights = {0: 1, 1: class_counts[0]/class_counts[1]}
        print(f"\nClass weights: {self.class_weights}")
        
        return X, y
    
    def build_tensorflow_model(self, input_dim):
        """Build enhanced TensorFlow neural network model"""
        print("\n=== Building Enhanced TensorFlow Model ===")
        
        # Input layer
        inputs = tf.keras.layers.Input(shape=(input_dim,))
        
        # Feature normalization
        x = tf.keras.layers.BatchNormalization()(inputs)
        
        # First hidden block with residual connection
        x1 = tf.keras.layers.Dense(256, activation='swish', 
                                 kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.001, l2=0.001))(x)
        x1 = tf.keras.layers.Dropout(0.4)(x1)
        x1 = tf.keras.layers.BatchNormalization()(x1)
        
        # Second hidden block
        x2 = tf.keras.layers.Dense(128, activation='swish',
                                 kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.001, l2=0.001))(x1)
        x2 = tf.keras.layers.Dropout(0.3)(x2)
        x2 = tf.keras.layers.BatchNormalization()(x2)
        
        # Third hidden block
        x3 = tf.keras.layers.Dense(64, activation='swish',
                                 kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.001, l2=0.001))(x2)
        x3 = tf.keras.layers.Dropout(0.2)(x3)
        
        # Attention mechanism
        attention = tf.keras.layers.Dense(64, activation='softmax')(x3)
        x_att = tf.keras.layers.Multiply()([x3, attention])
        
        # Output layer
        outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x_att)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        # Custom optimizer with weight decay
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=0.001,
            weight_decay=0.0001
        )
        
        # Compile with focal loss for class imbalance
        def focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0):
            y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
            pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
            loss = -tf.reduce_mean(alpha * tf.pow(1.0 - pt, gamma) * tf.math.log(pt))
            return loss
        
        model.compile(
            optimizer=optimizer,
            loss=focal_loss,
            metrics=[
                'accuracy',
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.AUC(name='auc')
            ]
        )
        
        model.summary()
        return model
    
    def train_models(self, X, y, test_size=0.2, validation_split=0.2):
        """Train both TensorFlow and Decision Tree models with enhanced techniques"""
        print("\n=== Splitting Data ===")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
        
        # Handle class imbalance with SMOTE and undersampling
        print("\n=== Handling Class Imbalance ===")
        # Inside cross_validate() method:
        over = SMOTE(sampling_strategy=0.8, random_state=42)  # Changed from 0.5 → 0.8
        under = RandomUnderSampler(sampling_strategy=1.0, random_state=42)  # Changed from 0.8 → 1.0
        steps = [('o', SMOTE(random_state=42))]  # Only SMOTE, no undersampling
        pipeline = ImbPipeline(steps=steps)
        X_train_res, y_train_res = pipeline.fit_resample(X_train, y_train)
        
        # Scale features for neural network
        print("\n=== Scaling Features ===")
        X_train_scaled = self.scaler.fit_transform(X_train_res)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Build and train TensorFlow model
        self.tf_model = self.build_tensorflow_model(X_train.shape[1])
        
        # Enhanced callbacks for training
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_auc', patience=20, restore_best_weights=True, mode='max'
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_auc', factor=0.5, patience=10, min_lr=1e-6, mode='max'
            ),
            tf.keras.callbacks.ModelCheckpoint(
                'best_model.h5', save_best_only=True, monitor='val_auc', mode='max'
            )
        ]
        
        print("\n=== Training TensorFlow Model ===")
        start_time = time.time()
        
        self.history = self.tf_model.fit(
            X_train_scaled, y_train_res,
            epochs=150,
            batch_size=64,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1,
            class_weight=self.class_weights
        )
        
        print(f"\nTraining completed in {(time.time()-start_time)/60:.2f} minutes")
        
        # Optimize decision threshold
        self.optimize_threshold(X_train_scaled, y_train_res)
        
        # Train Decision Tree for comparison
        print("\n=== Training Decision Tree Model ===")
        self.tree_model = DecisionTreeClassifier(
            class_weight='balanced',
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        self.tree_model.fit(X_train_res, y_train_res)
        
        return X_train_scaled, X_test_scaled, X_test, y_train, y_test
    
    def optimize_threshold(self, X_val, y_val):
        """Optimize the decision threshold using validation data"""
        print("\n=== Optimizing Decision Threshold ===")
        y_proba = self.tf_model.predict(X_val).flatten()
        
        # Find best threshold that maximizes F1 score
        thresholds = np.linspace(0.1, 0.9, 50)
        best_threshold = 0.5
        best_f1 = 0
        
        for threshold in thresholds:
            y_pred = (y_proba > threshold).astype(int)
            f1 = metrics.f1_score(y_val, y_pred)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
                
        self.best_threshold = best_threshold
        print(f"Optimal threshold: {best_threshold:.3f} (F1={best_f1:.3f})")
    
    def evaluate_models(self, X_test_scaled, X_test, y_test):
        """Enhanced model evaluation with more metrics and analysis"""
        print("\n=== Enhanced Model Evaluation ===")
        
        # TensorFlow predictions
        tf_pred_proba = self.tf_model.predict(X_test_scaled).flatten()
        tf_pred = (tf_pred_proba > self.best_threshold).astype(int)
        
        # Decision Tree predictions
        dt_pred = self.tree_model.predict(X_test)
        dt_pred_proba = self.tree_model.predict_proba(X_test)[:, 1]
        
        # Print comprehensive evaluation metrics
        print("\n--- TensorFlow Neural Network ---")
        print(f"Using threshold: {self.best_threshold:.3f}")
        print("Accuracy:", round(metrics.accuracy_score(y_test, tf_pred) * 100, 2), "%")
        print("Precision:", round(metrics.precision_score(y_test, tf_pred) * 100, 2), "%")
        print("Recall:", round(metrics.recall_score(y_test, tf_pred) * 100, 2), "%")
        print("F1 Score:", round(metrics.f1_score(y_test, tf_pred) * 100, 2), "%")
        print("ROC AUC:", round(metrics.roc_auc_score(y_test, tf_pred_proba) * 100, 2), "%")
        print("\nClassification Report:")
        print(classification_report(y_test, tf_pred, target_names=['Legitimate', 'Phishing']))
        
        print("\n--- Decision Tree ---")
        print("Accuracy:", round(metrics.accuracy_score(y_test, dt_pred) * 100, 2), "%")
        print("Precision:", round(metrics.precision_score(y_test, dt_pred) * 100, 2), "%")
        print("Recall:", round(metrics.recall_score(y_test, dt_pred) * 100, 2), "%")
        print("F1 Score:", round(metrics.f1_score(y_test, dt_pred) * 100, 2), "%")
        print("ROC AUC:", round(metrics.roc_auc_score(y_test, dt_pred_proba) * 100, 2), "%")
        print("\nClassification Report:")
        print(classification_report(y_test, dt_pred, target_names=['Legitimate', 'Phishing']))
        
        # Plot training history
        self.plot_training_history()
        
        # Plot confusion matrices
        self.plot_confusion_matrices(y_test, tf_pred, dt_pred)
        
        # Plot ROC curves
        self.plot_roc_curves(y_test, tf_pred_proba, dt_pred_proba)
        
        # Feature importance analysis
        self.analyze_feature_importance(X_test, y_test)
        
        # Error analysis
        self.error_analysis(X_test, y_test, tf_pred_proba, dt_pred_proba)
        
        return tf_pred, dt_pred, tf_pred_proba, dt_pred_proba
    
    def analyze_feature_importance(self, X_test, y_test):
        """Analyze and visualize feature importance"""
        print("\n=== Feature Importance Analysis ===")
        
        # SHAP analysis for TensorFlow model
        print("\n--- SHAP Values for TensorFlow Model ---")
        try:
            explainer = shap.DeepExplainer(self.tf_model, 
                                         self.scaler.transform(X_test[:100]))  # Use subset for speed
            shap_values = explainer.shap_values(self.scaler.transform(X_test[:100]))
            shap.summary_plot(shap_values[0], X_test[:100], feature_names=self.feature_columns)
            plt.savefig('shap_summary.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("SHAP summary plot saved as shap_summary.png")
        except Exception as e:
            print(f"Could not compute SHAP values: {e}")
        
        # Permutation importance for Decision Tree
        print("\n--- Permutation Importance for Decision Tree ---")
        result = permutation_importance(
            self.tree_model, X_test, y_test, n_repeats=10, random_state=42
        )
        
        # Sort features by importance
        sorted_idx = result.importances_mean.argsort()[::-1]
        
        plt.figure(figsize=(10, 6))
        plt.boxplot(
            result.importances[sorted_idx].T,
            vert=False, labels=np.array(self.feature_columns)[sorted_idx]
        )
        plt.title("Permutation Importance (Decision Tree)")
        plt.tight_layout()
        plt.savefig('permutation_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Permutation importance plot saved as permutation_importance.png")
    
    def error_analysis(self, X_test, y_test, tf_proba, dt_proba):
        """Analyze misclassified examples"""
        print("\n=== Error Analysis ===")
        
        # Get misclassified examples
        tf_pred = (tf_proba > self.best_threshold).astype(int)
        dt_pred = (dt_proba > 0.5).astype(int)
        
        tf_errors = X_test[y_test != tf_pred]
        dt_errors = X_test[y_test != dt_pred]
        
        print(f"\nTensorFlow misclassified {len(tf_errors)} samples:")
        print("False positives:", sum((y_test == 0) & (tf_pred == 1)))
        print("False negatives:", sum((y_test == 1) & (tf_pred == 0)))
        
        print(f"\nDecision Tree misclassified {len(dt_errors)} samples:")
        print("False positives:", sum((y_test == 0) & (dt_pred == 1)))
        print("False negatives:", sum((y_test == 1) & (dt_pred == 0)))
        
        # Analyze feature distributions in errors
        if not tf_errors.empty:
            plt.figure(figsize=(12, 6))
            for i, feature in enumerate(self.feature_columns[:4]):  # First 4 features
                plt.subplot(2, 2, i+1)
                sns.kdeplot(X_test[feature], label='All')
                sns.kdeplot(tf_errors[feature], label='TF Errors')
                plt.title(feature)
                plt.legend()
            plt.suptitle("Feature Distributions: All vs TensorFlow Errors")
            plt.tight_layout()
            plt.savefig('error_analysis_tf.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("TensorFlow error analysis plot saved as error_analysis_tf.png")
    
    def cross_validate(self, X, y, n_splits=5):
        """Enhanced cross-validation with more metrics"""
        print(f"\n=== {n_splits}-Fold Cross Validation ===")
        kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        metrics_history = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'auc': []
        }
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y), 1):
            print(f"\n--- Fold {fold}/{n_splits} ---")
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Handle class imbalance
            over = SMOTE(sampling_strategy=0.5, random_state=42)
            under = RandomUnderSampler(sampling_strategy=0.8, random_state=42)
            steps = [('o', SMOTE(random_state=42))]  # Only SMOTE, no undersampling
            pipeline = ImbPipeline(steps=steps)
            X_train_res, y_train_res = pipeline.fit_resample(X_train, y_train)
            
            # Scale features
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train_res)
            X_val_scaled = scaler.transform(X_val)
            
            # Build and train model
            model = self.build_tensorflow_model(X_train.shape[1])
            
            model.fit(
                X_train_scaled, y_train_res,
                epochs=100,
                batch_size=64,
                validation_data=(X_val_scaled, y_val),
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(
                        patience=15, restore_best_weights=True, monitor='val_auc', mode='max'
                    )
                ],
                verbose=0
            )
            
            # Evaluate
            y_proba = model.predict(X_val_scaled).flatten()
            y_pred = (y_proba > 0.5).astype(int)
            
            # Store metrics
            metrics_history['accuracy'].append(metrics.accuracy_score(y_val, y_pred))
            metrics_history['precision'].append(metrics.precision_score(y_val, y_pred))
            metrics_history['recall'].append(metrics.recall_score(y_val, y_pred))
            metrics_history['f1'].append(metrics.f1_score(y_val, y_pred))
            metrics_history['auc'].append(metrics.roc_auc_score(y_val, y_proba))
            
            print(f"Fold {fold} - AUC: {metrics_history['auc'][-1]:.4f}, F1: {metrics_history['f1'][-1]:.4f}")
        
        # Print summary
        print("\n=== Cross-Validation Results ===")
        for metric, values in metrics_history.items():
            print(f"{metric.capitalize()}: {np.mean(values):.3f} ± {np.std(values):.3f}")
        
        return metrics_history
    
    def plot_training_history(self):
        """Enhanced training history visualization"""
        if self.history is None:
            return
            
        metrics_to_plot = ['loss', 'accuracy', 'precision', 'recall', 'auc']
        
        fig, axes = plt.subplots(len(metrics_to_plot), 1, figsize=(10, 15))
        
        for i, metric in enumerate(metrics_to_plot):
            axes[i].plot(self.history.history[metric], label=f'Training {metric}')
            if f'val_{metric}' in self.history.history:
                axes[i].plot(self.history.history[f'val_{metric}'], label=f'Validation {metric}')
            axes[i].set_title(metric.capitalize())
            axes[i].set_xlabel('Epoch')
            axes[i].set_ylabel(metric)
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('training_history_enhanced.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Training history saved as training_history_enhanced.png")
    
    def plot_confusion_matrices(self, y_test, tf_pred, dt_pred):
        """Enhanced confusion matrices with percentages"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Function to plot confusion matrix with percentages
        def plot_confusion_matrix(cm, ax, title):
            cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
            sns.heatmap(cm, annot=cm_percent, fmt='.1f%%', cmap='Blues', ax=ax)
            ax.set_title(title)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_xticklabels(['Legitimate', 'Phishing'])
            ax.set_yticklabels(['Legitimate', 'Phishing'])
        
        # TensorFlow confusion matrix
        tf_cm = confusion_matrix(y_test, tf_pred)
        plot_confusion_matrix(tf_cm, axes[0], 'TensorFlow Neural Network')
        
        # Decision Tree confusion matrix
        dt_cm = confusion_matrix(y_test, dt_pred)
        plot_confusion_matrix(dt_cm, axes[1], 'Decision Tree')
        
        plt.tight_layout()
        plt.savefig('confusion_matrices_percentage.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Confusion matrices saved as confusion_matrices_percentage.png")
    
    def plot_roc_curves(self, y_test, tf_proba, dt_proba):
        """Enhanced ROC curves with more metrics"""
        plt.figure(figsize=(10, 8))
        
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
        
        # Add precision-recall curve inset
        ax2 = plt.axes([0.5, 0.2, 0.4, 0.4])
        tf_precision, tf_recall, _ = metrics.precision_recall_curve(y_test, tf_proba)
        dt_precision, dt_recall, _ = metrics.precision_recall_curve(y_test, dt_proba)
        ax2.plot(tf_recall, tf_precision, label=f'TF (AP={metrics.average_precision_score(y_test, tf_proba):.2f})')
        ax2.plot(dt_recall, dt_precision, label=f'DT (AP={metrics.average_precision_score(y_test, dt_proba):.2f})')
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title('Precision-Recall Curve')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('roc_curves_enhanced.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("ROC curves saved as roc_curves_enhanced.png")
    
    def predict_url(self, features, threshold=None, use_ensemble=True):
        """Enhanced prediction with confidence intervals"""
        if threshold is None:
            threshold = self.best_threshold
            
        # Ensure features are in correct order
        if isinstance(features, dict):
            # Fill missing features with median values
            for col in self.feature_columns:
                if col not in features:
                    features[col] = 0  # In production, we might want to use median values
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
            # Enhanced ensemble with confidence estimation
            ensemble_proba = 0.7 * tf_proba + 0.3 * dt_proba
            prediction = int(ensemble_proba > threshold)
            
            # Confidence level based on distance from threshold
            confidence_level = abs(ensemble_proba - threshold) * 2  # Scale to 0-1 range
            
            return {
                'prediction': 'Phishing' if prediction else 'Legitimate',
                'confidence': confidence_level,
                'confidence_level': self._get_confidence_level(confidence_level),
                'tensorflow_prob': tf_proba,
                'decision_tree_prob': dt_proba,
                'ensemble_prob': ensemble_proba,
                'threshold_used': threshold
            }
        else:
            tf_prediction = int(tf_proba > threshold)
            confidence_level = abs(tf_proba - threshold) * 2
            
            return {
                'prediction': 'Phishing' if tf_prediction else 'Legitimate',
                'confidence': confidence_level,
                'confidence_level': self._get_confidence_level(confidence_level),
                'tensorflow_prob': tf_proba,
                'decision_tree_prob': dt_proba,
                'threshold_used': threshold
            }
    
    def _get_confidence_level(self, confidence_score):
        """Convert numerical confidence to qualitative level"""
        if confidence_score > 0.8:
            return "Very High"
        elif confidence_score > 0.6:
            return "High"
        elif confidence_score > 0.4:
            return "Medium"
        else:
            return "Low"
    
    def save_models(self, prefix='phishing_detector'):
        """Save all trained components"""
        # Save TensorFlow model
        self.tf_model.save(f'{prefix}_tensorflow_model.h5')
        
        # Save Decision Tree model
        joblib.dump(self.tree_model, f'{prefix}_decision_tree.pkl')
        
        # Save scaler and feature columns
        joblib.dump(self.scaler, f'{prefix}_scaler.pkl')
        joblib.dump(self.feature_columns, f'{prefix}_features.pkl')
        
        # Save feature selector
        if self.feature_selector is not None:
            joblib.dump(self.feature_selector, f'{prefix}_feature_selector.pkl')
        
        # Save threshold
        with open(f'{prefix}_threshold.pkl', 'wb') as f:
            joblib.dump(self.best_threshold, f)
        
        print(f"All models and components saved with prefix: {prefix}")
    
    def load_models(self, prefix='phishing_detector'):
        """Load all trained components"""
        try:
            self.tf_model = tf.keras.models.load_model(f'{prefix}_tensorflow_model.h5')
            self.tree_model = joblib.load(f'{prefix}_decision_tree.pkl')
            self.scaler = joblib.load(f'{prefix}_scaler.pkl')
            self.feature_columns = joblib.load(f'{prefix}_features.pkl')
            
            # Load feature selector if exists
            try:
                self.feature_selector = joblib.load(f'{prefix}_feature_selector.pkl')
            except:
                self.feature_selector = None
                
            # Load threshold if exists
            try:
                self.best_threshold = joblib.load(f'{prefix}_threshold.pkl')
            except:
                self.best_threshold = 0.5
                
            print(f"All models loaded successfully from prefix: {prefix}")
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
    
    # Perform cross-validation
    cv_results = detector.cross_validate(X, y, n_splits=5)
    
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
    
    result = detector.predict_url(example_features, threshold=None, use_ensemble=True)
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence_level']} ({result['confidence']:.3f})")
    print(f"Ensemble Probability: {result['ensemble_prob']:.3f}")
    print(f"TensorFlow Probability: {result['tensorflow_prob']:.3f}")
    print(f"Decision Tree Probability: {result['decision_tree_prob']:.3f}")
    print(f"Threshold Used: {result['threshold_used']:.3f}")