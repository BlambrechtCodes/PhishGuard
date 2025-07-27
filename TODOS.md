## PhishGuard Machine Learning Project - TODO List

This TODO list tracks remaining and upcoming tasks for the PhishGuard project as of July 26, 2025.  

**Completed tasks** are noted for context.

### ✅ Completed Tasks

- [x] Created the GitHub repository
- [x] Built the initial enhanced TensorFlow + Decision Tree ML pipeline (`phish1_MAIN.py`)
- [x] Performed data quality analysis and feature engineering
- [x] Performed model training, evaluation, and comparison
- [x] Implemented ROC/confusion matrix plotting and training history visualization
- [x] Ran cross-validation, saved models, and performed sample prediction

### 🚧 Main TODOs

#### 1. UI, Application, & Format Research
- [ ] **Finalize application/UI architecture**
    - Choose: Web app, desktop GUI, CLI, browser extension, or hybrid
    - Define component responsibilities (frontend, backend, model serving)
    - **Research browser extension formatting (Chrome, Edge, Firefox)**
        - Analyze architecture (manifest.json, popup scripts, backend APIs) and data flow for ML-powered phishing extensions[5][6][8]
        - Understand integration approaches (feature extraction in extension or via backend)
    - **Research official website creation**
        - Compare: Single-page website, interactive demo hosting, documentation best practices
        - Review similar open-source Phishing Detector project sites for content and structure

#### 2. Model Input Pipeline & Accuracy Verification
- [ ] **Implement feature translation logic:**  
    - _Input should be translated from raw link (URL string) to the required structured feature list (as used in training)_
    - Ensure browser or web app can extract necessary features in real-time[5][6][8]
    - Validate that UI never passes full URLs directly to the model, but always a feature dict/list
- [ ] **Verify accuracy end-to-end**
    - Build a test pipeline to check that predictions on the same URL via app match offline script results
    - Set up automated regression tests using labeled dataset samples via the UI

#### 3. Model & Engineering Enhancement
- [ ] Refactor model code for reusability and clarity (modularize as needed)
- [ ] Optionally add additional feature extraction for URLs (expand feature set)
- [ ] Implement feature selection/importance visualization for transparency
- [ ] Tune thresholds and model settings for best real-world performance
- [ ] Consider adding continuous model evaluation (monitor drift on real user data)

#### 4. Documentation & Collaboration
- [ ] Document the code thoroughly (class/method docstrings, inline comments)
- [ ] Update README with usage instructions and screenshots of the current UI
- [ ] Create a CONTRIBUTING.md for new developers
- [ ] Track open issues in GitHub Issues
- [ ] (Optional) Record short demo video for stakeholders


## 🎤 Presentation Planning Section (Max 10 minutes)

**To include in our Canva presentation:**

- **Project Motivation & Overview**
    - Dangers of phishing and limitations of existing solutions
    - Our project goal: Combining state-of-the-art ML with practical deployment
- **Technical Architecture** (make the complexity visible/impressive)
    - End-to-end ML pipeline (preprocessing, feature extraction, model training: neural net + tree, ensemble techniques)
    - Real-time feature translation system: Turning URLs into feature vectors, not trivial string matching
    - Cross-platform compatibility (API, UI, and possible browser extension)
- **Key Engineering Accomplishments**
    - Ensemble model with live ROC, confusion matrix, and performance monitoring
    - Automated real-time feature engineering based on browser/webpage context
    - Model persistence, cross-validation, reproducibility, and UI/UX plans
- **Challenges & Solutions**
    - Issues with data leakage and robust validation
    - Ensuring predictions are explainable (feature importances)
    - Testing for accuracy across interfaces (CLI, UI, API)
- **Demonstration (Live or Video)**
    - Show interactive workflow:
        - URL input → Feature extraction → Model prediction → User feedback
        - Highlight explanation or output confidence, not just “Phishing/Legitimate” label
        - Visuals: ROC/confusion matrix, real-time results, errors & edge cases
    - (If extension or web demo is ready:) Quick run-through as an end user
- **Roadmap & Impact**
    - Extension format/website plan and future scalability
    - Real-world applications in business, education, cybersecurity

### Keep this file pruned and up-to-date! Thanks!
