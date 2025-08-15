// JavaScript code for PhishAlert

const form = document.getElementById('threat-analyzer');
const input = document.getElementById('url-input');
const scanner = document.getElementById('scanner');
const scanStatus = document.getElementById('scan-status');
const resultPanel = document.getElementById('result-panel');
const imageAnalysisPanel = document.getElementById('image-analysis-panel');
const analysisOutput = document.getElementById('analysis-output');
const analysisContent = document.getElementById('analysis-content');
const analyzeBtn = document.getElementById('analyze-btn');
const imageOnlyBtn = document.getElementById('image-only-btn');
const copyBtn = document.getElementById('copy-report');
const copyImageBtn = document.getElementById('copy-image-report');
const expandBtn = document.getElementById('expand-report');
const reportTime = document.getElementById('report-time');
const reportUrl = document.getElementById('report-url');
const reportFeatures = document.getElementById('report-features');
const reportHtmlStatus = document.getElementById('report-html-status');

const scanMessages = [
    'Initializing threat detection...',
    'Analyzing URL structure...',
    'Fetching HTML content...',
    'Parsing webpage elements...',
    'Analyzing images and media...',
    'Checking domain reputation...',
    'Validating SSL certificates...',
    'Scanning for malicious patterns...',
    'Cross-referencing threat databases...',
    'Finalizing security assessment...'
];

let scanMessageIndex = 0;
let scanInterval;

// Enhanced scan messages for image-only analysis
const imageScanMessages = [
    'Initializing image analysis...',
    'Fetching webpage content...',
    'Parsing HTML structure...',
    'Extracting image elements...',
    'Analyzing image metadata...',
    'Detecting tracking pixels...',
    'Checking external resources...',
    'Finalizing image report...'
];

form.addEventListener('submit', async (e) => {
    e.preventDefault();
    await performAnalysis('/api/predict', false);
});

imageOnlyBtn.addEventListener('click', async () => {
    await performAnalysis('/api/analyze-images', true);
});

async function performAnalysis(endpoint, isImageOnly = false) {
    const url = input.value.trim();
    if (!url) return;

    // Reset UI
    resultPanel.className = 'result-panel';
    imageAnalysisPanel.className = 'image-analysis-panel';
    analysisOutput.className = 'analysis-output';

    // Start scanning animation
    scanner.classList.add('active');
    analyzeBtn.disabled = true;
    imageOnlyBtn.disabled = true;
    analyzeBtn.textContent = 'Scanning...';

    // Use appropriate scan messages
    const messages = isImageOnly ? imageScanMessages : scanMessages;
    scanMessageIndex = 0;
    scanStatus.textContent = messages[0];
    scanInterval = setInterval(() => {
    scanMessageIndex = (scanMessageIndex + 1) % messages.length;
    scanStatus.textContent = messages[scanMessageIndex];
    }, 1000);

    try {
    const requestBody = { url: url };

    const response = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestBody),
    });

    if (!response.ok) {
        throw new Error(await response.text());
    }

    const data = await response.json();

    // Stop scanning animation
    clearInterval(scanInterval);
    scanner.classList.remove('active');
    analyzeBtn.disabled = false;
    imageOnlyBtn.disabled = false;
    analyzeBtn.textContent = 'Initialize Scan';

    if (isImageOnly) {
        displayImageAnalysis(data);
    } else {
        displayFullAnalysis(data);
    }

    } catch (error) {
    clearInterval(scanInterval);
    scanner.classList.remove('active');
    analyzeBtn.disabled = false;
    imageOnlyBtn.disabled = false;
    analyzeBtn.textContent = 'Initialize Scan';

    resultPanel.innerHTML = `
        <div class="result-header">
        <span>❌</span>
        <span>Uh Oh!</span>
        </div>
        <div class="result-details">
        Failed to analyze URL: Please check the URL format to validate it is correct.
        <br>Details: ${error.message || 'Unknown error occurred.'}
        </div>`;
    resultPanel.classList.add('visible', 'warning');
    console.error('Analysis error:', error);
    }
}

// Paste button
const pasteBtn = document.getElementById('paste-btn');
if (pasteBtn) {
    pasteBtn.addEventListener('click', async () => {
        try {
            const text = await navigator.clipboard.readText();
            if (text) {
                input.value = text;
                // Optional: Animate the field when pasted
                input.classList.add('pasted');
                setTimeout(() => input.classList.remove('pasted'), 500);
            } else {
                alert("Clipboard is empty!");
            }
        } catch (err) {
            alert("Unable to read clipboard. Please allow clipboard permissions.");
            console.error(err);
        }
    });
}


function displayFullAnalysis(data) {
    let prediction, detailsHtml, featuresHtml = '';
    const riskClass = data.risk_level.toLowerCase();

    if (data.prediction === 1) {
    prediction = 'threat';
    detailsHtml = `
        <div class="result-header">
        <span>⚠️</span>
        <span>Threat Detected</span>
        </div>
        <div class="result-details">
        ${riskClass.toUpperCase()}-risk phishing attempt identified with ${data.risk_score}% confidence.
        This URL exhibits multiple malicious patterns and should be avoided. 
        If you believe this is not a phishing scam, proceed with caution.
        </div>
    `;
    } else {
    prediction = 'safe';
    detailsHtml = `
        <div class="result-header">
        <span>✅</span>
        <span>Secure</span>
        </div>
        <div class="result-details">
        URL analysis complete with ${100 - data.risk_score}% safety confidence.
        No significant threats detected. This appears to be a legitimate website.
        </div>
    `;
    }

    // Add prediction details with confidence bar and features grid container
    detailsHtml += `
    <div class="prediction-details">
        <div class="details-header">
        <div class="details-title">Prediction Details</div>
        <div class="confidence-meter">
            <span class="confidence-value">${data.risk_score}% Chance of Phishing </span>
            <div class="confidence-bar" role="progressbar" aria-valuemin="0" aria-valuemax="100" aria-valuenow="${data.risk_score}">
            <div class="confidence-fill ${riskClass}" style="width: ${data.risk_score}%;"></div>
            </div>
        </div>
        </div>
        <div class="feature-indicators" id="feature-indicators"></div>
    </div>
    `;

    resultPanel.innerHTML = detailsHtml;
    resultPanel.classList.add('visible', riskClass);

    // Populate feature indicators grid
    const featureIndicators = document.getElementById('feature-indicators');
    if (data.features) {
    for (const [feature, value] of Object.entries(data.features)) {
        if (feature === 'true_label' || feature === 'predicted_label') continue;
        
        let valueRiskClass = 'neutral';
        if (typeof value === 'number') {
        valueRiskClass = value > 0.7 ? 'threat' : value > 0.3 ? 'warning' : 'safe';
        }

        const featureName = feature
        .replace(/_/g, ' ')
        .replace(/([a-z])([A-Z])/g, '$1 $2')
        .replace(/\b\w/g, (c) => c.toUpperCase());

        featuresHtml += `
        <div class="feature-indicator">
            <div class="feature-name">${featureName}</div>
            <div class="feature-value ${valueRiskClass}">${value}</div>
        </div>`;
    }
    featureIndicators.innerHTML = featuresHtml;
    }

    // Display image analysis if available
    if (data.image_analysis) {
    displayImageAnalysis(data, false);
    }

    // Fill detailed report metadata and content
    reportTime.textContent = data.timestamp;
    reportUrl.textContent = data.url;
    reportFeatures.textContent = data.features ? Object.keys(data.features).length : 'N/A';
    reportHtmlStatus.textContent = data.html_analyzed ? 'Yes' : 'No';
    analysisContent.innerHTML = `<pre style="font-family: var(--font-mono); font-size: 0.85em; overflow-x: auto; max-height: 300px; white-space: pre-wrap;">${JSON.stringify(data, null, 2)}</pre>`;

    analysisOutput.classList.add('visible');
}

function displayImageAnalysis(data, standalone = true) {
    const imageData = data.image_analysis;
    if (!imageData) return;

    const imageStatsGrid = document.getElementById('image-stats-grid');
    const imageDetailsGrid = document.getElementById('image-details-grid');

    // Create image statistics overview
    imageStatsGrid.innerHTML = `
    <div class="stat-card">
        <div class="stat-icon">🖼️</div>
        <div class="stat-content">
        <div class="stat-value">${imageData.NoOfImage}</div>
        <div class="stat-label">Total Images</div>
        </div>
    </div>
    <div class="stat-card ${imageData.external_images > 0 ? 'warning' : ''}">
        <div class="stat-icon">🌐</div>
        <div class="stat-content">
        <div class="stat-value">${imageData.external_images}</div>
        <div class="stat-label">External Images</div>
        </div>
    </div>
    <div class="stat-card ${imageData.suspicious_images > 0 ? 'threat' : ''}">
        <div class="stat-icon">⚠️</div>
        <div class="stat-content">
        <div class="stat-value">${imageData.suspicious_images}</div>
        <div class="stat-label">Suspicious Images</div>
        </div>
    </div>
    <div class="stat-card ${imageData.broken_images > 0 ? 'warning' : ''}">
        <div class="stat-icon">❌</div>
        <div class="stat-content">
        <div class="stat-value">${imageData.broken_images}</div>
        <div class="stat-label">Broken Images</div>
        </div>
    </div>
    <div class="stat-card">
        <div class="stat-icon">📊</div>
        <div class="stat-content">
        <div class="stat-value">${formatBytes(imageData.total_image_size)}</div>
        <div class="stat-label">Total Size</div>
        </div>
    </div>
    `;

    // Create detailed image grid
    if (imageData.image_details && imageData.image_details.length > 0) {
    let imageDetailsHtml = '';
    imageData.image_details.slice(0, 12).forEach((img, index) => {
        const statusClass = img.is_suspicious ? 'suspicious' : img.is_external ? 'external' : 'normal';
        const statusIcon = img.is_suspicious ? '⚠️' : img.is_external ? '🌐' : '✅';
        
        imageDetailsHtml += `
        <div class="image-detail-card ${statusClass}">
            <div class="image-header">
            <span class="image-index">#${index + 1}</span>
            <span class="image-status">${statusIcon}</span>
            </div>
            <div class="image-info">
            <div class="image-src" title="${img.src}">${truncateText(img.src, 40)}</div>
            <div class="image-meta">
                ${img.alt ? `<div class="meta-item"><strong>Alt:</strong> ${truncateText(img.alt, 30)}</div>` : ''}
                ${img.size ? `<div class="meta-item"><strong>Size:</strong> ${formatBytes(img.size)}</div>` : ''}
                ${img.format ? `<div class="meta-item"><strong>Format:</strong> ${img.format}</div>` : ''}
                <div class="meta-item"><strong>External:</strong> ${img.is_external ? 'Yes' : 'No'}</div>
                <div class="meta-item"><strong>Suspicious:</strong> ${img.is_suspicious ? 'Yes' : 'No'}</div>
            </div>
            </div>
        </div>
        `;
    });
    imageDetailsGrid.innerHTML = imageDetailsHtml;
    } else {
    imageDetailsGrid.innerHTML = '<div class="no-images">No images found in the webpage</div>';
    }

    imageAnalysisPanel.classList.add('visible');

    if (standalone) {
    // For image-only analysis, also update the report metadata
    reportTime.textContent = data.timestamp;
    reportUrl.textContent = data.url;
    reportFeatures.textContent = 'Image Analysis Only';
    reportHtmlStatus.textContent = 'Yes';
    analysisContent.textContent = JSON.stringify(data, null, 2);
    analysisOutput.classList.add('visible');
    }
}

function formatBytes(bytes) {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
}

function truncateText(text, maxLength) {
    return text.length > maxLength ? text.substring(0, maxLength) + '...' : text;
}

// Copy report to clipboard
copyBtn.addEventListener('click', () => {
    const reportContent = analysisContent.textContent;
    navigator.clipboard.writeText(reportContent).then(() => {
    const originalTitle = copyBtn.getAttribute('title');
    copyBtn.setAttribute('title', 'Copied!');
    setTimeout(() => {
        copyBtn.setAttribute('title', originalTitle);
    }, 2000);
    });
});

// Copy image report to clipboard
copyImageBtn.addEventListener('click', () => {
    const imageStatsText = document.getElementById('image-stats-grid').textContent;
    const imageDetailsText = document.getElementById('image-details-grid').textContent;
    const fullReport = `Image Analysis Report\n\n${imageStatsText}\n\nImage Details:\n${imageDetailsText}`;
    
    navigator.clipboard.writeText(fullReport).then(() => {
    const originalTitle = copyImageBtn.getAttribute('title');
    copyImageBtn.setAttribute('title', 'Copied!');
    setTimeout(() => {
        copyImageBtn.setAttribute('title', originalTitle);
    }, 2000);
    });
});

// Expand/collapse detailed JSON report
expandBtn.addEventListener('click', () => {
    const expanded = analysisOutput.classList.toggle('expanded');
    expandBtn.setAttribute('aria-expanded', expanded);
    expandBtn.querySelector('svg').style.transform = expanded
    ? 'rotate(180deg)'
    : 'rotate(0deg)';
});