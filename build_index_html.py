import json
import base64
import os

with open("results.json", "r") as f:
    res = json.load(f)

with open("model_export.json", "r") as f:
    model_data = json.load(f)

def to_base64(path):
    if os.path.exists(path):
        with open(path, "rb") as img_file:
            return "data:image/png;base64," + base64.b64encode(img_file.read()).decode("utf-8")
    return ""

dist_b64 = to_base64("plots/class_distribution.png")
cm_b64 = to_base64("plots/confusion_matrix.png")

html_app = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Probabilistic SMS Spam Detection - BAMAT207</title>
<link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
    * {{
        box-sizing: border-box;
        margin: 0;
        padding: 0;
        font-family: 'Montserrat', sans-serif;
    }}
    body {{
        background: #f8fafc;
        color: #1e293b;
        line-height: 1.6;
        padding-bottom: 60px;
    }}
    .navbar {{
        background: #0f172a;
        color: #ffffff;
        padding: 18px 30px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
    }}
    .nav-title h1 {{
        font-size: 18px;
        font-weight: 700;
        letter-spacing: 0.5px;
    }}
    .nav-title p {{
        font-size: 12px;
        color: #94a3b8;
    }}
    .btn-pdf {{
        background: #3b82f6;
        color: #ffffff;
        padding: 8px 16px;
        border-radius: 6px;
        text-decoration: none;
        font-weight: 600;
        font-size: 12px;
        transition: 0.2s;
    }}
    .btn-pdf:hover {{
        background: #2563eb;
    }}
    .container {{
        max-width: 1050px;
        margin: 30px auto;
        padding: 0 20px;
    }}
    .card {{
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 24px;
        margin-bottom: 25px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }}
    .card h2 {{
        font-size: 16px;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 15px;
        border-left: 4px solid #0f172a;
        padding-left: 10px;
        text-transform: uppercase;
    }}
    .grid-2 {{
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 20px;
    }}
    .grid-4 {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 15px;
    }}
    .metric-box {{
        background: #f8fafc;
        border: 1px solid #cbd5e1;
        border-radius: 6px;
        padding: 14px;
        text-align: center;
    }}
    .metric-value {{
        font-size: 22px;
        font-weight: 800;
        color: #0f172a;
    }}
    .metric-label {{
        font-size: 11px;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        margin-top: 4px;
    }}
    textarea {{
        width: 100%;
        height: 100px;
        border: 1.5px solid #cbd5e1;
        border-radius: 6px;
        padding: 12px;
        font-size: 13px;
        resize: vertical;
        outline: none;
    }}
    textarea:focus {{
        border-color: #0f172a;
    }}
    .preset-container {{
        margin: 10px 0 16px 0;
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
    }}
    .btn-preset {{
        background: #f1f5f9;
        border: 1px solid #cbd5e1;
        padding: 6px 12px;
        border-radius: 4px;
        font-size: 11px;
        font-weight: 600;
        cursor: pointer;
        transition: 0.15s;
    }}
    .btn-preset:hover {{
        background: #e2e8f0;
    }}
    .btn-classify {{
        background: #0f172a;
        color: #ffffff;
        border: none;
        padding: 10px 22px;
        font-size: 13px;
        font-weight: 700;
        border-radius: 6px;
        cursor: pointer;
        transition: 0.2s;
    }}
    .btn-classify:hover {{
        background: #334155;
    }}
    .result-box {{
        margin-top: 18px;
        padding: 16px;
        border-radius: 6px;
        display: none;
    }}
    .result-spam {{
        background: #fee2e2;
        border: 1px solid #fca5a5;
        color: #991b1b;
    }}
    .result-ham {{
        background: #dcfce7;
        border: 1px solid #86efac;
        color: #166534;
    }}
    table {{
        width: 100%;
        border-collapse: collapse;
        font-size: 12px;
        margin: 12px 0;
    }}
    th, td {{
        border: 1px solid #cbd5e1;
        padding: 8px 10px;
        text-align: center;
    }}
    th {{
        background: #f1f5f9;
        font-weight: 600;
        color: #0f172a;
    }}
    img.chart-img {{
        width: 100%;
        max-width: 480px;
        border: 1px solid #cbd5e1;
        border-radius: 6px;
        display: block;
        margin: 10px auto;
    }}
</style>
</head>
<body>

<div class="navbar">
    <div class="nav-title">
        <h1>📱 Probabilistic SMS Spam Detection</h1>
        <p>BAMAT207: Probability and Statistics &bull; VIT | Sai Shreya (25BCE2074)</p>
    </div>
    <a href="BAMAT207_SMS_Spam_Detection_Project_Report.pdf" class="btn-pdf" download>📥 Download Project Report (PDF)</a>
</div>

<div class="container">

    <!-- Activity 8: Real-Time Live Testing -->
    <div class="card">
        <h2>Activity 8: Real-Time SMS Message Classification</h2>
        <p style="font-size: 12.5px; color: #475569; margin-bottom: 10px;">
            Test the trained <strong>Multinomial Naive Bayes</strong> classifier live. Click any sample message or type your own text below:
        </p>

        <div class="preset-container">
            <button class="btn-preset" onclick="setPreset('Congratulations! You have won a free prize. Claim now.')">Sample 1: 'Won free prize'</button>
            <button class="btn-preset" onclick="setPreset('Hey bro, are we meeting at the campus library this evening?')">Sample 2: 'Campus meeting'</button>
            <button class="btn-preset" onclick="setPreset('URGENT! Your mobile number won 1000 prize. Call now to claim offer.')">Sample 3: 'Urgent prize offer'</button>
            <button class="btn-preset" onclick="setPreset('Can you send me the lecture notes for probability and statistics?')">Sample 4: 'Lecture notes'</button>
        </div>

        <textarea id="smsInput" placeholder="Type or paste an SMS message here..."></textarea>
        
        <div style="margin-top: 12px;">
            <button class="btn-classify" onclick="classifyText()">Run Naive Bayes Inference</button>
        </div>

        <div id="resultBox" class="result-box">
            <h3 id="resultLabel" style="font-size: 16px; font-weight: 800; margin-bottom: 6px;"></h3>
            <p id="resultProb" style="font-size: 13px; font-weight: 600;"></p>
        </div>
    </div>

    <!-- Metrics Summary -->
    <div class="card">
        <h2>Model Performance & Evaluation Metrics</h2>
        <div class="grid-4">
            <div class="metric-box">
                <div class="metric-value">98.36%</div>
                <div class="metric-label">Accuracy</div>
            </div>
            <div class="metric-box">
                <div class="metric-value">95.97%</div>
                <div class="metric-label">Precision</div>
            </div>
            <div class="metric-box">
                <div class="metric-value">90.84%</div>
                <div class="metric-label">Recall</div>
            </div>
            <div class="metric-box">
                <div class="metric-value">93.33%</div>
                <div class="metric-label">F1-Score</div>
            </div>
        </div>
        
        <div class="grid-2" style="margin-top: 20px;">
            <div>
                <img src="{cm_b64}" class="chart-img" alt="Confusion Matrix">
            </div>
            <div style="font-size: 12.5px; padding: 10px;">
                <h4 style="font-size: 14px; font-weight: 700; margin-bottom: 8px;">Confusion Matrix (1,034 Test Samples):</h4>
                <ul style="margin-left: 20px; line-height: 1.8;">
                    <li><strong>True Negatives (TN):</strong> 898 (Genuine correctly classified)</li>
                    <li><strong>False Positives (FP):</strong> 5 (Genuine misclassified as Spam &mdash; 0.55% false alarm rate)</li>
                    <li><strong>False Negatives (FN):</strong> 12 (Spam misclassified as Genuine)</li>
                    <li><strong>True Positives (TP):</strong> 119 (Spam correctly blocked)</li>
                </ul>
            </div>
        </div>
    </div>

    <!-- Probability Analysis -->
    <div class="card">
        <h2>Activities 2, 3 & 4: Probability & Random Variable Modeling</h2>
        <div class="grid-2">
            <div>
                <div class="metric-box" style="margin-bottom: 12px; text-align: left;">
                    <strong>Prior Probabilities:</strong><br>
                    &bull; P(Spam) = 653 / 5169 = <strong>0.1263 (12.63%)</strong><br>
                    &bull; P(Ham) = 4516 / 5169 = <strong>0.8737 (87.37%)</strong>
                </div>
                <div class="metric-box" style="text-align: left;">
                    <strong>Bernoulli Random Variable Y ~ Bern(p):</strong><br>
                    &bull; Expectation E[Y] = p = <strong>0.1263</strong><br>
                    &bull; Variance Var(Y) = p(1 - p) = <strong>0.1104</strong><br>
                    &bull; Standard Deviation &sigma; = <strong>0.3322</strong>
                </div>
            </div>
            <div>
                <img src="{dist_b64}" class="chart-img" alt="Class Distribution">
            </div>
        </div>

        <h3 style="font-size: 14px; font-weight: 700; margin: 20px 0 10px 0;">Bayes' Theorem Verification for Suspicious Keywords:</h3>
        <table>
            <thead>
                <tr>
                    <th>Keyword</th>
                    <th>Total Occurrences</th>
                    <th>Spam Count</th>
                    <th>P(Word)</th>
                    <th>Empirical P(Spam|Word)</th>
                    <th>Bayes' P(Spam|Word)</th>
                    <th>Verification</th>
                </tr>
            </thead>
            <tbody>
                <tr><td><strong>FREE</strong></td><td>203</td><td>148</td><td>0.0393</td><td>0.7291</td><td>0.7291</td><td>Match &#10003;</td></tr>
                <tr><td><strong>PRIZE</strong></td><td>73</td><td>73</td><td>0.0141</td><td>1.0000</td><td>1.0000</td><td>Match &#10003;</td></tr>
                <tr><td><strong>WIN</strong></td><td>55</td><td>49</td><td>0.0106</td><td>0.8909</td><td>0.8909</td><td>Match &#10003;</td></tr>
                <tr><td><strong>URGENT</strong></td><td>62</td><td>57</td><td>0.0120</td><td>0.9194</td><td>0.9194</td><td>Match &#10003;</td></tr>
                <tr><td><strong>OFFER</strong></td><td>30</td><td>24</td><td>0.0058</td><td>0.8000</td><td>0.8000</td><td>Match &#10003;</td></tr>
            </tbody>
        </table>
    </div>

</div>

<!-- Client-side Inference Script -->
<script>
const modelData = {json.dumps(model_data)};

function setPreset(txt) {{
    document.getElementById("smsInput").value = txt;
    classifyText();
}}

function classifyText() {{
    const text = document.getElementById("smsInput").value.trim().toLowerCase();
    if (!text) return;

    // Tokenize
    const tokens = text.match(/\\b[a-z]{{2,}}\\b/g) || [];
    
    let logSpam = modelData.log_prior_spam;
    let logHam = modelData.log_prior_ham;

    for (const token of tokens) {{
        if (token in modelData.vocabulary) {{
            const idx = modelData.vocabulary[token];
            logHam += modelData.log_prob_ham[idx];
            logSpam += modelData.log_prob_spam[idx];
        }}
    }}

    // Softmax normalization
    const maxLog = Math.max(logHam, logSpam);
    const expHam = Math.exp(logHam - maxLog);
    const expSpam = Math.exp(logSpam - maxLog);
    const pSpam = expSpam / (expHam + expSpam);
    const pHam = expHam / (expHam + expSpam);

    const isSpam = pSpam >= 0.5;
    const box = document.getElementById("resultBox");
    const lbl = document.getElementById("resultLabel");
    const prob = document.getElementById("resultProb");

    box.style.display = "block";
    if (isSpam) {{
        box.className = "result-box result-spam";
        lbl.innerText = "🚨 Prediction: SPAM MESSAGE";
        prob.innerText = "P(Spam | Message) = " + (pSpam * 100).toFixed(2) + "% | P(Ham | Message) = " + (pHam * 100).toFixed(2) + "%";
    }} else {{
        box.className = "result-box result-ham";
        lbl.innerText = "✅ Prediction: GENUINE MESSAGE (HAM)";
        prob.innerText = "P(Ham | Message) = " + (pHam * 100).toFixed(2) + "% | P(Spam | Message) = " + (pSpam * 100).toFixed(2) + "%";
    }}
}}

window.onload = function() {{
    setPreset("Congratulations! You have won a free prize. Claim now.");
}};
</script>

</body>
</html>
"""

with open("index.html", "w", encoding="utf-8") as f:
    f.write(html_app)

print("[+] index.html successfully compiled!")
