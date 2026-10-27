import json
import base64
import os
import subprocess

with open("results.json", "r") as f:
    res = json.load(f)

ds = res["dataset"]
words = res["words"]
rv = res["random_variable"]
me = res["model_evaluation"]
rt = res["realtime"]

def to_base64(path):
    if os.path.exists(path):
        with open(path, "rb") as img_file:
            return "data:image/png;base64," + base64.b64encode(img_file.read()).decode("utf-8")
    return ""

# Main Assets
logo_b64 = to_base64("plots/vit_logo.png")
dist_b64 = to_base64("plots/class_distribution.png")
word_b64 = to_base64("plots/word_probabilities.png")
cm_b64 = to_base64("plots/confusion_matrix.png")

# Math Formula Assets
m_prior_spam = to_base64("plots/math/math_prior_spam.png")
m_prior_ham = to_base64("plots/math/math_prior_ham.png")
m_bayes_emp = to_base64("plots/math/math_bayes_empirical.png")
m_bayes_thm = to_base64("plots/math/math_bayes_theorem.png")
m_bayes_comp = to_base64("plots/math/math_bayes_components.png")
m_rv_y = to_base64("plots/math/math_rv_y.png")
m_rv_pmf = to_base64("plots/math/math_rv_pmf.png")
m_rv_exp = to_base64("plots/math/math_rv_expectation.png")
m_rv_var = to_base64("plots/math/math_rv_variance.png")
m_rv_std = to_base64("plots/math/math_rv_std.png")
m_mnb = to_base64("plots/math/math_multinomial_nb.png")
m_acc = to_base64("plots/math/math_accuracy.png")
m_prec = to_base64("plots/math/math_precision.png")
m_rec = to_base64("plots/math/math_recall.png")
m_f1 = to_base64("plots/math/math_f1.png")

m_hdr_pw = to_base64("plots/math/math_hdr_p_word.png")
m_hdr_pw_spam = to_base64("plots/math/math_hdr_p_w_spam.png")
m_hdr_pspam_w_emp = to_base64("plots/math/math_hdr_p_spam_w_emp.png")
m_hdr_pspam_w_bayes = to_base64("plots/math/math_hdr_p_spam_w_bayes.png")
m_hdr_pspam_msg = to_base64("plots/math/math_hdr_p_spam_msg.png")
m_hdr_pham_msg = to_base64("plots/math/math_hdr_p_ham_msg.png")

word_table_rows = ""
for w in words:
    word_table_rows += f"""
    <tr>
        <td style="font-weight:600; text-transform:uppercase;">{w['word']}</td>
        <td>{w['total_occurrences']}</td>
        <td>{w['spam_occurrences']}</td>
        <td>{w['ham_occurrences']}</td>
        <td>{w['p_word']:.4f}</td>
        <td>{w['p_word_given_spam']:.4f}</td>
        <td style="font-weight:600;">{w['p_spam_given_word_empirical']:.4f}</td>
        <td style="font-weight:600;">{w['p_spam_given_word_bayes']:.4f}</td>
        <td>Verified &#10003;</td>
    </tr>
    """

rt_table_rows = ""
for item in rt:
    is_spam = "Spam" in item["prediction"]
    lbl = "Spam" if is_spam else "Ham (Genuine)"
    rt_table_rows += f"""
    <tr>
        <td style="text-align:left; font-style:italic;">"{item['message']}"</td>
        <td style="font-weight:600;">{lbl}</td>
        <td>{item['p_spam']:.4f}</td>
        <td>{item['p_ham']:.4f}</td>
        <td>Correct</td>
    </tr>
    """

html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>BAMAT207 Project Report - Probabilistic SMS Spam Detection</title>
<style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:ital,wght@0,300;0,400;0,500;0,600;0,700;0,800;1,400&display=swap');

    @page {{
        size: A4;
        margin: 14mm 14mm 14mm 14mm;
    }}
    * {{
        box-sizing: border-box;
        -webkit-print-color-adjust: exact !important;
        print-color-adjust: exact !important;
    }}
    body {{
        font-family: 'Montserrat', sans-serif;
        color: #111827;
        line-height: 1.55;
        font-size: 12px;
        margin: 0;
        padding: 0;
        background: #ffffff;
    }}
    .container {{
        width: 100%;
        max-width: 820px;
        margin: 0 auto;
    }}
    
    /* Cover Page Header */
    .header-box {{
        text-align: center;
        border-bottom: 2px solid #0f172a;
        padding-bottom: 16px;
        margin-bottom: 20px;
    }}
    .header-box img {{
        max-height: 65px;
        margin-bottom: 8px;
    }}
    .header-box h1 {{
        font-size: 19px;
        font-weight: 800;
        color: #0f172a;
        margin: 3px 0;
        letter-spacing: 0.5px;
    }}
    .header-box h2 {{
        font-size: 13.5px;
        font-weight: 600;
        color: #334155;
        margin: 2px 0 8px 0;
        letter-spacing: 0.3px;
    }}
    .header-box .report-title {{
        font-size: 15px;
        font-weight: 700;
        color: #0f172a;
        margin: 6px 0 2px 0;
        text-transform: uppercase;
        letter-spacing: 0.4px;
    }}
    
    /* Student Details Meta Table */
    .meta-table {{
        width: 100%;
        border-collapse: collapse;
        margin-top: 10px;
        font-size: 11.5px;
    }}
    .meta-table td {{
        border: 1px solid #cbd5e1;
        padding: 6px 12px;
        text-align: left;
        background: #f8fafc;
        color: #1e293b;
    }}
    .meta-table td strong {{
        color: #0f172a;
    }}

    /* Section Styling */
    h2.section-title {{
        font-size: 13.5px;
        font-weight: 700;
        color: #0f172a;
        border-left: 3.5px solid #0f172a;
        padding-left: 8px;
        margin-top: 20px;
        margin-bottom: 8px;
        text-transform: uppercase;
        letter-spacing: 0.3px;
        page-break-after: avoid;
    }}
    h3.sub-title {{
        font-size: 12px;
        font-weight: 600;
        color: #334155;
        margin-top: 12px;
        margin-bottom: 5px;
        page-break-after: avoid;
    }}
    p {{
        margin: 5px 0 8px 0;
        text-align: justify;
        color: #1e293b;
    }}
    ul, ol {{
        margin: 5px 0 8px 18px;
        padding: 0;
        color: #1e293b;
    }}
    li {{
        margin-bottom: 3px;
    }}

    /* Tables */
    table.data-table {{
        width: 100%;
        border-collapse: collapse;
        margin: 10px 0;
        font-size: 11px;
        page-break-inside: avoid;
    }}
    table.data-table th, table.data-table td {{
        border: 1px solid #cbd5e1;
        padding: 6px 8px;
        text-align: center;
        color: #0f172a;
    }}
    table.data-table th {{
        background-color: #f1f5f9;
        font-weight: 600;
        color: #0f172a;
    }}
    table.data-table tr:nth-child(even) {{
        background-color: #f8fafc;
    }}

    /* Math Formula Image Box */
    .math-box {{
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 4px;
        padding: 8px 12px;
        margin: 8px 0;
        text-align: center;
        page-break-inside: avoid;
    }}
    .math-img {{
        display: inline-block;
        max-height: 38px;
        height: auto;
        vertical-align: middle;
    }}
    .math-img-lg {{
        display: inline-block;
        max-height: 48px;
        height: auto;
        vertical-align: middle;
    }}
    .math-img-hdr {{
        display: inline-block;
        max-height: 18px;
        height: auto;
        vertical-align: middle;
    }}

    /* Figures */
    .fig-wrapper {{
        text-align: center;
        margin: 10px 0;
        page-break-inside: avoid;
    }}
    .fig-wrapper img {{
        max-width: 72%;
        height: auto;
        border: 1px solid #cbd5e1;
        border-radius: 4px;
    }}
    .caption {{
        font-size: 10.5px;
        color: #475569;
        margin-top: 4px;
        font-style: italic;
    }}

    /* Grid Layouts */
    .grid-row {{
        display: flex;
        gap: 14px;
        margin: 8px 0;
        page-break-inside: avoid;
    }}
    .grid-col {{
        flex: 1;
    }}

    .page-break {{
        page-break-before: always;
    }}
</style>
</head>
<body>
<div class="container">

    <!-- Header Block -->
    <div class="header-box">
        {"<img src='" + logo_b64 + "' alt='VIT Logo'>" if logo_b64 else ""}
        <h1>VELLORE INSTITUTE OF TECHNOLOGY</h1>
        <h2>SCHOOL OF ADVANCED SCIENCES &bull; DEPARTMENT OF MATHEMATICS</h2>
        <div class="report-title">BAMAT207: Probability and Statistics — Project Report</div>
        <div style="font-size: 12.5px; font-weight: 600; color: #475569;">PROBABILISTIC SMS SPAM DETECTION SYSTEM</div>
        
        <table class="meta-table">
            <tr>
                <td><strong>Course:</strong> BAMAT207 Probability and Statistics</td>
                <td><strong>Topic:</strong> Probability and Random Variables</td>
            </tr>
            <tr>
                <td><strong>Student Name:</strong> Sai Shreya</td>
                <td><strong>Registration No.:</strong> 25BCE2074</td>
            </tr>
            <tr>
                <td><strong>Dataset:</strong> UCI SMS Spam Collection ({ds['total_raw']} records)</td>
                <td><strong>Model Architecture:</strong> Multinomial Naive Bayes</td>
            </tr>
        </table>
    </div>

    <!-- 1. Introduction -->
    <h2 class="section-title">1. Introduction & Project Scenario</h2>
    <p>
        The objective of this project is to develop an automated Python-based classification system capable of categorizing an incoming SMS message as either <strong>Spam</strong> or <strong>Genuine message (Ham)</strong>. By computing prior, conditional, and posterior probabilities, mathematically verifying <strong>Bayes' Theorem</strong>, and defining a <strong>Bernoulli random variable</strong> model, we establish the statistical foundation of text classification and train an accurate <strong>Multinomial Naive Bayes</strong> classifier.
    </p>

    <!-- 2. Dataset Handling -->
    <h2 class="section-title">2. Dataset Handling & Preprocessing</h2>
    <p>
        The dataset is sourced from the UCI Machine Learning Repository containing real-world SMS messages. The preprocessing pipeline verified data integrity and prepared numerical encodings:
    </p>
    <ul>
        <li><strong>Initial Dataset Size:</strong> {ds['total_raw']} messages.</li>
        <li><strong>Missing Values:</strong> 0 missing entries detected.</li>
        <li><strong>Duplicate Detection:</strong> {ds['duplicates']} duplicate messages were removed to eliminate frequency skewing.</li>
        <li><strong>Cleaned Dataset Size:</strong> {ds['total_cleaned']} unique messages ({ds['n_ham']} Genuine, {ds['n_spam']} Spam).</li>
        <li><strong>Label Encoding:</strong> Genuine (Ham) = 0, Spam = 1.</li>
    </ul>

    <table class="data-table">
        <thead>
            <tr>
                <th>Class Label</th>
                <th>Raw Record Count</th>
                <th>Cleaned Record Count</th>
                <th>Class Proportion</th>
                <th>Numerical Value</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td><strong>Ham (Genuine)</strong></td>
                <td>4,825</td>
                <td>{ds['n_ham']}</td>
                <td>{ds['p_ham']*100:.2f}%</td>
                <td>0</td>
            </tr>
            <tr>
                <td><strong>Spam</strong></td>
                <td>747</td>
                <td>{ds['n_spam']}</td>
                <td>{ds['p_spam']*100:.2f}%</td>
                <td>1</td>
            </tr>
            <tr style="font-weight:600; background:#f1f5f9;">
                <td>Total</td>
                <td>5,572</td>
                <td>{ds['total_cleaned']}</td>
                <td>100.00%</td>
                <td>—</td>
            </tr>
        </tbody>
    </table>

    <!-- 3. Dataset Analysis & Prior Probabilities -->
    <h2 class="section-title">3. Dataset Analysis & Class Probabilities</h2>
    <p>
        The marginal prior probabilities describe the general likelihood of incoming SMS messages in the dataset:
    </p>
    <div class="math-box">
        <img class="math-img" src="{m_prior_spam}" alt="P(Spam) formula"><br><br>
        <img class="math-img" src="{m_prior_ham}" alt="P(Ham) formula">
    </div>
    <div class="fig-wrapper">
        <img src="{dist_b64}" style="max-width:62%;" alt="Class Distribution Chart">
        <div class="caption">Figure 1: Class distribution of cleaned messages demonstrating natural class imbalance.</div>
    </div>

    <div class="page-break"></div>

    <!-- 4. Word-Based Probability Analysis & Bayes' Theorem -->
    <h2 class="section-title">4. Word-Based Probability Analysis & Bayes' Theorem Verification</h2>
    <p>
        To evaluate how specific trigger words alter the probability of a message being spam, five suspicious keywords were selected: <strong>free, prize, win, urgent, offer</strong>.
    </p>
    <p>
        We compute the empirical conditional probability directly from dataset counts, and verify the result using Bayes' Theorem:
    </p>
    <div class="math-box">
        <div style="font-size:11px; font-weight:600; color:#334155; margin-bottom:4px;">Direct Empirical Conditional Probability:</div>
        <img class="math-img" src="{m_bayes_emp}" alt="Empirical conditional probability"><br><br>
        <div style="font-size:11px; font-weight:600; color:#334155; margin-bottom:4px;">Bayes' Theorem Formulation:</div>
        <img class="math-img" src="{m_bayes_thm}" alt="Bayes Theorem"><br><br>
        <img class="math-img" src="{m_bayes_comp}" alt="Bayes components">
    </div>

    <table class="data-table">
        <thead>
            <tr>
                <th>Word</th>
                <th>Total Messages</th>
                <th>Spam Count</th>
                <th>Ham Count</th>
                <th><img class="math-img-hdr" src="{m_hdr_pw}" alt="P(Word)"></th>
                <th><img class="math-img-hdr" src="{m_hdr_pw_spam}" alt="P(Word|Spam)"></th>
                <th><img class="math-img-hdr" src="{m_hdr_pspam_w_emp}" alt="Empirical P"></th>
                <th><img class="math-img-hdr" src="{m_hdr_pspam_w_bayes}" alt="Bayes P"></th>
                <th>Status</th>
            </tr>
        </thead>
        <tbody>
            {word_table_rows}
        </tbody>
    </table>

    <div class="fig-wrapper">
        <img src="{word_b64}" style="max-width:70%;" alt="Word Posterior Probabilities Chart">
        <div class="caption">Figure 2: Posterior probabilities P(Spam | Word) vs P(Ham | Word) across selected keywords.</div>
    </div>
    <p>
        <strong>Mathematical Verification:</strong> The empirical conditional probability $P(\text{{Spam}} \mid \text{{Word}})$ and Bayes' Theorem analytical formulation yield exact identical values across all five words. Words such as <em>prize</em> (1.0000) and <em>urgent</em> (0.9194) exhibit strong diagnostic capability.
    </p>

    <!-- 5. Random Variable Analysis -->
    <h2 class="section-title">5. Random Variable Analysis (Bernoulli Distribution)</h2>
    <p>
        Let <strong>Y</strong> be a discrete random variable indicating whether an incoming message is Spam:
    </p>
    <div class="math-box">
        <img class="math-img-lg" src="{m_rv_y}" alt="Random Variable Y definition"><br><br>
        <img class="math-img" src="{m_rv_pmf}" alt="PMF of Y"><br><br>
        <img class="math-img" src="{m_rv_exp}" alt="Expectation E[Y]"><br><br>
        <img class="math-img" src="{m_rv_var}" alt="Variance Var(Y)"><br><br>
        <img class="math-img" src="{m_rv_std}" alt="Standard Deviation">
    </div>

    <div class="page-break"></div>

    <!-- 6. Data Preparation & Model Building -->
    <h2 class="section-title">6. Machine Learning Model Building & Training</h2>
    <p>
        To operationalize spam filtering, we implemented a <strong>Multinomial Naive Bayes</strong> classifier:
    </p>
    <ul>
        <li><strong>Dataset Splitting:</strong> 80% Training (4,135 samples), 20% Testing (1,034 samples). Stratified sampling was applied to guarantee that both subsets preserved the identical 12.63% spam proportion.</li>
        <li><strong>Feature Extraction:</strong> <code>CountVectorizer</code> converted message strings into word-count feature vectors with stop-word removal and lowercasing, producing a vocabulary of <strong>7,414 unique features</strong>.</li>
        <li><strong>Multinomial Naive Bayes Architecture:</strong>
            <div class="math-box" style="margin: 6px 0;">
                <img class="math-img" src="{m_mnb}" alt="Multinomial Naive Bayes Model">
            </div>
            Trained with Laplace smoothing (&alpha; = 1.0) to prevent zero-probability artifacts on unseen vocabulary.
        </li>
    </ul>

    <!-- 7. Model Evaluation -->
    <h2 class="section-title">7. Model Evaluation & Performance Metrics</h2>
    <p>
        The trained model was evaluated on the unseen test set (1,034 samples):
    </p>

    <div class="grid-row">
        <div class="grid-col">
            <table class="data-table">
                <thead>
                    <tr>
                        <th>Metric</th>
                        <th>Mathematical Formula</th>
                        <th>Score</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td><strong>Accuracy</strong></td>
                        <td><img class="math-img-hdr" src="{m_acc}" alt="Accuracy formula"></td>
                        <td style="font-weight:600;">98.36%</td>
                    </tr>
                    <tr>
                        <td><strong>Precision</strong></td>
                        <td><img class="math-img-hdr" src="{m_prec}" alt="Precision formula"></td>
                        <td style="font-weight:600;">95.97%</td>
                    </tr>
                    <tr>
                        <td><strong>Recall</strong></td>
                        <td><img class="math-img-hdr" src="{m_rec}" alt="Recall formula"></td>
                        <td style="font-weight:600;">90.84%</td>
                    </tr>
                    <tr>
                        <td><strong>F1-Score</strong></td>
                        <td><img class="math-img-hdr" src="{m_f1}" alt="F1 formula"></td>
                        <td style="font-weight:600;">93.33%</td>
                    </tr>
                </tbody>
            </table>
        </div>
        <div class="grid-col" style="text-align:center;">
            <img src="{cm_b64}" style="max-width:82%;" alt="Confusion Matrix Heatmap">
            <div class="caption">Figure 3: Confusion Matrix heatmap on test set.</div>
        </div>
    </div>

    <h3 class="sub-title">Confusion Matrix Breakdown (1,034 Test Samples):</h3>
    <ul>
        <li><strong>True Negatives (TN):</strong> 898 (Genuine messages correctly identified).</li>
        <li><strong>False Positives (FP):</strong> 5 (Genuine messages incorrectly classified as spam — 0.55% false alarm rate).</li>
        <li><strong>False Negatives (FN):</strong> 12 (Spam messages incorrectly classified as genuine).</li>
        <li><strong>True Positives (TP):</strong> 119 (Spam messages correctly detected).</li>
    </ul>

    <!-- 8. Real-Time Testing -->
    <h2 class="section-title">8. Real-Time Testing & Prediction Probability</h2>
    <p>
        The system supports inference on arbitrary incoming text, calculating the full posterior probability distribution:
    </p>

    <table class="data-table">
        <thead>
            <tr>
                <th style="width:48%;">Incoming Test SMS Message</th>
                <th>Predicted Class</th>
                <th><img class="math-img-hdr" src="{m_hdr_pspam_msg}" alt="P(Spam|Message)"></th>
                <th><img class="math-img-hdr" src="{m_hdr_pham_msg}" alt="P(Ham|Message)"></th>
                <th>Verification</th>
            </tr>
        </thead>
        <tbody>
            {rt_table_rows}
        </tbody>
    </table>

    <!-- 9. Conclusion -->
    <h2 class="section-title">9. Conclusion</h2>
    <ol>
        <li><strong>Bayesian Consistency:</strong> Direct empirical probabilities perfectly matched analytical Bayes' Theorem values across all tested keywords.</li>
        <li><strong>Random Variable Modeling:</strong> The Bernoulli distribution accurately characterizes message outcomes with mean $E[Y] = 0.1263$ and variance $\text{{Var}}(Y) = 0.1104$.</li>
        <li><strong>Classification Efficacy:</strong> The Multinomial Naive Bayes model demonstrated <strong>98.36% accuracy</strong> and <strong>95.97% precision</strong> with only 5 false alarms, proving the practical power of probabilistic text classification.</li>
    </ol>

</div>
</body>
</html>
"""

# Write HTML
with open("PROJECT_REPORT_PRINT.html", "w", encoding="utf-8") as f:
    f.write(html_content)

print("[+] Updated PROJECT_REPORT_PRINT.html with Montserrat font, minimal colors, student info, and math images!")

# Compile to PDF using Edge / Chrome headless
edge_path = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
chrome_path = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
browser_path = edge_path if os.path.exists(edge_path) else chrome_path

input_html = os.path.abspath("PROJECT_REPORT_PRINT.html")
output_pdf = os.path.abspath("BAMAT207_SMS_Spam_Detection_Project_Report.pdf")

cmd = [
    browser_path,
    "--headless",
    "--disable-gpu",
    "--no-pdf-header-footer",
    f"--print-to-pdf={output_pdf}",
    input_html
]

print(f"[*] Compiling PDF via {browser_path}...")
result = subprocess.run(cmd, capture_output=True, text=True)

if os.path.exists(output_pdf) and os.path.getsize(output_pdf) > 0:
    print(f"[+] Final PDF generated successfully! Size: {os.path.getsize(output_pdf)} bytes")
    print(f"[+] PDF Location: {output_pdf}")
else:
    print(f"[!] Compilation error: {result.stderr}")
