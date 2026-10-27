import os
import matplotlib.pyplot as plt

MATH_DIR = "plots/math"
os.makedirs(MATH_DIR, exist_ok=True)

def render_equation(latex_code, output_file, fontsize=13, dpi=300):
    fig = plt.figure(figsize=(0.1, 0.1), dpi=dpi)
    fig.patch.set_facecolor('white')
    plt.axis('off')
    plt.text(0.5, 0.5, f"${latex_code}$", fontsize=fontsize, ha='center', va='center')
    plt.savefig(output_file, bbox_inches='tight', pad_inches=0.06, dpi=dpi, facecolor='white')
    plt.close()

equations = {
    "math_prior_spam.png": r"P(\mathrm{Spam}) = \frac{\mathrm{Number\ of\ Spam\ Messages}}{\mathrm{Total\ Messages}} = \frac{653}{5169} = 0.1263",
    "math_prior_ham.png": r"P(\mathrm{Ham}) = \frac{\mathrm{Number\ of\ Ham\ Messages}}{\mathrm{Total\ Messages}} = \frac{4516}{5169} = 0.8737",
    "math_bayes_empirical.png": r"P(\mathrm{Spam} \mid \mathrm{Word}) = \frac{\mathrm{Number\ of\ Spam\ messages\ containing\ Word}}{\mathrm{Total\ messages\ containing\ Word}}",
    "math_bayes_theorem.png": r"P(\mathrm{Spam} \mid \mathrm{Word}) = \frac{P(\mathrm{Word} \mid \mathrm{Spam}) \cdot P(\mathrm{Spam})}{P(\mathrm{Word})}",
    "math_bayes_components.png": r"P(\mathrm{Word}) = \frac{\mathrm{Count}(\mathrm{Word})}{N}, \quad P(\mathrm{Word} \mid \mathrm{Spam}) = \frac{\mathrm{Count}(\mathrm{Word} \cap \mathrm{Spam})}{N_{\mathrm{Spam}}}, \quad P(\mathrm{Spam}) = \frac{N_{\mathrm{Spam}}}{N}",
    "math_rv_pmf.png": r"P(Y = y) = p^y (1 - p)^{1 - y}, \quad y \in \{0, 1\}, \quad p = P(\mathrm{Spam}) = 0.1263",
    "math_rv_expectation.png": r"E[Y] = \sum_{y} y \cdot P(Y = y) = 0 \cdot (1 - p) + 1 \cdot p = p = 0.1263",
    "math_rv_variance.png": r"\mathrm{Var}(Y) = E[Y^2] - (E[Y])^2 = p - p^2 = p(1 - p) = 0.1263 \times (1 - 0.1263) = 0.1104",
    "math_rv_std.png": r"\sigma_Y = \sqrt{\mathrm{Var}(Y)} = \sqrt{0.1104} = 0.3322",
    "math_multinomial_nb.png": r"P(Y = c \mid \mathbf{x}) \propto P(Y = c) \prod_{j=1}^{D} P(w_j \mid Y = c)^{x_j}",
    "math_accuracy.png": r"\mathrm{Accuracy} = \frac{\mathrm{TP} + \mathrm{TN}}{\mathrm{Total}} = \frac{119 + 898}{1034} = 98.36\%",
    "math_precision.png": r"\mathrm{Precision} = \frac{\mathrm{TP}}{\mathrm{TP} + \mathrm{FP}} = \frac{119}{119 + 5} = 95.97\%",
    "math_recall.png": r"\mathrm{Recall} = \frac{\mathrm{TP}}{\mathrm{TP} + \mathrm{FN}} = \frac{119}{119 + 12} = 90.84\%",
    "math_f1.png": r"\mathrm{F1\text{-}Score} = 2 \times \frac{\mathrm{Precision} \times \mathrm{Recall}}{\mathrm{Precision} + \mathrm{Recall}} = 93.33\%",
    "math_hdr_p_word.png": r"P(\mathrm{Word})",
    "math_hdr_p_w_spam.png": r"P(\mathrm{Word} \mid \mathrm{Spam})",
    "math_hdr_p_spam_w_emp.png": r"P_{\mathrm{Emp}}(\mathrm{Spam} \mid \mathrm{Word})",
    "math_hdr_p_spam_w_bayes.png": r"P_{\mathrm{Bayes}}(\mathrm{Spam} \mid \mathrm{Word})",
    "math_hdr_p_spam_msg.png": r"P(\mathrm{Spam} \mid \mathrm{Message})",
    "math_hdr_p_ham_msg.png": r"P(\mathrm{Ham} \mid \mathrm{Message})"
}

for name, code in equations.items():
    filepath = os.path.join(MATH_DIR, name)
    render_equation(code, filepath, fontsize=11 if "hdr" in name else 13)
    print(f"Rendered {name}")

# Piecewise Y
fig = plt.figure(figsize=(4.5, 0.9), dpi=300)
plt.axis('off')
plt.text(0.05, 0.72, r'$Y = 1\quad (\mathrm{if\ the\ incoming\ message\ is\ Spam})$', fontsize=12)
plt.text(0.05, 0.22, r'$Y = 0\quad (\mathrm{if\ the\ incoming\ message\ is\ Ham\ /\ Genuine})$', fontsize=12)
plt.savefig(os.path.join(MATH_DIR, 'math_rv_y.png'), bbox_inches='tight', pad_inches=0.08, dpi=300, facecolor='white')
plt.close()
print("Rendered math_rv_y.png")

print("[+] All equations rendered successfully!")
