# Generating a PDF document with the response in a readable format
from fpdf import FPDF

# Initialize PDF document
pdf = FPDF()
pdf.add_page()

# Title
pdf.set_font("Arial", "B", 16)
pdf.cell(0, 10, "Analysis of Linear Regression Output", ln=True, align="C")
pdf.ln(10)

# Section (a)
pdf.set_font("Arial", "B", 14)
pdf.cell(0, 10, "(a) Identification of Statistically Significant and Insignificant Features", ln=True)
pdf.set_font("Arial", "", 12)
pdf.multi_cell(0, 10, """
To determine significance, we examine the P-values of each feature with a threshold of P < 0.05.
Features with P-values below 0.05 are considered statistically significant.

- Significant Features: Intercept (P = 0.000), Wealth (P = 0.000)
- Insignificant Features: Region [T,E] (P = 0.117), Region [T,N] (P = 0.283), Region [T,S] (P = 0.165),
  Region [T,W] (P = 0.603), Literacy (P = 0.232)
""")
pdf.ln(5)

# Section (b)
pdf.set_font("Arial", "B", 14)
pdf.cell(0, 10, "(b) Regression Equation with All Features", ln=True)
pdf.set_font("Arial", "", 12)
pdf.multi_cell(0, 10, """
The regression equation with all features included is:

Lottery = 38.6517 + (-15.4278 * Region [T,E]) + (-10.0170 * Region [T,N]) + (-4.5483 * Region [T,S])
         + (-10.0913 * Region [T,W]) + (-8.1858 * Literacy) + (0.4515 * Wealth)
""")
pdf.ln(5)

# Section (c)
pdf.set_font("Arial", "B", 14)
pdf.cell(0, 10, "(c) Calculation of Target Variable Value", ln=True)
pdf.set_font("Arial", "", 12)
pdf.multi_cell(0, 10, """
Using the equation above and given values for the features (Region [T,E] = 0, Region [T,N] = 0, Region [T,S] = 1,
Region [T,W] = 0, Literacy = 10, and Wealth = 20):

Lottery = 38.6517 + (-15.4278 * 0) + (-10.0170 * 0) + (-4.5483 * 1) + (-10.0913 * 0) + (-8.1858 * 10) + (0.4515 * 20)

= 38.6517 - 4.5483 - 81.858 + 9.03 = -38.7246

The predicted value of Lottery is approximately -38.72.
""")
pdf.ln(5)

# Section (d)
pdf.set_font("Arial", "B", 14)
pdf.cell(0, 10, "(d) Approach to Improve the R^2 Value", ln=True)
pdf.set_font("Arial", "", 12)
pdf.multi_cell(0, 10, """
To improve the R^2 value, consider these strategies:

1. Feature Engineering: Add or transform features to better capture variance.
2. Add Non-linear Terms: Use polynomial terms or feature interactions.
3. Data Collection: Increase the data size or quality to improve model performance.
4. Model Selection: Try more complex models (e.g., decision trees, random forests, ensemble methods).

Each approach may increase the R^2 value by capturing more variance in the data.
""")

# Save PDF
pdf_output_path = "Linear_Regression_Analysis.pdf"
pdf.output(pdf_output_path)

