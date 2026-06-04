# credit-risk-model
## Credit Scoring Business Understanding

### 1. Basel II and the Need for Interpretability

The Basel II Accord emphasizes accurate risk measurement and regulatory transparency in financial institutions. This creates a strong requirement for models that are both interpretable and well-documented.

Financial institutions must be able to explain how a model arrives at its decisions, especially when determining whether to approve or reject a loan. Regulators require clear justification of risk predictions to ensure fairness, accountability, and compliance.

As a result:

* Black-box models are less preferred unless supported with explainability tools.
* Documentation of model assumptions, features, and performance is essential.
* Auditable and reproducible workflows are required.

Thus, Basel II pushes organizations toward models that are not only accurate but also transparent and explainable.

### 2. Proxy Variables for Default Prediction

In many real-world datasets, a direct "default" label (i.e., whether a customer failed to repay a loan) may not be available. In such cases, proxy variables are used to approximate default behavior.

Examples of proxy variables:

* Number of missed payments
* Days past due (DPD)
* Account delinquency status

Why proxy variables are necessary:

* Historical default data may be incomplete or unavailable.
* Some institutions track behavior differently.
* Data collection limitations.

However, using proxy variables introduces risks:

* The proxy may not perfectly represent true default behavior.
* It can lead to biased or inaccurate predictions.
* Misclassification may result in financial losses or unfair decisions.

Therefore, careful selection and validation of proxy variables is critical.

### 3. Trade-offs: Interpretable vs High-Performance Models

* **Simple/Interpretable Models (Logistic Regression + WoE):** Highly favored in regulated spaces. They offer transparent, linear log-odds coefficients and smooth scorecard binnings that are easy to explain to compliance auditors, though they may miss complex non-linear feature interactions.
* **High-Performance Models (Gradient Boosting/XGBoost):** Achieve superior predictive accuracy and lower loss rates by optimizing non-linear patterns. However, they lack direct interpretability, requiring secondary explanation frameworks (like SHAP values) which are harder to defend under strict regulatory scrutiny.
#### Interpretable Models (e.g., Logistic Regression with WoE)

Advantages:

* Easy to understand and explain
* Preferred by regulators
* Transparent feature contributions
* Easier to debug and validate

Disadvantages:

* May have lower predictive accuracy
* Limited ability to capture complex patterns

#### High-Performance Models (e.g., Gradient Boosting)

Advantages:

* Higher accuracy and better predictive power
* Captures nonlinear relationships
* Handles complex feature interactions

Disadvantages:

* Less interpretable (black-box nature)
* Harder to justify decisions to regulators
* Requires additional explainability tools (e.g., SHAP, LIME)

### Conclusion

In a regulated financial environment, the choice of model must balance performance with interpretability. While advanced models can improve accuracy, simpler models are often preferred due to their transparency and compliance with regulatory requirements like Basel II.
