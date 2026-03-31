# Two Europes, Two Mechanisms: Regional Heterogeneity in the Poverty–Crime Relationship (EU27)

This repository contains the full analytical pipeline for a panel data study on the socio-economic determinants of crime across the European Union (2014–2022). The study tests the competing and complementary roles of **Marxian (material deprivation)** and **Weberian (institutional/anomie)** mechanisms.

## 📌 Project Overview
The research explores why poverty and institutional quality affect crime differently across the EU. The main finding identifies a clear divergence:
* **Central and Eastern Europe (ECE):** Dominated by the **Marxian mechanism** (poverty and long-term unemployment).
* **Western & Northern Europe (WEST):** Dominated by the **Weberian mechanism** (Rule of Law and institutional quality).

## 🛠️ Repository Structure
The project is organized into a sequential pipeline:

- `01_collect_data.py`: Automated data extraction from **Eurostat API** and **World Bank (WGI)**.
- `02_clean_impute.py`: Data cleaning, outlier detection, and K-Nearest Neighbors (KNN) imputation for missing values.
- `03_stationarity.py`: Unit root testing (IPS Test) and stationarity checks via ACF plots.
- `04_panel_fe.py`: Estimation of **Fixed Effects (FE) Models** with Driscoll-Kraay robust standard errors.
- `05_tables.py`: Automated generation of publication-ready tables and results summary.
- `/data`: Contains raw, processed datasets and regression outputs.
- `/plots`: Visualizations (Correlation matrices, ACF plots, and Coefficient charts).

## 📊 Key Findings
The study validates four main hypotheses:
1.  **H1 (Marxian):** Material deprivation is a significant predictor of crime in the full EU27 sample.
2.  **H2 (Weberian):** Institutional quality (Rule of Law) is a primary predictor, specifically for property crime.
3.  **H3 (Heterogeneity):** There is a structural divergence; Rule of Law is only significant in the WEST subsample.
4.  **H4 (Crime Type):** Unemployment has a stronger effect on property crime (instrumental) than on homicide (expressive).

## 🚀 How to Run
1. **Clone the repo:**
   ```bash
   git clone [https://github.com/your-username/two-europes-crime.git](https://github.com/your-username/two-europes-crime.git)
