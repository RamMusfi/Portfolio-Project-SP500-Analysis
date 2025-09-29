# Portfolio-Project-SP500-Analysis

## 📌 Project Overview
This project explores **S&P 500 company data** to analyze industry performance, revenue trends, and market capitalization.  
The goal: demonstrate **end-to-end data engineering and analysis skills** using Databricks for ingestion, transformations, and modeling, then visualize in Power BI.

---

## 🛠 Tech Stack
- **Databricks (Azure)** – ingestion 
- **Python (Pandas, PySpark)** – bronze/silver/gold layering, transformations 
- **Power BI** – DAX, final dashboards and visuals  

---

## 🔑 Key Steps
1. **Data Ingestion** – loaded S&P 500 dataset into Databricks Workspace and created a Databricks cluster (`sp500-cluster`)  
2. **Bronze/Silver/Gold Architecture** – standardized and curated data for BI consumption in Jupyter Notebook.
3. **Data Modeling** – built star schema with fact/dimension tables (`fact_company`, `dim_sector`, `dim_financials`, etc.)  
4. **Python Analysis** – cleaned and transformed using Pandas/PySpark  
5. **Dashboard Design** – built Power BI visuals to compare revenue, profit, and sector growth  

---

## 📊 Dashboard Preview
<img width="1763" height="936" alt="image" src="https://github.com/user-attachments/assets/02423a2d-4975-43f9-a20e-f0762cadf9a0" />
