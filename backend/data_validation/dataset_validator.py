# -*- coding: utf-8 -*-
import os
import pandas as pd

def run_validation_checks() -> dict:
    dataset_dir = "AgriSense-Dataset"
    report_file = os.path.join("validation_reports", "dataset_validation_report.html")
    os.makedirs("validation_reports", exist_ok=True)
    
    results = {}
    html_sections = []
    
    # 1. Validate Crop Recommendations
    crop_path = os.path.join(dataset_dir, "consolidated_agriculture_dataset.csv")
    if os.path.exists(crop_path):
        df = pd.read_csv(crop_path)
        df = df[df["source_file"] == "Crop_recommendation.csv"].dropna(how="all", axis=1)
        nulls = int(df.isnull().sum().sum())
        duplicates = int(df.duplicated().sum())
        
        # Expectation checks
        ph_ok = bool((df['ph'] >= 0.0).all() and (df['ph'] <= 14.0).all())
        temp_ok = bool((df['temperature'] >= -20.0).all() and (df['temperature'] <= 60.0).all())
        
        results["crop_recommendation"] = {
            "rows": len(df),
            "columns": list(df.columns),
            "null_count": nulls,
            "duplicate_count": duplicates,
            "ph_validation": "PASSED" if ph_ok else "FAILED",
            "temperature_validation": "PASSED" if temp_ok else "FAILED"
        }
        
        html_sections.append(f"""
        <div class='section'>
            <h2>Crop Recommendation Dataset Quality</h2>
            <p>Rows: {len(df)} | Columns: {len(df.columns)}</p>
            <p>Duplicates: {duplicates} | Null values: {nulls}</p>
            <p>pH expectation [0-14]: <strong>{results["crop_recommendation"]["ph_validation"]}</strong></p>
            <p>Temperature expectation [-20 to 60C]: <strong>{results["crop_recommendation"]["temperature_validation"]}</strong></p>
        </div>
        """)
        
    # Generate HTML report
    html_content = f"""
    <html>
    <head>
        <title>AgriSense Dataset Quality Report</title>
        <style>
            body {{ font-family: sans-serif; background: #f4f6f8; color: #333; padding: 20px; }}
            .section {{ background: white; padding: 20px; margin-bottom: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            h1 {{ color: #16a34a; }}
            strong {{ color: #16a34a; }}
        </style>
    </head>
    <body>
        <h1>🌾 AgriSense Great Expectations Dataset Validation Report</h1>
        <p>Run time: {pd.Timestamp.now()}</p>
        {"".join(html_sections)}
    </body>
    </html>
    """
    
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(html_content)
        
    return results

if __name__ == "__main__":
    run_validation_checks()
