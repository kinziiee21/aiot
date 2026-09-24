import pandas as pd
import numpy as np
from typing import Dict, Any, List
from modules.profiling import DataProfiler
from utils.helpers import get_data_summary

class EDAAgent:
    """Agent responsible for Exploratory Data Analysis (EDA) and profiling"""

    def __init__(self):
        self.profiler = DataProfiler()

    def generate_eda_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate structured Exploratory Data Analysis metrics and summaries"""
        if df is None or df.empty:
            return {}

        basic_stats = self.profiler.get_basic_statistics(df)
        helper_summary = get_data_summary(df)

        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        date_cols = df.select_dtypes(include=['datetime', 'datetime64']).columns.tolist()

        # Correlation matrix for numeric columns
        corr_dict = {}
        if len(num_cols) >= 2:
            corr_matrix = df[num_cols].corr().round(3)
            corr_dict = corr_matrix.to_dict()

        # Outlier counts per numeric column
        outliers_summary = {}
        for col in num_cols:
            col_data = df[col].dropna()
            if len(col_data) > 0:
                q1 = col_data.quantile(0.25)
                q3 = col_data.quantile(0.75)
                iqr = q3 - q1
                outlier_count = ((col_data < (q1 - 1.5 * iqr)) | (col_data > (q3 + 1.5 * iqr))).sum()
                outliers_summary[col] = {
                    "count": int(outlier_count),
                    "percentage": round(float(outlier_count / len(col_data) * 100), 2)
                }

        # Key dataset patterns
        patterns = []
        if len(num_cols) > 0:
            skewed_cols = [c for c in num_cols if abs(df[c].dropna().skew()) > 1.0]
            if skewed_cols:
                patterns.append(f"Highly skewed numerical distribution detected in: {', '.join(skewed_cols)}")

        if corr_dict:
            # Find strongest correlations (> 0.6)
            strong_corrs = []
            for col1 in num_cols:
                for col2 in num_cols:
                    if col1 < col2:
                        val = corr_dict[col1][col2]
                        if abs(val) >= 0.6:
                            strong_corrs.append(f"{col1} & {col2} (r={val})")
            if strong_corrs:
                patterns.append(f"Strong correlations found: {', '.join(strong_corrs[:3])}")

        if not patterns:
            patterns.append("Dataset has balanced column distributions without extreme correlations.")

        return {
            "basic_stats": basic_stats,
            "helper_summary": helper_summary,
            "numeric_columns": num_cols,
            "categorical_columns": cat_cols,
            "date_columns": date_cols,
            "correlations": corr_dict,
            "outliers": outliers_summary,
            "patterns": patterns
        }

    def generate_html_report(self, df: pd.DataFrame, title: str = "EDA Profiling Report") -> str:
        """Generate ydata-profiling HTML report file path"""
        return self.profiler.generate_profile_report(df, title=title)
