import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime

class ReportAgent:
    """Agent responsible for producing structured analysis reports from dataset stats & Q&A history"""

    def generate_report(
        self,
        df: pd.DataFrame,
        quality_info: Dict[str, Any],
        qa_history: List[Dict[str, Any]],
        dataset_name: str = "Uploaded Dataset"
    ) -> str:
        """Generate a complete Markdown analysis report"""
        if df is None or df.empty:
            return "# Data Analysis Report\n\nNo dataset uploaded."

        rows, cols = df.shape
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        dt_cols = df.select_dtypes(include=['datetime', 'datetime64']).columns.tolist()

        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        report_lines = [
            f"# 📊 Data Analysis Report: {dataset_name}",
            f"**Generated on:** {now_str} | **Application:** Natural Language Data Analyst",
            "\n---",
            "## 1. Executive Summary & Dataset Overview",
            f"- **Total Rows:** {rows:,}",
            f"- **Total Columns:** {cols}",
            f"- **Numerical Columns ({len(num_cols)}):** {', '.join(num_cols[:5]) if num_cols else 'None'}",
            f"- **Categorical Columns ({len(cat_cols)}):** {', '.join(cat_cols[:5]) if cat_cols else 'None'}",
            f"- **Date Columns ({len(dt_cols)}):** {', '.join(dt_cols) if dt_cols else 'None'}",
            "\n---",
            "## 2. Data Quality & Health Assessment",
            f"- **Quality Score:** {quality_info.get('quality_score', 'N/A')}/100",
            f"- **Missing Cells Count:** {quality_info.get('missing_count', 0):,} ({quality_info.get('missing_percentage', 0):.2f}%)",
            f"- **Duplicate Rows Count:** {quality_info.get('duplicate_count', 0):,} ({quality_info.get('duplicate_percentage', 0):.2f}%)",
            f"- **Data Type Issues / Numeric Strings:** {quality_info.get('type_issues', 0)} column(s)",
            "\n---",
            "## 3. Important Descriptive Statistics",
        ]

        if num_cols:
            stats_df = df[num_cols].describe().T[['mean', 'std', 'min', '50%', 'max']]
            stats_df.columns = ['Mean', 'Std Dev', 'Min', 'Median', 'Max']
            report_lines.append(stats_df.to_markdown())
        else:
            report_lines.append("No numerical columns available for statistics.")

        report_lines.extend([
            "\n---",
            "## 4. Key Questions & AI Analysis Findings"
        ])

        if qa_history:
            for idx, item in enumerate(qa_history, 1):
                question = item.get("question", "")
                answer = item.get("answer", "")
                insights = item.get("insights", "")

                report_lines.append(f"### Q{idx}: {question}")
                if answer:
                    report_lines.append(f"**Result:** {answer}")
                if insights:
                    report_lines.append(f"**Insights:**\n{insights}\n")
        else:
            report_lines.append("No natural language queries performed yet.")

        report_lines.extend([
            "\n---",
            "## 5. Patterns, Trends & Recommendations",
            "- **Data Cleaning Recommendation:** Handle missing values and verify column data types prior to modeling.",
            "- **Exploratory Recommendation:** Continue exploring relationships between categorical segments and top numerical metrics.",
            "- **Reporting Note:** All computed figures in this report are based strictly on verifiable Pandas DataFrame operations.",
            "\n---",
            "*Report generated automatically by Natural Language Data Analyst.*"
        ])

        return "\n".join(report_lines)
