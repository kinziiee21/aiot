import pandas as pd
from typing import Dict, Any, Tuple, List
from modules.data_cleaning import DataCleaning

class CleaningAgent:
    """Agent responsible for dataset assessment, quality recommendations, and data cleaning"""

    def __init__(self):
        self.cleaner = DataCleaning()

    def assess_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform comprehensive data quality assessment"""
        if df is None or df.empty:
            return {
                "missing_count": 0,
                "duplicate_count": 0,
                "type_issues": 0,
                "numeric_string_cols": [],
                "possible_date_cols": [],
                "quality_score": 100
            }

        info = self.cleaner.assess_data_quality(df)
        
        # Calculate dataset summary metrics
        num_cols = df.select_dtypes(include=['number']).columns.tolist()
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        dt_cols = df.select_dtypes(include=['datetime', 'datetime64']).columns.tolist()

        info['numeric_cols_count'] = len(num_cols)
        info['categorical_cols_count'] = len(cat_cols)
        info['date_cols_count'] = len(dt_cols) + len(info.get('possible_date_cols', []))
        
        # Calculate overall quality score (0 - 100)
        total_cells = df.size if df.size > 0 else 1
        missing_cells = info['missing_count']
        dup_rows = info['duplicate_count']
        
        comp_score = max(0, 100 - (missing_cells / total_cells * 100))
        uniq_score = max(0, 100 - (dup_rows / len(df) * 100)) if len(df) > 0 else 100
        type_score = max(0, 100 - (info['type_issues'] * 15))

        info['quality_score'] = round(comp_score * 0.4 + uniq_score * 0.3 + type_score * 0.3, 1)
        return info

    def get_recommendations(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Get actionable cleaning recommendations based on data assessment"""
        assessment = self.assess_quality(df)
        recommendations = []

        if assessment['missing_count'] > 0:
            recommendations.append({
                "action": "handle_missing",
                "label": "Handle Missing Values",
                "detail": f"Detected {assessment['missing_count']} missing cells across {len(assessment['columns_with_missing'])} columns.",
                "recommended": True
            })

        if assessment['duplicate_count'] > 0:
            recommendations.append({
                "action": "remove_duplicates",
                "label": "Remove Duplicate Rows",
                "detail": f"Detected {assessment['duplicate_count']} duplicate rows.",
                "recommended": True
            })

        if assessment.get('type_issues', 0) > 0:
            num_str_cols = ", ".join(assessment.get('numeric_string_cols', []))
            recommendations.append({
                "action": "fix_data_types",
                "label": "Convert Numeric Strings to Numbers",
                "detail": f"Detected formatted numeric strings in columns: {num_str_cols}.",
                "recommended": True
            })

        if assessment.get('possible_date_cols'):
            date_cols = ", ".join(assessment.get('possible_date_cols', []))
            recommendations.append({
                "action": "standardize_dates",
                "label": "Standardize Date Formats",
                "detail": f"Detected possible date columns: {date_cols}.",
                "recommended": True
            })

        text_cols = df.select_dtypes(include=['object']).columns.tolist()
        if text_cols:
            recommendations.append({
                "action": "clean_whitespace",
                "label": "Clean Whitespace in Text",
                "detail": f"Strip extra spaces in {len(text_cols)} text columns.",
                "recommended": True
            })

        recommendations.append({
            "action": "remove_outliers",
            "label": "Remove Numerical Outliers",
            "detail": "Filter out extreme outliers using the Interquartile Range (IQR) method.",
            "recommended": False
        })

        return recommendations

    def clean(self, df: pd.DataFrame, options: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """Apply selected cleaning operations to a copy of the dataframe"""
        return self.cleaner.clean_data(df, options)
