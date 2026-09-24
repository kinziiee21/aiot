import pandas as pd
import numpy as np
import re
from typing import Dict, Any, Tuple, List, Optional
from sklearn.preprocessing import StandardScaler

class DataCleaning:
    """Comprehensive data cleaning and preprocessing module"""
    
    def __init__(self):
        self.cleaning_methods = {
            'handle_missing': self._handle_missing_values,
            'remove_duplicates': self._remove_duplicates,
            'standardize_dates': self._standardize_dates,
            'clean_whitespace': self._clean_whitespace,
            'fix_data_types': self._fix_data_types,
            'remove_outliers': self._remove_outliers
        }
    
    @staticmethod
    def try_parse_datetime_series(series: pd.Series) -> Tuple[bool, Optional[pd.Series]]:
        """
        Detect and parse date/datetime values in a Series even if formats vary or are mixed.
        Returns (is_date_col, parsed_series_or_None).
        """
        if series.empty or pd.api.types.is_datetime64_any_dtype(series):
            return True, series
        if pd.api.types.is_bool_dtype(series):
            return False, None
        if pd.api.types.is_numeric_dtype(series):
            col_name = str(series.name).lower() if series.name else ""
            if not any(kw in col_name for kw in ['date', 'timestamp', 'dob', 'time_stamp']):
                return False, None

        s_clean = series.dropna().astype(str).str.strip()
        if len(s_clean) == 0:
            return False, None

        sample = s_clean.head(100)

        def has_date_structure(v: str) -> bool:
            v_str = v.strip()
            if not v_str:
                return False
            if re.search(r'[-/.\s,T]', v_str):
                return True
            if len(v_str) == 8 and v_str.isdigit() and (v_str.startswith('19') or v_str.startswith('20')):
                return True
            return False

        trait_matches = sum(has_date_structure(v) for v in sample)
        col_name = str(series.name).lower() if series.name else ""
        has_date_kw = any(kw in col_name for kw in ['date', 'time', 'dob', 'created', 'updated', 'timestamp', 'day', 'month', 'year', 'dt'])

        if (trait_matches / len(sample)) < 0.35 and not has_date_kw:
            return False, None

        try:
            parsed_sample = pd.to_datetime(sample, errors='coerce', format='mixed')
        except (TypeError, ValueError):
            try:
                parsed_sample = pd.to_datetime(sample, errors='coerce')
            except Exception:
                return False, None

        valid_count = parsed_sample.notnull().sum()
        if valid_count / len(sample) >= 0.5:
            valid_years = parsed_sample.dt.year.dropna()
            if not valid_years.empty and ((valid_years >= 1700) & (valid_years <= 2100)).mean() >= 0.7:
                try:
                    full_parsed = pd.to_datetime(series, errors='coerce', format='mixed')
                except (TypeError, ValueError):
                    full_parsed = pd.to_datetime(series, errors='coerce')
                return True, full_parsed

        return False, None

    @staticmethod
    def clean_numeric_string_series(series: pd.Series) -> pd.Series:
        """Parse numeric values from strings formatted with currency, commas, percentages, accounting parens"""
        def clean_val(v):
            if pd.isna(v) or v is None:
                return np.nan
            s = str(v).strip()
            if not s or s.lower() in ['nan', 'null', 'none', 'n/a', '-', 'na', '']:
                return np.nan
            if s.startswith('(') and s.endswith(')'):
                s = '-' + s[1:-1]
            s = re.sub(r'(?i)(Rs\.?|INR|USD|EUR|GBP|AUD|CAD|JPY)', '', s)
            s = re.sub(r'(?i)Rs\.?', '', s)
            s = re.sub(r'[\$₹€£¥,%]', '', s)
            m = re.search(r'[-+]?\d+(?:\.\d+)?', s)
            if m:
                try:
                    return float(m.group(0))
                except ValueError:
                    return np.nan
            return np.nan

        return series.apply(clean_val)

    @classmethod
    def try_parse_numeric_series(cls, series: pd.Series) -> Tuple[bool, Optional[pd.Series]]:
        """
        Detect if an object/string series represents numerical data and return converted numeric series.
        """
        if series.empty or pd.api.types.is_numeric_dtype(series):
            return True, series
        if pd.api.types.is_bool_dtype(series) or pd.api.types.is_datetime64_any_dtype(series):
            return False, None

        s_clean = series.dropna().astype(str).str.strip()
        if len(s_clean) == 0:
            return False, None

        sample = s_clean.head(100)
        parsed_sample = cls.clean_numeric_string_series(sample)
        valid_count = parsed_sample.notnull().sum()

        if valid_count / len(sample) >= 0.5:
            full_parsed = cls.clean_numeric_string_series(series)
            return True, full_parsed

        return False, None

    def assess_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess data quality and return quality metrics"""
        quality_info = {
            'missing_count': df.isnull().sum().sum(),
            'missing_percentage': (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100 if df.size > 0 else 0,
            'duplicate_count': df.duplicated().sum(),
            'duplicate_percentage': (df.duplicated().sum() / df.shape[0]) * 100 if len(df) > 0 else 0,
            'type_issues': 0,
            'numeric_string_cols': [],
            'possible_date_cols': [],
            'columns_with_missing': df.columns[df.isnull().any()].tolist(),
            'data_types': df.dtypes.to_dict(),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024 * 1024)
        }
        
        for col in df.columns:
            if df[col].dtype == 'object':
                is_date, _ = self.try_parse_datetime_series(df[col])
                if is_date:
                    quality_info['possible_date_cols'].append(col)
                    quality_info['type_issues'] += 1
                    continue

                is_num, _ = self.try_parse_numeric_series(df[col])
                if is_num:
                    quality_info['type_issues'] += 1
                    quality_info['numeric_string_cols'].append(col)
                    continue

        return quality_info

    def clean_data(self, df: pd.DataFrame, options: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """Clean data based on specified options"""
        cleaned_df = df.copy()
        cleaning_report = {}
        
        for method_name, should_apply in options.items():
            if should_apply and method_name in self.cleaning_methods:
                try:
                    if method_name == 'handle_missing':
                        cleaned_df, report = self.cleaning_methods[method_name](
                            cleaned_df, options.get('missing_strategy', 'drop')
                        )
                    else:
                        cleaned_df, report = self.cleaning_methods[method_name](cleaned_df)
                    
                    cleaning_report[method_name] = report
                    
                except Exception as e:
                    cleaning_report[method_name] = f"Failed: {str(e)}"
        
        return cleaned_df, cleaning_report

    def _handle_missing_values(self, df: pd.DataFrame, strategy: str = 'drop') -> Tuple[pd.DataFrame, str]:
        """Handle missing values based on strategy"""
        initial_shape = df.shape
        
        if strategy == 'drop':
            df_cleaned = df.dropna()
            report = f"Dropped {initial_shape[0] - df_cleaned.shape[0]} rows with missing values"
            
        elif strategy == 'mean':
            df_cleaned = df.copy()
            numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
            df_cleaned[numeric_cols] = df_cleaned[numeric_cols].fillna(df_cleaned[numeric_cols].mean())
            report = f"Filled missing values with mean for {len(numeric_cols)} numeric columns"
            
        elif strategy == 'median':
            df_cleaned = df.copy()
            numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
            df_cleaned[numeric_cols] = df_cleaned[numeric_cols].fillna(df_cleaned[numeric_cols].median())
            report = f"Filled missing values with median for {len(numeric_cols)} numeric columns"
            
        elif strategy == 'mode':
            df_cleaned = df.copy()
            for col in df_cleaned.columns:
                mode_value = df_cleaned[col].mode()
                if len(mode_value) > 0:
                    df_cleaned[col] = df_cleaned[col].fillna(mode_value[0])
            report = "Filled missing values with mode for all columns"
            
        elif strategy == 'forward_fill':
            df_cleaned = df.ffill()
            report = "Applied forward fill for missing values"
            
        elif strategy == 'backward_fill':
            df_cleaned = df.bfill()
            report = "Applied backward fill for missing values"
            
        else:
            df_cleaned = df.copy()
            report = f"Unknown strategy '{strategy}', no action taken"
        
        return df_cleaned, report

    def _remove_duplicates(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
        """Remove duplicate rows"""
        initial_count = len(df)
        df_cleaned = df.drop_duplicates()
        duplicates_removed = initial_count - len(df_cleaned)
        
        report = f"Removed {duplicates_removed} duplicate rows"
        return df_cleaned, report

    def _standardize_dates(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
        """Standardize date formats across columns regardless of format variations"""
        df_cleaned = df.copy()
        date_columns_fixed = 0
        
        for col in df_cleaned.columns:
            if df_cleaned[col].dtype == 'object':
                is_date, parsed_series = self.try_parse_datetime_series(df_cleaned[col])
                if is_date and parsed_series is not None:
                    df_cleaned[col] = parsed_series
                    date_columns_fixed += 1
        
        report = f"Standardized {date_columns_fixed} date columns"
        return df_cleaned, report

    def _clean_whitespace(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
        """Clean whitespace from string columns"""
        df_cleaned = df.copy()
        columns_cleaned = 0
        
        for col in df_cleaned.columns:
            if df_cleaned[col].dtype == 'object':
                df_cleaned[col] = df_cleaned[col].astype(str).str.strip()
                df_cleaned[col] = df_cleaned[col].str.replace(r'\s+', ' ', regex=True)
                columns_cleaned += 1
        
        report = f"Cleaned whitespace in {columns_cleaned} text columns"
        return df_cleaned, report

    def _fix_data_types(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
        """Attempt to fix data type issues (converting numeric strings & low cardinality categories)"""
        df_cleaned = df.copy()
        types_fixed = 0
        
        for col in df_cleaned.columns:
            if df_cleaned[col].dtype == 'object':
                is_date, parsed_date = self.try_parse_datetime_series(df_cleaned[col])
                if is_date and parsed_date is not None:
                    df_cleaned[col] = parsed_date
                    types_fixed += 1
                    continue

                is_num, parsed_num = self.try_parse_numeric_series(df_cleaned[col])
                if is_num and parsed_num is not None:
                    df_cleaned[col] = parsed_num
                    types_fixed += 1
                    continue
                
                unique_ratio = df_cleaned[col].nunique() / len(df_cleaned[col]) if len(df_cleaned[col]) > 0 else 1
                if unique_ratio < 0.1 and df_cleaned[col].nunique() > 1:
                    df_cleaned[col] = df_cleaned[col].astype('category')
                    types_fixed += 1
        
        report = f"Fixed data types for {types_fixed} columns"
        return df_cleaned, report

    def _remove_outliers(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
        """Remove outliers using IQR method"""
        df_cleaned = df.copy()
        numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
        outliers_removed = 0
        
        for col in numeric_cols:
            Q1 = df_cleaned[col].quantile(0.25)
            Q3 = df_cleaned[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers_before = len(df_cleaned)
            df_cleaned = df_cleaned[
                (df_cleaned[col] >= lower_bound) & (df_cleaned[col] <= upper_bound)
            ]
            outliers_removed += outliers_before - len(df_cleaned)
        
        report = f"Removed {outliers_removed} outlier rows from {len(numeric_cols)} numeric columns"
        return df_cleaned, report

    def add_cleaning_method(self, name: str, method_function):
        """Add a custom cleaning method"""
        self.cleaning_methods[name] = method_function

    def get_cleaning_suggestions(self, df: pd.DataFrame) -> List[str]:
        """Get suggestions for cleaning operations based on data assessment"""
        suggestions = []
        quality_info = self.assess_data_quality(df)
        
        if quality_info['missing_count'] > 0:
            suggestions.append(f"Handle {quality_info['missing_count']} missing values")
        if quality_info['duplicate_count'] > 0:
            suggestions.append(f"Remove {quality_info['duplicate_count']} duplicate rows")
        if quality_info['type_issues'] > 0:
            suggestions.append(f"Fix data types for {quality_info['type_issues']} columns")
        
        text_columns = df.select_dtypes(include=['object']).columns
        if len(text_columns) > 0:
            suggestions.append(f"Clean whitespace in {len(text_columns)} text columns")
        
        return suggestions
