import pandas as pd
from typing import Optional, Dict, Any
from modules.data_cleaning import DataCleaning

class DataIngestion:
    """Handles data ingestion from file uploads with automated type detection"""
    
    def __init__(self):
        self.supported_formats = ['csv', 'xlsx', 'xls']
        self.max_file_size_mb = 100
    
    def auto_detect_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Automatically detects and converts date strings and numeric strings on upload"""
        df_conv = df.copy()
        cleaner = DataCleaning()
        for col in df_conv.columns:
            if df_conv[col].dtype == 'object':
                is_date, parsed_date = cleaner.try_parse_datetime_series(df_conv[col])
                if is_date and parsed_date is not None:
                    df_conv[col] = parsed_date
                    continue
                
                is_num, parsed_num = cleaner.try_parse_numeric_series(df_conv[col])
                if is_num and parsed_num is not None:
                    df_conv[col] = parsed_num
                    continue
        return df_conv

    def load_file(self, uploaded_file) -> pd.DataFrame:
        """Load data from uploaded CSV or XLSX file and auto-detect column formats"""
        try:
            file_extension = uploaded_file.name.split('.')[-1].lower()
            
            if file_extension == 'csv':
                try:
                    df = pd.read_csv(uploaded_file, encoding='utf-8')
                except UnicodeDecodeError:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, encoding='latin-1')
                    
            elif file_extension in ['xlsx', 'xls']:
                df = pd.read_excel(uploaded_file)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
            
            if df.empty:
                raise ValueError("The uploaded file is empty")
                
            return self.auto_detect_types(df)
            
        except Exception as e:
            raise Exception(f"Error loading file: {str(e)}")
    
    def validate_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate loaded data and return basic info"""
        return {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.to_dict(),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024 * 1024),
            'missing_values': df.isnull().sum().to_dict(),
            'duplicate_rows': df.duplicated().sum()
        }
    
    def get_sample_data(self, df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
        """Get sample of the data for preview"""
        return df.head(n)
