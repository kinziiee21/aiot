import os
import json
import pandas as pd
from typing import Dict, Any, List, Optional
from services.groq_service import GroqService

class AIServices:
    """AI-powered services for data analysis and insights using Groq API (Replaced Hugging Face)"""
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.groq_service = GroqService(api_key=api_key, model=model)
        self.api_key = self.groq_service.api_key
        self.model = self.groq_service.model
        self.use_ai = self.groq_service.is_configured()
        self.last_error = self.groq_service.last_error
    
    def validate_token(self) -> Dict[str, Any]:
        """Validate Groq API connection"""
        return self.groq_service.validate_connection()
    
    def is_configured(self) -> bool:
        return self.groq_service.is_configured()
    
    def generate_completion(self, prompt: str, system_prompt: Optional[str] = None, 
                            temperature: float = 0.2, max_tokens: int = 1024, json_mode: bool = False) -> str:
        return self.groq_service.generate_completion(prompt, system_prompt=system_prompt, 
                                                    temperature=temperature, max_tokens=max_tokens, json_mode=json_mode)
    
    def generate_analysis_plan(self, question: str, column_info: List[Dict[str, str]], sample_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        return self.groq_service.generate_analysis_plan(question, column_info, sample_rows)
    
    def generate_insights(self, question: str, computed_summary: Dict[str, Any]) -> str:
        return self.groq_service.generate_insights(question, computed_summary)
