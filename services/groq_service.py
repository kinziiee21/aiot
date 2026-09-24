import os
import json
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from .env
env_path = Path(__file__).resolve().parent.parent / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
else:
    load_dotenv()

class GroqService:
    """Service wrapper for Groq LLM API integration"""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        raw_key = api_key or os.getenv("GROQ_API_KEY", "")
        self.api_key = raw_key.strip() if raw_key else ""
        raw_model = model or os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        self.model = raw_model.strip() if raw_model else "llama-3.3-70b-versatile"
        self._client = None
        self.last_error = None

        if self.api_key and self.api_key != "your_groq_api_key_here":
            try:
                from groq import Groq
                self._client = Groq(api_key=self.api_key)
            except Exception as e:
                self._client = None
                self.last_error = str(e)
        else:
            self.last_error = "GROQ_API_KEY is not configured"

    def is_configured(self) -> bool:
        """Check if Groq client is configured and initialized"""
        return self._client is not None

    def validate_connection(self) -> Dict[str, Any]:
        """Validate connection to Groq API with a lightweight request"""
        if not self.is_configured():
            return {
                "valid": False,
                "message": "Groq API key not configured.",
                "details": "Please set GROQ_API_KEY in your .env or .env.example file."
            }

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "Respond with the word OK."}],
                max_tokens=10,
                temperature=0.0
            )
            content = response.choices[0].message.content.strip()
            return {
                "valid": True,
                "message": "Groq API connected successfully!",
                "details": f"Model: {self.model} | Response: {content}"
            }
        except Exception as e:
            return {
                "valid": False,
                "message": "Groq API connection test failed.",
                "details": str(e)
            }

    def generate_completion(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        json_mode: bool = False
    ) -> str:
        """Generate text completion using Groq API"""
        if not self.is_configured():
            raise ValueError("Groq API key is not configured.")

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        response = self._client.chat.completions.create(**kwargs)
        return response.choices[0].message.content.strip()

    def generate_analysis_plan(
        self,
        question: str,
        column_info: List[Dict[str, str]],
        sample_rows: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate a structured JSON analysis plan for a natural language question.
        """
        system_prompt = (
            "You are an expert Data Analyst AI agent. Your job is to convert natural language "
            "questions about a dataset into a precise, structured JSON analysis plan. "
            "You must ONLY return a valid JSON object matching the required schema. Do NOT execute Python code."
        )

        prompt = f"""
Dataset Schema:
Columns: {json.dumps(column_info, indent=2)}

Sample Data Preview (3 rows):
{json.dumps(sample_rows, indent=2, default=str)}

User Question: "{question}"

Instructions:
Create a structured analysis plan as a JSON object with the following fields:
- "intent": string (e.g. "summary", "comparison", "trend", "distribution", "correlation", "outliers", "filter_top")
- "primary_column": string or null (the main column being analyzed/measured)
- "groupby_column": string or null (the category column to group by, if any)
- "operation": string (one of: "sum", "mean", "median", "count", "min", "max", "describe", "correlation", "top_n", "value_counts")
- "limit": integer or null (e.g., top 5 or top 10)
- "filters": list of objects (e.g. [{{"column": "Sales", "operator": ">", "value": 100}}]) or empty list
- "recommended_visualization": string (one of: "bar", "line", "scatter", "histogram", "box", "pie", "heatmap", "none")
- "explanation": string (short description of the calculation strategy)

Respond with ONLY valid JSON:
"""

        if not self.is_configured():
            return self._heuristic_fallback_plan(question, column_info)

        try:
            raw_response = self.generate_completion(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.1,
                json_mode=True
            )
            plan = json.loads(raw_response)
            return plan
        except Exception as e:
            return self._heuristic_fallback_plan(question, column_info, error=str(e))

    def generate_insights(
        self,
        question: str,
        computed_summary: Dict[str, Any]
    ) -> str:
        """
        Generate concise data-driven insights based strictly on actual computed Pandas results.
        """
        if not self.is_configured():
            return self._heuristic_fallback_insights(computed_summary)

        system_prompt = (
            "You are a professional Data Analyst. Provide key insights based STRICTLY on the "
            "actual computed statistics provided. Do NOT invent or hallucinate numbers or stats. "
            "Keep the response concise, formatted in markdown with bullet points."
        )

        prompt = f"""
Question asked: "{question}"

Actual Computed Results:
{json.dumps(computed_summary, indent=2, default=str)}

Instructions:
Provide:
1. Key Findings (bullet points citing exact numbers from computed results)
2. Business/Practical Insight (what this result implies)
3. Suggested Next Question
"""

        try:
            return self.generate_completion(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.3
            )
        except Exception as e:
            return self._heuristic_fallback_insights(computed_summary, error=str(e))

    def _heuristic_fallback_plan(
        self,
        question: str,
        column_info: List[Dict[str, str]],
        error: Optional[str] = None
    ) -> Dict[str, Any]:
        """Rule-based plan fallback when AI service is offline"""
        q_lower = question.lower()
        cols = [c['name'] for c in column_info]
        num_cols = [c['name'] for c in column_info if any(t in c['type'].lower() for t in ['int', 'float', 'num'])]
        cat_cols = [c['name'] for c in column_info if any(t in c['type'].lower() for t in ['obj', 'str', 'cat', 'text'])]

        primary_col = num_cols[0] if num_cols else (cols[0] if cols else None)
        group_col = cat_cols[0] if cat_cols else None

        for c in cols:
            if c.lower() in q_lower:
                if c in num_cols:
                    primary_col = c
                elif c in cat_cols:
                    group_col = c

        operation = "mean"
        viz = "bar"
        intent = "summary"

        if "total" in q_lower or "sum" in q_lower:
            operation = "sum"
        elif "average" in q_lower or "avg" in q_lower or "mean" in q_lower:
            operation = "mean"
        elif "count" in q_lower or "how many" in q_lower:
            operation = "count"
        elif "highest" in q_lower or "top" in q_lower or "max" in q_lower:
            operation = "top_n"
            viz = "bar"

        if "trend" in q_lower or "time" in q_lower or "monthly" in q_lower:
            viz = "line"
            intent = "trend"
        elif "relationship" in q_lower or "versus" in q_lower or "vs" in q_lower:
            viz = "scatter"
            intent = "correlation"
        elif "distribution" in q_lower:
            viz = "histogram"
            intent = "distribution"

        return {
            "intent": intent,
            "primary_column": primary_col,
            "groupby_column": group_col,
            "operation": operation,
            "limit": 10 if "top" in q_lower or "highest" in q_lower else None,
            "filters": [],
            "recommended_visualization": viz,
            "explanation": f"Rule-based fallback strategy for: '{question}'",
            "fallback_note": f"Rule-based plan generated ({error or 'Groq API Key not set'})"
        }

    def _heuristic_fallback_insights(
        self,
        computed_summary: Dict[str, Any],
        error: Optional[str] = None
    ) -> str:
        """Rule-based text summary when AI service is offline"""
        lines = ["### Key Findings (Computed)"]
        if "summary" in computed_summary:
            lines.append(f"- **Summary**: {computed_summary['summary']}")
        if "top_result" in computed_summary:
            lines.append(f"- **Top Result**: {computed_summary['top_result']}")
        if "statistics" in computed_summary:
            stats = computed_summary['statistics']
            for k, v in stats.items():
                lines.append(f"- **{k}**: {v}")

        lines.append("\n*Note: Basic rule-based insight generated. Configure GROQ_API_KEY in Settings for AI insights.*")
        return "\n".join(lines)
