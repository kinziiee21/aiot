import pandas as pd
import plotly.graph_objects as go
from typing import Dict, Any, Optional
from modules.visualization import Visualization

class VisualizationAgent:
    """Agent responsible for intelligent chart selection and Plotly chart generation"""

    def __init__(self):
        self.visualizer = Visualization()

    def select_chart_type(
        self,
        question: str,
        result_df: pd.DataFrame,
        plan: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Determine chart parameters (type, x_col, y_col, title) based on question & computed data"""
        if result_df is None or result_df.empty:
            return {"chart_type": "none", "title": "No Data to Display"}

        recommended = plan.get("recommended_visualization")
        groupby_col = plan.get("groupby_column")
        primary_col = plan.get("primary_column")
        intent = plan.get("intent", "").lower()
        q_lower = question.lower()

        cols = result_df.columns.tolist()
        num_cols = result_df.select_dtypes(include=['number']).columns.tolist()
        cat_cols = result_df.select_dtypes(include=['object', 'category']).columns.tolist()

        x_col = None
        y_col = None
        chart_type = recommended or "bar"

        # 1. Semantic override from question keywords
        if any(w in q_lower for w in ["trend", "monthly", "over time", "evolution"]):
            chart_type = "line"
        elif any(w in q_lower for w in ["relationship", "versus", "vs", "scatter", "correlation"]):
            chart_type = "scatter" if len(num_cols) >= 2 else "bar"
        elif any(w in q_lower for w in ["distribution", "histogram", "spread"]):
            chart_type = "histogram"
        elif any(w in q_lower for w in ["proportion", "share", "percentage", "pie"]):
            chart_type = "pie"

        # 2. Match X and Y columns based on computed result schema
        if groupby_col and groupby_col in cols:
            x_col = groupby_col
            if primary_col and primary_col in cols and primary_col != groupby_col:
                y_col = primary_col
            elif num_cols:
                y_col = num_cols[0]
        else:
            if len(cols) >= 2:
                x_col = cols[0]
                y_col = cols[1] if len(num_cols) > 0 else None
            elif len(cols) == 1:
                x_col = cols[0]

        title = f"{y_col or 'Value'} by {x_col}" if (x_col and y_col) else f"Analysis of {x_col or 'Data'}"

        return {
            "chart_type": chart_type,
            "x_column": x_col,
            "y_column": y_col,
            "title": title
        }

    def render_visualization(
        self,
        result_df: pd.DataFrame,
        chart_config: Dict[str, Any]
    ) -> Optional[go.Figure]:
        """Render Plotly figure based on computed chart config"""
        if result_df is None or result_df.empty or chart_config.get("chart_type") == "none":
            return None

        try:
            return self.visualizer.create_chart(result_df, chart_config)
        except Exception as e:
            return self.visualizer.create_error_chart(f"Visualization error: {str(e)}")
