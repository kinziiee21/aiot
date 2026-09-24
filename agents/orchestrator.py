import pandas as pd
from typing import Dict, Any, Optional
from services.groq_service import GroqService
from agents.analysis_agent import AnalysisAgent
from agents.visualization_agent import VisualizationAgent

class OrchestratorAgent:
    """Central Orchestrator coordinating user requests, Groq plans, Pandas analysis, and Plotly visualization"""

    def __init__(self, groq_service: Optional[GroqService] = None):
        self.groq_service = groq_service or GroqService()
        self.analysis_agent = AnalysisAgent()
        self.viz_agent = VisualizationAgent()

    def process_request(
        self,
        question: str,
        df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Process a user question through the orchestrated pipeline:
        User -> Groq (Plan) -> Analysis Agent (Pandas) -> Viz Agent (Plotly) -> Groq (Insights)
        """
        if df is None or df.empty:
            return {
                "success": False,
                "error": "No dataset uploaded.",
                "summary": "Please upload a CSV or XLSX file first."
            }

        # 1. Extract schema information for plan generation
        column_info = []
        for col in df.columns:
            column_info.append({
                "name": col,
                "type": str(df[col].dtype)
            })

        sample_rows = df.head(3).to_dict(orient="records")

        # 2. Generate structured analysis plan via Groq
        plan = self.groq_service.generate_analysis_plan(
            question=question,
            column_info=column_info,
            sample_rows=sample_rows
        )

        # 3. Execute safe Pandas computation via AnalysisAgent
        analysis_result = self.analysis_agent.execute_plan(df, plan)

        if not analysis_result.get("success", False):
            return {
                "success": False,
                "plan": plan,
                "error": analysis_result.get("error"),
                "summary": analysis_result.get("summary")
            }

        result_df = analysis_result.get("data", pd.DataFrame())
        summary_text = analysis_result.get("summary", "")

        # 4. Generate Plotly chart via VisualizationAgent
        chart_config = self.viz_agent.select_chart_type(question, result_df, plan)
        chart_fig = self.viz_agent.render_visualization(result_df, chart_config)

        # 5. Generate AI insights strictly from computed numbers via GroqService
        computed_summary = {
            "question": question,
            "summary": summary_text,
            "statistics": analysis_result.get("statistics", {}),
            "result_rows_count": len(result_df),
            "result_sample": result_df.head(5).to_dict(orient="records") if not result_df.empty else []
        }

        insights = self.groq_service.generate_insights(question, computed_summary)

        return {
            "success": True,
            "question": question,
            "plan": plan,
            "result_df": result_df,
            "summary": summary_text,
            "chart_config": chart_config,
            "chart": chart_fig,
            "insights": insights
        }
