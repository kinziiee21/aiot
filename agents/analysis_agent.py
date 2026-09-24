import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List, Optional

class AnalysisAgent:
    """Agent responsible for validating structured analysis plans and executing safe Pandas operations"""

    def validate_plan(self, df: pd.DataFrame, plan: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
        """Validate the analysis plan against dataset columns and structure"""
        if df is None or df.empty:
            return False, "Dataset is empty or not loaded.", plan

        cols = df.columns.tolist()
        validated_plan = plan.copy()

        primary_col = validated_plan.get("primary_column")
        groupby_col = validated_plan.get("groupby_column")

        # Fallback primary column if missing or invalid
        if primary_col and primary_col not in cols:
            # Case insensitive search
            matches = [c for c in cols if c.lower() == primary_col.lower()]
            if matches:
                validated_plan["primary_column"] = matches[0]
            else:
                num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                validated_plan["primary_column"] = num_cols[0] if num_cols else cols[0]

        if groupby_col and groupby_col not in cols:
            matches = [c for c in cols if c.lower() == groupby_col.lower()]
            if matches:
                validated_plan["groupby_column"] = matches[0]
            else:
                cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
                validated_plan["groupby_column"] = cat_cols[0] if cat_cols else None

        valid_ops = ["sum", "mean", "median", "count", "min", "max", "top_n", "describe", "correlation", "value_counts"]
        if validated_plan.get("operation") not in valid_ops:
            validated_plan["operation"] = "mean"

        return True, "Analysis plan validated successfully.", validated_plan

    def execute_plan(self, df: pd.DataFrame, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Execute safe Pandas operations based strictly on the validated plan"""
        is_valid, msg, valid_plan = self.validate_plan(df, plan)
        if not is_valid:
            return {"error": msg, "data": pd.DataFrame(), "summary": msg}

        df_work = df.copy()

        # Apply optional filters
        filters = valid_plan.get("filters", [])
        if isinstance(filters, list):
            for f in filters:
                if isinstance(f, dict) and "column" in f and "operator" in f and "value" in f:
                    col = f["column"]
                    op = f["operator"]
                    val = f["value"]
                    if col in df_work.columns:
                        if op == ">":
                            df_work = df_work[df_work[col] > float(val)]
                        elif op == ">=":
                            df_work = df_work[df_work[col] >= float(val)]
                        elif op == "<":
                            df_work = df_work[df_work[col] < float(val)]
                        elif op == "<=":
                            df_work = df_work[df_work[col] <= float(val)]
                        elif op == "==":
                            df_work = df_work[df_work[col] == val]
                        elif op == "!=":
                            df_work = df_work[df_work[col] != val]
                        elif op == "contains":
                            df_work = df_work[df_work[col].astype(str).str.contains(str(val), case=False, na=False)]

        primary_col = valid_plan.get("primary_column")
        groupby_col = valid_plan.get("groupby_column")
        operation = valid_plan.get("operation", "mean")
        limit = valid_plan.get("limit")

        result_df = pd.DataFrame()
        summary_text = ""
        statistics = {}

        try:
            if operation in ["sum", "mean", "median", "count", "min", "max"]:
                if groupby_col and groupby_col in df_work.columns:
                    if primary_col and primary_col in df_work.columns and pd.api.types.is_numeric_dtype(df_work[primary_col]):
                        res = df_work.groupby(groupby_col)[primary_col].agg(operation).reset_index()
                        res = res.sort_values(by=primary_col, ascending=False)
                    else:
                        res = df_work.groupby(groupby_col).size().reset_index(name="count")
                        primary_col = "count"
                        res = res.sort_values(by="count", ascending=False)

                    if limit and isinstance(limit, int) and limit > 0:
                        res = res.head(limit)

                    result_df = res
                    top_row = res.iloc[0] if len(res) > 0 else None
                    if top_row is not None:
                        summary_text = f"Highest {groupby_col} for {primary_col} ({operation}) is '{top_row[groupby_col]}' with {top_row[primary_col]:,.2f}."
                    statistics = {
                        "total_groups": len(res),
                        "operation": operation,
                        "primary_column": primary_col,
                        "groupby_column": groupby_col
                    }
                else:
                    # Overall column aggregation
                    if primary_col and primary_col in df_work.columns and pd.api.types.is_numeric_dtype(df_work[primary_col]):
                        val = float(getattr(df_work[primary_col], operation)())
                        result_df = pd.DataFrame([{primary_col: val, "Metric": operation.title()}])
                        summary_text = f"The overall {operation} of {primary_col} is {val:,.2f}."
                        statistics = {operation: val, "column": primary_col}
                    else:
                        count_val = len(df_work)
                        result_df = pd.DataFrame([{"Total Count": count_val}])
                        summary_text = f"Total rows count: {count_val}."
                        statistics = {"total_count": count_val}

            elif operation == "top_n":
                if groupby_col and groupby_col in df_work.columns and primary_col and primary_col in df_work.columns:
                    res = df_work.groupby(groupby_col)[primary_col].sum().reset_index()
                    res = res.sort_values(by=primary_col, ascending=False)
                    n = limit if (limit and isinstance(limit, int)) else 10
                    res = res.head(n)
                    result_df = res
                    top_row = res.iloc[0] if len(res) > 0 else None
                    if top_row is not None:
                        summary_text = f"Top performer: '{top_row[groupby_col]}' with total {primary_col} of {top_row[primary_col]:,.2f}."
                elif primary_col and primary_col in df_work.columns:
                    n = limit if (limit and isinstance(limit, int)) else 10
                    res = df_work.sort_values(by=primary_col, ascending=False).head(n)
                    result_df = res
                    summary_text = f"Showing top {len(res)} rows sorted by {primary_col}."

            elif operation == "value_counts":
                col_to_count = groupby_col or primary_col or df_work.columns[0]
                res = df_work[col_to_count].value_counts().reset_index()
                res.columns = [col_to_count, "Count"]
                if limit and isinstance(limit, int):
                    res = res.head(limit)
                result_df = res
                top_row = res.iloc[0] if len(res) > 0 else None
                if top_row is not None:
                    summary_text = f"Most frequent value in {col_to_count} is '{top_row[col_to_count]}' ({top_row['Count']} occurrences)."

            elif operation == "describe":
                target_col = primary_col if (primary_col and primary_col in df_work.columns) else None
                if target_col and pd.api.types.is_numeric_dtype(df_work[target_col]):
                    desc = df_work[target_col].describe().reset_index()
                    desc.columns = ["Statistic", "Value"]
                    result_df = desc
                    summary_text = f"Descriptive statistics computed for column '{target_col}'."
                else:
                    num_cols = df_work.select_dtypes(include=[np.number]).columns
                    desc = df_work[num_cols].describe().reset_index()
                    result_df = desc
                    summary_text = f"Descriptive statistics computed for all numeric columns."

            elif operation == "correlation":
                num_cols = df_work.select_dtypes(include=[np.number]).columns
                corr_df = df_work[num_cols].corr().reset_index()
                result_df = corr_df
                summary_text = f"Correlation matrix computed for {len(num_cols)} numeric columns."

            else:
                result_df = df_work.head(10)
                summary_text = "Data preview (first 10 rows)."

            return {
                "success": True,
                "data": result_df,
                "summary": summary_text,
                "statistics": statistics,
                "plan_executed": valid_plan
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "data": pd.DataFrame(),
                "summary": f"Computation failed: {str(e)}",
                "statistics": {}
            }
