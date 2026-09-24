import streamlit as st
import pandas as pd
import os
from services.groq_service import GroqService
from agents import (
    CleaningAgent,
    EDAAgent,
    AnalysisAgent,
    VisualizationAgent,
    ReportAgent,
    OrchestratorAgent
)
from modules.data_ingestion import DataIngestion
from utils.helpers import format_bytes, validate_file_size
from modules.version_control import VersionControl

# Page configuration
st.set_page_config(
    page_title="Natural Language Data Analyst",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'cleaned_data' not in st.session_state:
    st.session_state.cleaned_data = None
if 'active_dataset_name' not in st.session_state:
    st.session_state.active_dataset_name = "Uploaded Dataset"
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'current_step' not in st.session_state:
    st.session_state.current_step = 1
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = "📥 Data Ingestion"

# Initialize services and agents in session state
if 'groq_service' not in st.session_state:
    st.session_state.groq_service = GroqService()
if 'orchestrator' not in st.session_state:
    st.session_state.orchestrator = OrchestratorAgent(st.session_state.groq_service)
if 'cleaning_agent' not in st.session_state:
    st.session_state.cleaning_agent = CleaningAgent()
if 'eda_agent' not in st.session_state:
    st.session_state.eda_agent = EDAAgent()
if 'report_agent' not in st.session_state:
    st.session_state.report_agent = ReportAgent()
if 'version_control' not in st.session_state:
    st.session_state.version_control = VersionControl()


def main():
    st.title("📊 Natural Language Data Analyst")
    st.caption("Upload your dataset, ask questions in natural language, and get AI-powered analysis, visualizations, and insights via Groq.")
    
    nav_options = [
        "📥 Data Ingestion", 
        "🧹 Data Quality & Cleaning", 
        "🗣️ Natural Language Analysis", 
        "📊 EDA & Visualizations", 
        "📄 Report Generation",
        "⚙️ Settings & Groq"
    ]

    # Sidebar navigation & status
    with st.sidebar:
        st.header("📌 Navigation")
        
        for idx, option in enumerate(nav_options, 1):
            if st.session_state.active_tab == option:
                prefix = "👉 "
            elif st.session_state.current_step > idx:
                prefix = "✅ "
            else:
                prefix = "🔹 "
            
            clean_name = option.split(' ', 1)[1] if ' ' in option else option
            btn_label = f"{prefix}{idx}. {clean_name}"
            
            if st.button(btn_label, key=f"sidebar_nav_btn_{idx}", use_container_width=True):
                st.session_state.active_tab = option
                st.rerun()

        st.markdown("---")
        st.subheader("💡 Groq AI Status")
        if st.session_state.groq_service.is_configured():
            st.success(f"🟢 Groq API Connected ({st.session_state.groq_service.model})")
        else:
            st.warning("🟡 Groq Key Not Detected (Using Rule Fallback)")

        st.markdown("---")
        st.caption("Natural Language Data Analyst • Powered by Groq LLM")

    # Render active view strictly based on sidebar selection
    active_tab = st.session_state.active_tab

    if active_tab == "📥 Data Ingestion":
        handle_data_ingestion()
    elif active_tab == "🧹 Data Quality & Cleaning":
        handle_data_cleaning()
    elif active_tab == "🗣️ Natural Language Analysis":
        handle_nl_analysis()
    elif active_tab == "📊 EDA & Visualizations":
        handle_eda_and_visualization()
    elif active_tab == "📄 Report Generation":
        handle_report_generation()
    elif active_tab == "⚙️ Settings & Groq":
        handle_settings()

def get_current_df() -> pd.DataFrame:
    """Return cleaned dataset if available, otherwise raw data"""
    if st.session_state.cleaned_data is not None:
        return st.session_state.cleaned_data
    return st.session_state.data

def handle_data_ingestion():
    st.header("📥 Data Ingestion & Overview")
    
    st.subheader("Upload CSV or XLSX Dataset")
    st.caption("Upload your local data file. Date formats and numerical currency/percentage columns are detected automatically.")
    
    data_ingestion = DataIngestion()
    
    uploaded_file = st.file_uploader(
        "Choose a file to ingest",
        type=['csv', 'xlsx', 'xls'],
        help="Supported formats: CSV, XLSX, XLS. Maximum size: 100MB"
    )
    
    if uploaded_file is not None:
        if validate_file_size(uploaded_file, max_size_mb=100):
            with st.spinner("Loading, auto-detecting types, and profiling dataset..."):
                try:
                    st.session_state.data = data_ingestion.load_file(uploaded_file)
                    st.session_state.cleaned_data = None
                    st.session_state.active_dataset_name = uploaded_file.name
                    st.session_state.current_step = max(st.session_state.current_step, 2)
                    st.success(f"✅ Data loaded successfully! Shape: {st.session_state.data.shape[0]} rows × {st.session_state.data.shape[1]} columns")
                except Exception as e:
                    st.error(f"❌ Error loading file: {str(e)}")
        else:
            st.error("❌ File size exceeds 100MB limit")

    # Display Dataset Overview Dashboard if data is loaded
    df = get_current_df()
    if df is not None and not df.empty:
        st.markdown("---")
        st.subheader("📊 Dataset Overview Dashboard")

        # Dataset metrics cards
        num_cols = df.select_dtypes(include=['number']).columns.tolist()
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        dt_cols = df.select_dtypes(include=['datetime', 'datetime64']).columns.tolist()
        missing_count = int(df.isnull().sum().sum())
        dup_count = int(df.duplicated().sum())

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Rows", f"{len(df):,}")
        col2.metric("Columns", len(df.columns))
        col3.metric("Numeric Columns", len(num_cols))
        col4.metric("Categorical Columns", len(cat_cols))

        col5, col6, col7, col8 = st.columns(4)
        col5.metric("Date Columns", len(dt_cols))
        col6.metric("Missing Values", f"{missing_count:,}")
        col7.metric("Duplicate Rows", f"{dup_count:,}")
        col8.metric("Memory Usage", format_bytes(df.memory_usage(deep=True).sum()))

        st.subheader("👀 Dataset Preview")
        st.dataframe(df.head(10), use_container_width=True)

        st.subheader("📋 Column Information")
        col_summary = []
        for c in df.columns:
            col_summary.append({
                "Column Name": c,
                "Data Type": str(df[c].dtype),
                "Non-Null Count": df[c].count(),
                "Missing Values": df[c].isnull().sum(),
                "Unique Values": df[c].nunique()
            })
        st.dataframe(pd.DataFrame(col_summary), use_container_width=True)

def handle_data_cleaning():
    st.header("🧹 Data Quality Assessment & Cleaning")
    
    if st.session_state.data is None:
        st.warning("⚠️ Please load a dataset first in the Data Ingestion tab.")
        return

    cleaning_agent: CleaningAgent = st.session_state.cleaning_agent
    df_raw = st.session_state.data

    st.subheader("📊 Data Quality Assessment")
    quality_info = cleaning_agent.assess_quality(df_raw)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Overall Quality Score", f"{quality_info['quality_score']}/100")
    col2.metric("Missing Cells", quality_info['missing_count'])
    col3.metric("Duplicate Rows", quality_info['duplicate_count'])
    col4.metric("Data Type Issues", quality_info['type_issues'])

    # Show recommendations
    st.subheader("💡 Recommended Cleaning Operations")
    recs = cleaning_agent.get_recommendations(df_raw)
    for rec in recs:
        prefix = "✅ (Recommended)" if rec['recommended'] else "ℹ️ (Optional)"
        st.markdown(f"**{prefix} {rec['label']}**: {rec['detail']}")

    st.markdown("---")
    st.subheader("⚙️ Select Cleaning Operations")

    col1, col2 = st.columns(2)
    with col1:
        handle_missing = st.checkbox("Handle Missing Values", value=quality_info['missing_count'] > 0)
        remove_duplicates = st.checkbox("Remove Duplicate Rows", value=quality_info['duplicate_count'] > 0)
        fix_data_types = st.checkbox("Convert Numeric Strings & Fix Types", value=quality_info['type_issues'] > 0)

    with col2:
        standardize_dates = st.checkbox("Standardize Date Formats", value=bool(quality_info.get('possible_date_cols')))
        clean_whitespace = st.checkbox("Clean Whitespace in Text", value=True)
        remove_outliers = st.checkbox("Remove Outliers (IQR Method)", value=False)

    missing_strategy = "drop"
    if handle_missing:
        missing_strategy = st.selectbox(
            "Missing Value Imputation Strategy:",
            ["drop", "mean", "median", "mode", "forward_fill", "backward_fill"]
        )

    if st.button("🧹 Apply Selected Cleaning Operations"):
        with st.spinner("Cleaning dataset..."):
            options = {
                'handle_missing': handle_missing,
                'remove_duplicates': remove_duplicates,
                'standardize_dates': standardize_dates,
                'clean_whitespace': clean_whitespace,
                'fix_data_types': fix_data_types,
                'remove_outliers': remove_outliers,
                'missing_strategy': missing_strategy
            }
            cleaned_df, report = cleaning_agent.clean(df_raw, options)
            st.session_state.cleaned_data = cleaned_df
            st.session_state.current_step = max(st.session_state.current_step, 3)

            st.success("✅ Dataset cleaned successfully!")

            st.subheader("📋 Cleaning Summary Report")
            for step, res in report.items():
                st.write(f"- **{step}**: {res}")

            col_b, col_a = st.columns(2)
            with col_b:
                st.subheader("Original Data")
                st.write(f"Shape: {df_raw.shape}")
                st.dataframe(df_raw.head())
            with col_a:
                st.subheader("Cleaned Data")
                st.write(f"Shape: {cleaned_df.shape}")
                st.dataframe(cleaned_df.head())

    if st.session_state.cleaned_data is not None:
        st.markdown("---")
        col_dl, col_snap = st.columns(2)
        with col_dl:
            csv_data = st.session_state.cleaned_data.to_csv(index=False)
            st.download_button(
                label="📥 Download Cleaned CSV Dataset",
                data=csv_data,
                file_name="cleaned_dataset.csv",
                mime="text/csv"
            )

        with col_snap:
            commit_msg = st.text_input("Commit message for version snapshot:", "Data cleaning applied")
            if st.button("💾 Save Version Snapshot"):
                vid = st.session_state.version_control.save_version(
                    st.session_state.cleaned_data,
                    message=commit_msg
                )
                st.success(f"📌 Version saved! Version ID: {vid}")

def handle_nl_analysis():
    st.header("🗣️ Natural Language Data Analysis")

    df = get_current_df()
    if df is None or df.empty:
        st.warning("⚠️ Please load a dataset first in the Data Ingestion tab.")
        return

    st.info(f"📁 Active Dataset: **{st.session_state.active_dataset_name}** ({'Cleaned' if st.session_state.cleaned_data is not None else 'Original'}) | Shape: {df.shape[0]} rows × {df.shape[1]} cols")

    st.subheader("💡 Sample Questions You Can Ask")
    sample_qs = [
        "What is the average sales?",
        "Which product has the highest revenue?",
        "Which region generated the most sales?",
        "Compare sales between regions.",
        "Show the monthly sales trend.",
        "Are there any unusual values?",
        "What are the main insights from this dataset?"
    ]

    selected_q = None
    cols_q = st.columns(3)
    for idx, q in enumerate(sample_qs):
        if cols_q[idx % 3].button(q, key=f"sq_{idx}"):
            selected_q = q

    user_query = st.text_input("Ask a question in natural language:", value=selected_q or "", placeholder="e.g. Which region generated the highest sales?")

    if user_query and st.button("🔍 Analyze Question"):
        orchestrator: OrchestratorAgent = st.session_state.orchestrator
        with st.spinner("Analyzing question via Groq, executing calculations, and generating insights..."):
            res = orchestrator.process_request(user_query, df)

            if res.get("success"):
                st.session_state.chat_history.append(res)
                st.session_state.current_step = max(st.session_state.current_step, 4)

                st.markdown("---")
                st.subheader("📌 Analysis Result Summary")
                st.info(res.get("summary", "Analysis completed successfully."))

                with st.expander("🛠️ View Groq Analysis Plan (Structured JSON)", expanded=False):
                    st.json(res.get("plan", {}))

                st.subheader("📊 Computed Results")
                st.dataframe(res.get("result_df"), use_container_width=True)

                chart_fig = res.get("chart")
                if chart_fig:
                    st.subheader("📈 Visualization")
                    st.plotly_chart(chart_fig, use_container_width=True)

                st.subheader("💡 AI Insights & Key Findings (Powered by Groq)")
                st.markdown(res.get("insights", "No insights generated."))

            else:
                st.error(f"❌ Analysis failed: {res.get('error', 'Unknown error')}")

    if st.session_state.chat_history:
        st.markdown("---")
        st.subheader("📜 Question & Analysis History")
        
        for idx, past in enumerate(reversed(st.session_state.chat_history), 1):
            with st.expander(f"Q: {past['question']}", expanded=(idx == 1)):
                st.write(f"**Answer Summary**: {past['summary']}")
                st.dataframe(past['result_df'])
                if past.get("chart"):
                    st.plotly_chart(past['chart'], use_container_width=True, key=f"hist_chart_{idx}")
                st.markdown(f"**Insights**:\n{past['insights']}")

        if st.button("🗑️ Clear History"):
            st.session_state.chat_history = []
            st.rerun()

def generate_automatic_5_charts(df: pd.DataFrame):
    import plotly.express as px
    import plotly.graph_objects as go
    
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    dt_cols = df.select_dtypes(include=['datetime', 'datetime64']).columns.tolist()
    
    charts = []
    
    # 1. Chart 1: Primary Numerical Distribution (Histogram & Box Plot)
    if num_cols:
        target_num = num_cols[0]
        fig1 = px.histogram(df, x=target_num, marginal="box", title=f"Distribution of {target_num}",
                            color_discrete_sequence=['#6366F1'])
        fig1.update_layout(template="plotly_dark", height=360)
        charts.append((f"Distribution Analysis: {target_num}", fig1, f"Histogram and Box plot showing numerical distribution and outliers of {target_num}."))
    elif df.columns.tolist():
        c = df.columns[0]
        fig1 = px.bar(df[c].value_counts().head(10).reset_index(), x='index', y=c, title=f"Frequency of {c}")
        fig1.update_layout(template="plotly_dark", height=360)
        charts.append((f"Frequency Distribution: {c}", fig1, f"Count breakdown of values in {c}."))

    # 2. Chart 2: Categorical Breakdown (Bar Chart)
    if cat_cols:
        target_cat = cat_cols[0]
        top_cats = df[target_cat].value_counts().head(10).reset_index()
        top_cats.columns = [target_cat, 'Count']
        fig2 = px.bar(top_cats, x=target_cat, y='Count', color='Count', color_continuous_scale='Viridis',
                      title=f"Top 10 Categories in {target_cat}")
        fig2.update_layout(template="plotly_dark", height=360)
        charts.append((f"Category Breakdown: {target_cat}", fig2, f"Top 10 most frequent categories in {target_cat}."))
    elif len(num_cols) >= 2:
        fig2 = px.scatter(df, x=num_cols[0], y=num_cols[1], title=f"{num_cols[0]} vs {num_cols[1]} Scatter Plot")
        fig2.update_layout(template="plotly_dark", height=360)
        charts.append((f"Scatter Comparison: {num_cols[0]} vs {num_cols[1]}", fig2, f"Scatter plot showing correlation between {num_cols[0]} and {num_cols[1]}."))

    # 3. Chart 3: Metric Aggregation by Category
    if cat_cols and num_cols:
        cat_c = cat_cols[0]
        num_c = num_cols[0]
        agg_df = df.groupby(cat_c)[num_c].sum().nlargest(10).reset_index()
        fig3 = px.bar(agg_df, x=cat_c, y=num_c, color=num_c, color_continuous_scale='Magma',
                      title=f"Total {num_c} by {cat_c}")
        fig3.update_layout(template="plotly_dark", height=360)
        charts.append((f"Metric Aggregation: Total {num_c} by {cat_c}", fig3, f"Sum of {num_c} aggregated across top categories of {cat_c}."))
    elif len(num_cols) >= 2:
        fig3 = px.box(df, y=num_cols[1], title=f"Box Plot of {num_cols[1]}")
        fig3.update_layout(template="plotly_dark", height=360)
        charts.append((f"Box Plot: {num_cols[1]} Outlier Analysis", fig3, f"Quartiles and IQR outlier bounds for {num_cols[1]}."))
    else:
        fig3 = px.line(df.head(50), y=df.columns[0], title=f"Sequence Plot of {df.columns[0]}")
        fig3.update_layout(template="plotly_dark", height=360)
        charts.append((f"Sequence Plot: {df.columns[0]}", fig3, "Sequence plot of records."))

    # 4. Chart 4: Time Series Trend or Secondary Numerical Distribution
    if dt_cols and num_cols:
        dt_c = dt_cols[0]
        num_c = num_cols[0]
        ts_df = df.dropna(subset=[dt_c]).sort_values(dt_c)
        fig4 = px.line(ts_df, x=dt_c, y=num_c, title=f"Time Series Trend: {num_c} over {dt_c}",
                       color_discrete_sequence=['#10B981'])
        fig4.update_layout(template="plotly_dark", height=360)
        charts.append((f"Time-Series Trend: {num_c} over {dt_c}", fig4, f"Historical chronological trend of {num_c}."))
    elif len(num_cols) >= 2:
        target_num2 = num_cols[1]
        fig4 = px.histogram(df, x=target_num2, marginal="box", title=f"Distribution of {target_num2}",
                            color_discrete_sequence=['#EC4899'])
        fig4.update_layout(template="plotly_dark", height=360)
        charts.append((f"Distribution Analysis: {target_num2}", fig4, f"Histogram and Box plot showing spread of {target_num2}."))
    elif len(cat_cols) >= 2:
        target_cat2 = cat_cols[1]
        top_cats2 = df[target_cat2].value_counts().head(10).reset_index()
        top_cats2.columns = [target_cat2, 'Count']
        fig4 = px.bar(top_cats2, x=target_cat2, y='Count', title=f"Top Categories in {target_cat2}")
        fig4.update_layout(template="plotly_dark", height=360)
        charts.append((f"Category Breakdown: {target_cat2}", fig4, f"Top items in {target_cat2}."))

    # 5. Chart 5: Correlation Matrix Heatmap or Donut Pie Chart
    if len(num_cols) >= 2:
        corr_matrix = df[num_cols].corr().round(2)
        fig5 = px.imshow(corr_matrix, text_auto=True, color_continuous_scale='RdBu_r',
                         title="Numeric Correlation Matrix Heatmap")
        fig5.update_layout(template="plotly_dark", height=360)
        charts.append(("Correlation Matrix Heatmap", fig5, "Pairwise correlation coefficients across numerical columns."))
    elif cat_cols:
        target_cat = cat_cols[0]
        top_cats = df[target_cat].value_counts().head(6).reset_index()
        top_cats.columns = [target_cat, 'Count']
        fig5 = px.pie(top_cats, names=target_cat, values='Count', hole=0.4, title=f"Proportion Share: {target_cat}")
        fig5.update_layout(template="plotly_dark", height=360)
        charts.append((f"Proportion Share: {target_cat}", fig5, f"Donut chart showing categorical percentage breakdown of {target_cat}."))

    # Fallback to ensure at least 5 charts exist
    while len(charts) < 5:
        idx = len(charts) + 1
        col_to_plot = df.columns[(idx - 1) % len(df.columns)]
        fig_extra = px.bar(df[col_to_plot].value_counts().head(8).reset_index(), title=f"Overview of {col_to_plot}")
        fig_extra.update_layout(template="plotly_dark", height=360)
        charts.append((f"Auto Chart {idx}: {col_to_plot}", fig_extra, f"Automatic overview breakdown for column {col_to_plot}."))

    return charts[:6]


def handle_eda_and_visualization():
    st.header("📊 EDA & Visualizations")

    df = get_current_df()
    if df is None or df.empty:
        st.warning("⚠️ Please load a dataset first in the Data Ingestion tab.")
        return

    eda_agent: EDAAgent = st.session_state.eda_agent
    viz_agent: VisualizationAgent = st.session_state.orchestrator.viz_agent

    tab_auto_charts, tab_eda, tab_custom_viz = st.tabs([
        "⚡ Automated 5-Chart Dashboard",
        "📈 EDA Summary & Statistics",
        "🎨 Interactive Chart Builder"
    ])

    with tab_auto_charts:
        st.subheader("⚡ Automated Multi-Chart Dashboard (5+ Auto-Generated Charts)")
        st.caption("Automatically generates key interactive Plotly visualizations directly from your uploaded dataset.")
        
        auto_charts = generate_automatic_5_charts(df)
        
        col_a, col_b = st.columns(2)
        for idx, (chart_title, fig, desc) in enumerate(auto_charts):
            target_col = col_a if idx % 2 == 0 else col_b
            with target_col:
                st.markdown(f"#### {idx+1}. {chart_title}")
                st.caption(desc)
                if fig:
                    st.plotly_chart(fig, use_container_width=True, key=f"auto_chart_{idx}")

    with tab_eda:
        st.subheader("Automated EDA Summary & Statistics")
        with st.spinner("Generating EDA analysis..."):
            eda_info = eda_agent.generate_eda_summary(df)

            st.write("### Key Dataset Patterns")
            for p in eda_info.get("patterns", []):
                st.info(f"• {p}")

            num_cols = eda_info.get("numeric_columns", [])
            cat_cols = eda_info.get("categorical_columns", [])

            if num_cols:
                st.subheader("Numerical Columns Summary")
                st.dataframe(df[num_cols].describe().T, use_container_width=True)

            if cat_cols:
                st.subheader("Categorical Columns Overview")
                cat_info = []
                for c in cat_cols:
                    cat_info.append({
                        "Column": c,
                        "Unique Values": df[c].nunique(),
                        "Top Value": df[c].mode().iloc[0] if len(df[c].mode()) > 0 else None,
                        "Top Frequency": df[c].value_counts().iloc[0] if len(df) > 0 else 0
                    })
                st.dataframe(pd.DataFrame(cat_info), use_container_width=True)

            if len(num_cols) >= 2:
                st.subheader("Correlation Heatmap")
                fig_corr = viz_agent.visualizer.create_heatmap(df, title="Numeric Correlation Matrix")
                st.plotly_chart(fig_corr, use_container_width=True)

    with tab_custom_viz:
        st.subheader("Custom Plotly Chart Generator")
        chart_type = st.selectbox("Select Chart Type", ["bar", "line", "scatter", "histogram", "box", "pie", "heatmap"])
        
        all_cols = df.columns.tolist()
        x_col = st.selectbox("X-Axis / Category Column", all_cols)
        y_col = st.selectbox("Y-Axis / Metric Column (Optional)", [None] + all_cols)
        color_col = st.selectbox("Color / Group Column (Optional)", [None] + all_cols)
        title_input = st.text_input("Chart Title", f"{chart_type.title()} Chart of {x_col}")

        if st.button("📈 Render Custom Chart"):
            chart_config = {
                "chart_type": chart_type,
                "x_column": x_col,
                "y_column": y_col,
                "color_column": color_col,
                "title": title_input
            }
            fig = viz_agent.render_visualization(df, chart_config)
            if fig:
                st.plotly_chart(fig, use_container_width=True)


def handle_report_generation():
    st.header("📄 Final Analysis Report Generation")

    df = get_current_df()
    if df is None or df.empty:
        st.warning("⚠️ Please load a dataset first in the Data Ingestion tab.")
        return

    report_agent: ReportAgent = st.session_state.report_agent
    cleaning_agent: CleaningAgent = st.session_state.cleaning_agent

    st.subheader("Generate Structured Report")
    st.caption("Includes Dataset Overview, Health Assessment, Descriptive Statistics, Q&A Findings, and Groq Recommendations.")

    if st.button("📄 Generate Report Now"):
        with st.spinner("Generating analysis report..."):
            quality_info = cleaning_agent.assess_quality(df)
            qa_hist = st.session_state.chat_history
            dataset_name = st.session_state.active_dataset_name

            report_md = report_agent.generate_report(df, quality_info, qa_hist, dataset_name=dataset_name)
            st.session_state.generated_report = report_md
            st.session_state.current_step = max(st.session_state.current_step, 6)
            st.success("✅ Report generated successfully!")

    if 'generated_report' in st.session_state:
        st.markdown("---")
        st.download_button(
            label="📥 Download Report (.md)",
            data=st.session_state.generated_report,
            file_name="Natural_Language_Data_Analyst_Report.md",
            mime="text/markdown"
        )
        st.markdown("### Report Preview")
        st.markdown(st.session_state.generated_report)

def handle_settings():
    st.header("⚙️ Settings & Groq Configuration")

    st.subheader("🤖 Groq API Status & Key Setup")
    groq_service: GroqService = st.session_state.groq_service

    if groq_service.is_configured():
        st.success(f"✅ Groq API Key Configured | Active Model: `{groq_service.model}`")
    else:
        st.warning("⚠️ GROQ_API_KEY is not detected. Please enter your key below or save in `.env.example` / `.env`")

    new_key = st.text_input("Enter/Update Groq API Key:", type="password", placeholder="gsk_...")
    if st.button("💾 Save Key to Session & .env"):
        if new_key:
            st.session_state.groq_service = GroqService(api_key=new_key)
            st.session_state.orchestrator = OrchestratorAgent(st.session_state.groq_service)
            
            # Save to .env
            env_path = os.path.join(os.path.dirname(__file__), '.env')
            with open(env_path, 'w', encoding='utf-8') as f:
                f.write(f"GROQ_API_KEY={new_key}\nGROQ_MODEL=qwen/qwen3.8-27b\n")
            
            st.success("✅ Groq API Key saved successfully!")
            st.rerun()

    if st.button("🔍 Test Groq Connection"):
        with st.spinner("Testing Groq API connection..."):
            res = groq_service.validate_connection()
            if res.get("valid"):
                st.success(f"✅ {res['message']}")
                st.info(res['details'])
            else:
                st.error(f"❌ {res['message']}")
                st.warning(res['details'])

    st.markdown("---")
    if st.button("🗑️ Reset Application Data"):
        st.session_state.data = None
        st.session_state.cleaned_data = None
        st.session_state.chat_history = []
        st.session_state.current_step = 1
        st.session_state.active_tab = "📥 Data Ingestion"
        st.success("✅ All application state cleared!")
        st.rerun()

if __name__ == "__main__":
    main()
