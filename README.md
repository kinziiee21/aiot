# 📊 Natural Language Data Analyst

> An AI-powered agentic data analysis application that enables users to upload datasets, ask questions in natural language, and receive automated data cleaning, Pandas computations, Plotly visualizations, AI insights, and comprehensive analysis reports.

---

## 📌 Problem Statement

Traditional data analysis requires proficiency in programming languages (Python, R) or SQL query syntax to extract meaningful business insights from raw tabular data. Non-technical stakeholders often struggle to clean data, select appropriate chart types, or perform complex aggregations. 

**Natural Language Data Analyst** bridges this gap by acting as an intelligent AI assistant. Users simply upload a CSV or XLSX file and ask questions in plain English (e.g., *"Which region has the highest revenue?"*). The system uses an agentic architecture powered by Groq LLM to generate structured analysis plans, execute safe Pandas computations, render Plotly charts, and deliver data-driven insights without exposing raw code execution risks.

---

## 🎯 Project Objectives

- **Natural Language Interaction**: Translate plain-text user queries into structured analysis plans.
- **Agentic Architecture**: Coordinate specialized modular agents (Orchestrator, Cleaning, EDA, Analysis, Visualization, Report).
- **Safe Pandas Execution**: Compute data statistics directly with Pandas/DataFrame operations using validated JSON plans instead of arbitrary code execution.
- **Intelligent Data Cleaning**: Automatically detect and standardize numeric strings with currency formatting (`₹2,500`, `$1,200`, `1,500`), missing values, duplicates, and dates.
- **High-Performance LLM Integration**: Utilize Groq's high-speed LLM infrastructure (`llama-3.3-70b-versatile`) with automatic rule-based fallbacks.
- **Automated Reporting**: Produce downloadable, formatted analysis reports summarizing dataset health, statistics, and Q&A findings.

---

## 🧩 Technology Stack

| Category | Tools & Libraries |
|----------|------------------|
| **Core Language** | Python 3.9+ |
| **Frontend Framework** | Streamlit |
| **Data Manipulation** | Pandas, NumPy |
| **Visualization** | Plotly Express, Plotly Graph Objects |
| **LLM Service Provider** | Groq API (`groq` Python SDK) |
| **Default LLM Model** | `llama-3.3-70b-versatile` |
| **Profiling & Reports** | `ydata-profiling` |
| **Environment & Utils** | `python-dotenv`, `openpyxl`, `SQLAlchemy` |

---

## 🏗️ System Architecture Workflow

```
USER
  │
  ▼
STREAMLIT UI (Natural Language Input)
  │
  ▼
ORCHESTRATOR AGENT (agents/orchestrator.py)
  │
  ├────────► GROQ LLM SERVICE (services/groq_service.py)
  │            Generate Structured Analysis Plan (JSON)
  │
  ├────────► ANALYSIS AGENT (agents/analysis_agent.py)
  │            Validate & Execute Safe Pandas Operations
  │
  ├────────► VISUALIZATION AGENT (agents/visualization_agent.py)
  │            Generate Plotly Charts (Bar, Line, Scatter, Pie, Heatmap)
  │
  ├────────► GROQ LLM SERVICE (services/groq_service.py)
  │            Generate Data-Driven Insights (Based strictly on computed results)
  │
  ▼
FINAL RESPONSE (Answer, Chart, Key Insights, Report)
```

---

## 🤖 Agent Roles & Responsibilities

1. **Orchestrator Agent (`agents/orchestrator.py`)**:
   - Manages overall workflow execution.
   - Coordinates between Groq service, Analysis Agent, and Visualization Agent.
2. **Cleaning Agent (`agents/cleaning_agent.py`)**:
   - Assesses dataset quality (missing values, duplicate rows, data type issues).
   - Detects formatted numeric strings (`₹2,500`, `$1,200`, `1,500`) and date columns.
   - Recommends and applies user-approved data cleaning operations.
3. **EDA Agent (`agents/eda_agent.py`)**:
   - Generates automated descriptive statistics for numerical, categorical, and datetime columns.
   - Identifies outliers, correlation matrices, and distribution patterns.
4. **Analysis Agent (`agents/analysis_agent.py`)**:
   - Parses structured JSON analysis plans (`groupby`, `operation`, `filters`, `limit`).
   - Executes deterministic Pandas calculations without invoking risky `eval()` or `exec()` code execution.
5. **Visualization Agent (`agents/visualization_agent.py`)**:
   - Matches semantic query intent with dataset column types.
   - Selects optimal Plotly chart types (bar, line, scatter, histogram, box plot, pie, heatmap).
6. **Report Agent (`agents/report_agent.py`)**:
   - Compiles dataset statistics, cleaning reports, and Q&A history into downloadable Markdown reports.

---

## 📁 Project Structure

```
Data-Analytics-AI-Agent/
│
├── app.py                     # Streamlit Main Application UI
│
├── agents/                    # Agentic Architecture
│   ├── __init__.py
│   ├── orchestrator.py        # Central Orchestrator
│   ├── cleaning_agent.py      # Quality & Cleaning Agent
│   ├── eda_agent.py           # Exploratory Data Analysis Agent
│   ├── analysis_agent.py      # Safe Pandas Computation Agent
│   ├── visualization_agent.py # Plotly Chart Agent
│   └── report_agent.py        # Markdown Report Generation Agent
│
├── services/                  # External API Services
│   ├── __init__.py
│   └── groq_service.py        # Groq LLM API Wrapper & Prompts
│
├── modules/                   # Core Processing & Database Modules
│   ├── __init__.py
│   ├── data_ingestion.py      # CSV/XLSX/API loading
│   ├── data_cleaning.py       # Data cleaning engine & string parser
│   ├── database_manager.py    # SQLite database connection manager
│   ├── profiling.py           # ydata-profiling wrapper
│   ├── visualization.py       # Plotly chart utilities
│   └── version_control.py     # Dataset snapshot versioning
│
├── utils/                     # Helper Functions & Formatting
│   └── helpers.py
│
├── .env.example               # Environment Variables Template
├── .gitignore
├── requirements.txt           # Python Dependencies
├── LICENSE                    # MIT License
└── README.md                  # Project Documentation
```

---

## ⚙️ Installation & Setup

### 1. Prerequisites
Ensure you have Python 3.9+ installed on your system.

### 2. Clone Repository & Navigate to Directory
```bash
git clone https://github.com/your-username/Data-Analytics-AI-Agent.git
cd Data-Analytics-AI-Agent
```

### 3. Create & Activate Virtual Environment
- **Windows**:
  ```powershell
  python -m venv venv
  .\venv\Scripts\activate
  ```
- **macOS / Linux**:
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Configure Groq API Key
Create a `.env` file in the project root based on `.env.example`:
```env
GROQ_API_KEY=your_actual_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile
```
*(Get a free API key at [console.groq.com](https://console.groq.com/keys))*

---

## 🚀 How to Run Application

Start the Streamlit application with:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 💡 Example Natural Language Questions

- *"What is the average sales?"*
- *"Which product has the highest revenue?"*
- *"Which region generated the most sales?"*
- *"Compare sales between regions."*
- *"Show the monthly sales trend."*
- *"Are there any unusual values?"*
- *"What are the main insights from this dataset?"*
- *"Which category performs best?"*
- *"Show me the relationship between age and income."*

---

## 🔄 Project Workflow

1. **Upload Dataset**: Upload any CSV or XLSX file. Review metric cards (rows, columns, missing cells, memory).
2. **Review & Clean**: Inspect data quality score. Approve recommended cleaning operations (e.g., convert `₹1,500` / `$2,500` strings to numbers, remove duplicates).
3. **Ask Questions**: Type natural language questions. View structured JSON plan, computed table, Plotly chart, and AI insights.
4. **Explore EDA**: View correlation heatmaps, numerical distributions, and generate full `ydata-profiling` HTML reports.
5. **Generate Report**: Download a complete Markdown analysis report combining statistics and Q&A findings.

---

## ⚠️ Limitations & Future Scope

### Current Limitations:
- Natural language analysis relies on structured Pandas aggregations; very complex multi-join relational logic requires standard SQL queries.
- HTML profiling generation for extremely large datasets (>100k rows) can take several seconds.

### Future Scope:
- Integration with cloud databases (Google BigQuery, PostgreSQL).
- Automated voice input for asking natural language questions.
- Scheduled email report exports.

---

## 📜 License & Attribution

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Original Author & Foundation:** David Singh (2025)  
**Enhanced Rebrand:** TYBCA Academic Project - Natural Language Data Analyst (2026)