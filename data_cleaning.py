# app.py - Data Cleaning Assistant with LLM and Sweetviz
import os
import pandas as pd
import sweetviz as sv
import streamlit as st
import json
import requests
import re
import google.generativeai as genai
import os
from datetime import datetime
import time
from typing import Any
import numpy as np
import random
import groq

# Patch numpy to add VisibleDeprecationWarning if missing, for Sweetviz compatibility with numpy 2.0+
if not hasattr(np, 'VisibleDeprecationWarning'):
    np.VisibleDeprecationWarning = DeprecationWarning

# Configuration 
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# GROQ_API_KEY = "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"  
#GEMINI_API_KEY = "AIzaSyDZlH8-uYeKK3p5Zh4GdVjrMj3uyTwlwbM"


GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# File upload and processing
def handle_file_upload():
    st.sidebar.header("Dataset Upload")
    uploaded_file = st.sidebar.file_uploader(
        "Choose a file", 
        type=["csv", "xlsx", "json"],
        help="Supported formats: CSV, Excel, JSON"
    )
    sample_limit = st.sidebar.number_input(
        "Max rows to analyze (0 = all)", min_value=0, max_value=1000000, value=2000, step=100,
        help="For large files, only the first N rows will be analyzed. Set to 0 for all rows."
    )
    if uploaded_file is not None:
        try:
            file_ext = os.path.splitext(uploaded_file.name)[1].lower()
            if file_ext == '.csv':
                df = pd.read_csv(uploaded_file)
            elif file_ext == '.xlsx':
                df = pd.read_excel(uploaded_file)
            elif file_ext == '.json':
                df = pd.read_json(uploaded_file)
            else:
                st.error("Unsupported file format")
                return None
            if sample_limit > 0 and len(df) > sample_limit:
                st.warning(f"Dataset has {len(df)} rows. Only the first {sample_limit} rows will be analyzed.")
                df = df.head(sample_limit)
            return df, uploaded_file.name
        except Exception as e:
            st.error(f"Error loading file: {e}")
            return None
    return None

# Preprocess mixed-type columns
def preprocess_mixed_columns(df):
    for col in df.columns:
        # Check for mixed types (object dtype with more than one type in non-null values)
        if df[col].dtype == 'O':
            types = set(type(x) for x in df[col].dropna())
            if len(types) > 1:
                # Try to convert to numeric, else fallback to string
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except Exception:
                    df[col] = df[col].astype(str)
    return df

# Generate Sweetviz report
def generate_sweetviz_report(df, dataset_name):
    try:
        df = preprocess_mixed_columns(df)
        os.makedirs("reports", exist_ok=True)
        report = sv.analyze(df)
        report_file = f"reports/{dataset_name}_report.html"
        report.show_html(report_file, open_browser=False)
        return report_file
    except Exception as e:
        st.error(f"Error generating Sweetviz report: {e}")
        return None

# Generate Sweetviz JSON report
def generate_sweetviz_json_report(df, dataset_name):
    try:
        df = preprocess_mixed_columns(df)
        os.makedirs("reports", exist_ok=True)
        report = sv.analyze(df)
        # Extract summary stats for each column
        summary = {}
        for col in df.columns:
            s = df[col]
            summary[col] = {
                'type': str(s.dtype),
                'unique': int(s.nunique()),
                'missing': int(s.isna().sum()),
                'min': s.min() if pd.api.types.is_numeric_dtype(s) else None,
                'max': s.max() if pd.api.types.is_numeric_dtype(s) else None,
                'mean': s.mean() if pd.api.types.is_numeric_dtype(s) else None,
                'std': s.std() if pd.api.types.is_numeric_dtype(s) else None,
                'top_values': s.value_counts().head(5).to_dict()
            }
        json_path = f"reports/{dataset_name}_report.json"
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2)
        return json_path, summary
    except Exception as e:
        st.error(f"Error generating Sweetviz JSON report: {e}")
        return None, None

# Generate pandas-based profile dictionary for LLM prompt (robust to all environments)
def generate_profile_dict_for_llm(df):
    profile = {}
    for col in df.columns:
        s = df[col]
        profile[col] = {
            'type': str(s.dtype),
            'unique': int(s.nunique()),
            'missing': int(s.isna().sum()),
            'min': s.min() if pd.api.types.is_numeric_dtype(s) else None,
            'max': s.max() if pd.api.types.is_numeric_dtype(s) else None,
            'mean': s.mean() if pd.api.types.is_numeric_dtype(s) else None,
            'std': s.std() if pd.api.types.is_numeric_dtype(s) else None,
            'top_values': s.value_counts().head(5).to_dict()
        }
    return profile

# Robust schema summary for LLM context
def get_schema_summary(df: pd.DataFrame) -> dict:
    summary = {}
    for col in df.columns:
        s = df[col]
        summary[col] = {
            'dtype': str(s.dtype),
            'nunique': int(s.nunique()),
            'missing': int(s.isna().sum()),
            'sample_values': s.dropna().unique()[:5].tolist(),
            'min': s.min() if pd.api.types.is_numeric_dtype(s) else None,
            'max': s.max() if pd.api.types.is_numeric_dtype(s) else None,
            'mean': s.mean() if pd.api.types.is_numeric_dtype(s) else None,
            'std': s.std() if pd.api.types.is_numeric_dtype(s) else None,
            'top_values': s.value_counts().head(5).to_dict()
        }
    return summary
def call_groq_with_backoff(headers, payload, max_retries=5):
    """
    Calls Groq API with exponential backoff to avoid 429 errors.
    """
    for attempt in range(max_retries):
        try:
            response = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=60)

            if response.status_code == 429:
                wait = min((2 ** attempt) + random.uniform(0.5, 1.5), 20)
                time.sleep(wait)
                continue

            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                raise e
            wait = min((2 ** attempt) + random.uniform(0.5, 1.5), 20)
            time.sleep(wait)

    return None

# LLM Integration for dirty data detection (chunked, new format)
def detect_dirty_data_with_llm(df, sweetviz_profile=None, chunk_size=50):
    st.info("Analyzing data with LLM in chunks (rate-limited)...")

    all_issues = []
    num_chunks = (len(df) + chunk_size - 1) // chunk_size
    progress = st.progress(0, text="LLM chunk analysis in progress...")
    failed_chunks = 0

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    for i in range(num_chunks):
        chunk = df.iloc[i * chunk_size:(i + 1) * chunk_size]
        data_for_llm = chunk.to_dict()

        prompt = f"""
You are a world-class data scientist and data cleaning expert.

Analyze the following dataset chunk and return ONLY valid JSON
exactly matching the schema below. Do not add explanations.

Data:
{json.dumps(data_for_llm, indent=2)}

JSON schema:
{{
  "columns": [
    {{
      "column": "column_name",
      "predicted_type": "int|float|str|date|category|other",
      "semantic_meaning": "description",
      "issues": [
        {{
          "issue_type": "missing|type_mismatch|outlier|inconsistent_format|semantic_error|duplicate|typo|illogical_value",
          "examples": ["example1"],
          "severity": "low|medium|high",
          "suggested_fix": "logical fix"
        }}
      ],
      "imputation_strategy": "meaningful strategy"
    }}
  ],
}}
"""

        payload = {
            "model": "llama-3.1-8b-instant",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 1200   # ⬅️ REDUCED (important)
        }

        try:
            result = call_groq_with_backoff(headers, payload)

            if not result:
                failed_chunks += 1
                continue

            llm_response = result["choices"][0]["message"]["content"]

            json_match = re.search(r"\{[\s\S]*\}", llm_response)
            if json_match:
                parsed = json.loads(json_match.group())
                if isinstance(parsed, dict) and "columns" in parsed:
                    all_issues.append(parsed)
                else:
                    failed_chunks += 1
            else:
                failed_chunks += 1

        except Exception as e:
            st.error(f"LLM API error on chunk {i}: {e}")
            failed_chunks += 1

        # ✅ HARD DELAY BETWEEN CHUNKS (CRITICAL)
        time.sleep(1.8)

        progress.progress(
            (i + 1) / num_chunks,
            text=f"Processed chunk {i + 1}/{num_chunks}"
        )

    progress.empty()

    if failed_chunks == num_chunks:
        st.error(
            "All LLM chunk calls failed. "
            "You are likely rate-limited. Try again later or reduce chunk count."
        )

    return all_issues
# # Generate cleaning suggestions from LLM issues
# def generate_cleaning_suggestions(issues, df, schema_summary=None):
#     st.info("Generating cleaning code  based on detected issues...")
#     gemini_key = GEMINI_API_KEY
#     if not gemini_key:
#         st.error("Gemini API key not found. Set the GEMINI_API_KEY environment variable or in config.")
#         return None
#     genai.configure(api_key=gemini_key)
#     model = genai.GenerativeModel('gemini-2.5-flash')
#     prompt = f"""
# You are a world-class data scientist and data cleaning expert.

# Here is the global schema summary:
# {json.dumps(schema_summary, indent=2) if schema_summary else ''}

# Your task: Generate a robust Python function called `clean_data(df)` to clean the dataset using the issues below.

# Guidelines:
# - Use the issues exactly as input, but apply expert judgment to fix them well.
# - Fix types, missing values (use contextual imputation), outliers, semantic errors, typos, duplicates, etc.
# - For each fix, include a comment explaining the logic.
# - Return the cleaned DataFrame.
# - Do not use bitwise operations (`|`, `&`, etc.) on non-boolean columns. Always use pandas methods that are type-safe and compatible with the column's dtype.
# - Ensure all code is valid Python and will run without syntax or type errors.

# Only return a valid JSON response in this format:
# {{
#   "code": "def clean_data(df):\\n  # cleaning steps...\\n  return df",
#   "description": "Explain what each cleaning step does and why."
# }}

# Sample data (first 20 rows):
# {json.dumps(df.head(20).to_dict(), indent=2)}

# Detected issues (from LLM):
# {json.dumps(issues, indent=2)}
#     """
#     try:
#         response = model.generate_content(prompt)
#         raw_text = response.text
#         json_match = re.search(r'\{[\s\S]*\}', raw_text)
#         if json_match:
#             try:
#                 return json.loads(json_match.group())
#             except Exception:
#                 st.warning("Failed to parse Gemini JSON block.")
#                 st.code(raw_text)
#                 return {}
#         else:
#             st.warning("No JSON block found in Gemini response.")
#             st.code(raw_text)
#             return {}
#     except Exception as e:
#         st.error(f"Gemini model error: {e}")
#         return None

def generate_cleaning_suggestions(issues, df, schema_summary=None):
    """
    Generate data cleaning code using Groq
    """
    st.info("Generating cleaning code using Groq")

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    prompt = f"""
You are a world-class data scientist and data cleaning expert.
IMPORTANT:
- Return STRICT JSON only.
- Use **double quotes** for all property names and string values.
- Escape all newlines inside strings as \\n.
- Do not use single quotes.
- Do NOT include markdown, code blocks, or text outside the JSON object.
- CRITICAL: Always check isinstance(x, str) before calling string methods like .lower(), .strip(), .replace(), .isalpha(), .isdigit()
- Do not assume all values in a column are strings - columns may contain NaN, floats, ints, etc.
- Handle NaN or numeric values gracefully by checking type first
- The code must work even if columns contain a mix of strings, numbers, or NaN
- Use pandas methods safely and avoid code that will raise exceptions
- For string operations, always use: if isinstance(x, str) and x.lower() == 'something' else x


Here is the global schema summary:
{json.dumps(schema_summary, indent=2) if schema_summary else ''}

Your task: Generate a robust Python function called `clean_data(df)` to clean the dataset using the issues below.

Guidelines:
- Use the issues exactly as input, but apply expert judgment to fix them well.
- Fix types, missing values (use contextual imputation), outliers, semantic errors, typos, duplicates, etc.
- For each fix, include a comment explaining the logic.
- Return the cleaned DataFrame.
- Do not use bitwise operations (`|`, `&`, etc.) on non-boolean columns. Always use pandas methods that are type-safe and compatible with the column's dtype.
- Ensure all code is valid Python and will run without syntax or type errors.
- Always wrap string method calls with isinstance(x, str) checks.

Only return a valid JSON response in this format:
{{
  "code": "def clean_data(df):\\n  # cleaning steps...\\n  return df",
  "description": "Explain what each cleaning step does and why."
}}

Sample data (first 20 rows):
{json.dumps(df.head(20).to_dict(), indent=2)}

Detected issues (from LLM):
{json.dumps(issues, indent=2)}
    """

    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1,
        "max_tokens": 1500
    }

    try:
        result = call_groq_with_backoff(headers, payload)

        if not result:
            st.error("Groq API returned no result.")
            return None

        raw_text = result["choices"][0]["message"]["content"]

        # Extract JSON block
        json_match = re.search(r"\{[\s\S]*\}", raw_text)
        if not json_match:
            st.warning("No JSON object found in model output.")
            st.code(raw_text)
            return {}

        json_str = json_match.group()

        # ✅ DO NOT over-replace quotes or newlines
        try:
            parsed = json.loads(json_str)
            return parsed
        except json.JSONDecodeError as e:
            st.error(f"Failed to parse JSON: {e}")
            st.code(raw_text)
            return {}



        # Extract JSON safely
        json_match = re.search(r"\{[\s\S]*\}", raw_text)
        if not json_match:
            st.warning("No JSON block found in Groq response.")
            st.code(raw_text)
            return {}

        return json.loads(json_match.group())

    except Exception as e:
        st.error(f"Groq  model error: {e}")
        return None

# Main application
def main():
    st.set_page_config(
        page_title="Data Cleaning Assistant",
        page_icon="🧹",
        layout="wide"
    )
    st.title("🧹 Data Cleaning Assistant")
    st.markdown("""
    An AI-powered tool for detecting and cleaning dirty data with:
    - **Automated profiling** with Sweetviz
    - **Smart detection** of data issues using LLMs
    - **Human-in-the-loop** validation
    - **Version control** for all changes
    """)
    st.header("1. Upload and Analyze Dataset")
    processing_tab, history_tab = st.tabs(["Process New Dataset", "View History"])
    with processing_tab:
        file_result = handle_file_upload()
        if file_result:
            df, filename = file_result
            st.success(f"Successfully loaded {filename} with {len(df)} rows and {len(df.columns)} columns")
            with st.expander("Dataset Preview"):
                st.dataframe(df)
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Data Types:**")
                    st.write(df.dtypes)
                with col2:
                    st.write("**Missing Values:**")
                    st.write(df.isna().sum())
            skip_profile = False
            if len(df) > 5000:
                skip_profile = not st.checkbox("Generate Sweetviz Profile for large dataset? (May be slow)", value=False)
            # Always generate Sweetviz profile before analysis
            sweetviz_profile = None
            if not skip_profile:
                if st.button("Generate Data Profile", help="Create comprehensive data profile with Sweetviz"):
                    with st.spinner("Generating Sweetviz report..."):
                        report_path = generate_sweetviz_report(df, os.path.splitext(filename)[0])
                        json_path, sweetviz_profile = generate_sweetviz_json_report(df, os.path.splitext(filename)[0])
                        if report_path:
                            st.success("Profile generated successfully!")
                            st.components.v1.html(open(report_path, 'r').read(), height=800, scrolling=True)
            else:
                _, sweetviz_profile = generate_sweetviz_json_report(df, os.path.splitext(filename)[0])
            if sweetviz_profile is not None:
                llm_issues_list = detect_dirty_data_with_llm(df, sweetviz_profile=sweetviz_profile)
                # Display detected dirty data in a user-friendly table
                dirty_data_rows = []
                for chunk_issues in llm_issues_list:
                    for column_issues in chunk_issues.get('columns', []):
                        column = column_issues.get('column')
                        for issue in column_issues.get('issues', []):
                            issue_type = issue.get('issue_type')
                            examples = issue.get('examples', [])
                            severity = issue.get('severity')
                            suggested_fix = issue.get('suggested_fix')
                            for example in examples:
                                dirty_data_rows.append({
                                    'column': column,
                                    'issue_type': issue_type,
                                    'example': example,
                                    'severity': severity,
                                    'suggested_fix': suggested_fix
                                })
                if dirty_data_rows:
                    st.header("Detected Dirty Data Issues")
                    st.dataframe(pd.DataFrame(dirty_data_rows))
                else:
                    st.warning("No dirty data detected by the LLM. Try a smaller chunk size or check your data.")

                # --- NEW: Generate cleaning code and clean the dataset ---
                # Aggregate all issues for the LLM
                all_dirty_issues = []
                for chunk in llm_issues_list:
                    if 'columns' in chunk:
                        for column_issues in chunk['columns']:
                            if 'issues' in column_issues:
                                all_dirty_issues.extend(column_issues['issues'])
                # Call LLM to generate cleaning code
                cleaning_result = generate_cleaning_suggestions(all_dirty_issues, df, get_schema_summary(df))
                cleaning_code = None
                if cleaning_result and 'code' in cleaning_result:
                    cleaning_code = cleaning_result['code'].strip()
                if cleaning_code:
                    st.subheader("Generated Cleaning Code")
                    #st.code(cleaning_code, language="python")
                    # Validate syntax before executing
                    try:
                        compile(cleaning_code, '<string>', 'exec')
                    except SyntaxError as se:
                        st.error(f"Syntax error in generated cleaning code: {se}\n\nGenerated code was:\n{cleaning_code}")
                    else:
                        try:
                            local_vars = {'df': df.copy()}
                            exec(cleaning_code, {'pd': pd, 'np': np, 're':re}, local_vars)
                            if 'clean_data' in local_vars:
                                cleaned_df = local_vars['clean_data'](df.copy())
                                st.success("Data cleaned successfully!")
                                st.dataframe(cleaned_df)
                                # Store cleaned_df in session state for duplicate removal and download
                                st.session_state.cleaned_df = cleaned_df
                                # Duplicate detection and removal UI
                                # ==============================
                                # DUPLICATE DETECTION & REMOVAL
                                # ==============================

                                # st.subheader("🔁 Duplicate Detection")

                                # cleaned_df = st.session_state.cleaned_df

                                # # Detect duplicates (show all duplicate rows)
                                # duplicate_rows = cleaned_df[cleaned_df.duplicated(keep=False)]
 
                                # if duplicate_rows.empty:
                                #     st.success("No duplicate rows found 🎉")
                                # else:
                                #     st.warning(f"Found {len(duplicate_rows)} duplicate rows")
                                #     st.dataframe(duplicate_rows)
                                #     remove_dupes = st.button("🗑️ Remove Duplicates")
 
                                #     if remove_dupes:
                                #         before = len(cleaned_df)
                                #         cleaned_df = cleaned_df.drop_duplicates()
                                #         after = len(cleaned_df)
                                   
                                #         st.session_state.cleaned_df = cleaned_df
                                   
                                #         st.success(f"Removed {before - after} duplicate rows")
                                # # Download button always uses session state cleaned_df
                                # # ==============================
                                # # FINAL CLEANED DATASET
                                # # ==============================

                                st.subheader("✅ Final Cleaned Dataset")

                                csv = st.session_state.cleaned_df.to_csv(index=False).encode("utf-8")

                                st.download_button(
                                    label="⬇️ Download Cleaned CSV",
                                    data=csv,
                                    file_name=f"cleaned_{filename}.csv",
                                    mime="text/csv"
                                )
                            else:
                                st.error("The generated code did not define a 'clean_data' function.")
                        except Exception as e:
                            st.error(f"Error executing cleaning code: {e}\n\nGenerated code was:\n{cleaning_code}")
                else:
                    st.warning("No cleaning code was generated by the LLM.")

if __name__ == '__main__':
    main()
