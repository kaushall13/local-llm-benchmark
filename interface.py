#imports
import streamlit as st
import pandas as pd
import plotly.express as px
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="LLM Benchmark Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Helper Functions ---
@st.cache_data
def load_data(uploaded_file):
    """Load data from the uploaded CSV file and prepare it for analysis."""
    try:
        df = pd.read_csv(uploaded_file)
        # Standardize column names that might vary
        rename_map = {
            'model': 'model_name',
            'context': 'context_size',
            'gen_len': 'gen_length',
            'ttft': 'ttft_s',
            'ram_peak': 'ram_peak_mb',
            'vram_peak': 'vram_peak_mb'
        }
        df.rename(columns=lambda c: rename_map.get(c, c), inplace=True)
        
        # Ensure correct data types
        for col in ['tps', 'tpm', 'ttft_s', 'ram_peak_mb', 'vram_peak_mb', 'context_size', 'gen_length']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Drop rows where essential data is missing
        df.dropna(subset=['tps', 'model_name', 'context_size', 'gen_length'], inplace=True)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def get_filtered_data(df, models, quants, context_range, gen_range, run_mode):
    """Apply all user-defined filters to the DataFrame."""
    filtered_df = df[
        df['model_name'].isin(models) &
        df['quant'].isin(quants) &
        df['context_size'].between(context_range[0], context_range[1]) &
        df['gen_length'].between(gen_range[0], gen_range[1])
    ].copy()

    if run_mode == "CPU Only":
        filtered_df = filtered_df[filtered_df['ngl'] == 0]
    elif run_mode == "GPU Only":
        filtered_df = filtered_df[filtered_df['ngl'] != 0]
        
    return filtered_df

# --- Main Application ---
st.title("🚀 LLM Benchmark Visualizer")
st.markdown("Upload your `benchmark_results_*.csv` file to interactively analyze performance.")

# --- File Uploader ---
uploaded_file = st.file_uploader("Choose a benchmark CSV file", type="csv")

if uploaded_file is None:
    st.info("Please upload a CSV file to begin.")
else:
    df = load_data(uploaded_file)

    if df is not None and not df.empty:
        # --- Sidebar Filters ---
        with st.sidebar:
            st.header("🔍 Filters")
            
            selected_models = st.multiselect("Model", sorted(df["model_name"].unique()), default=sorted(df["model_name"].unique()))
            
            selected_quants = st.multiselect("Quantization", sorted(df["quant"].unique()), default=sorted(df["quant"].unique()))
            
            run_mode = st.radio("Run Mode", ["All", "CPU Only", "GPU Only"])
            
            # Create sliders only if there is a range of values to select from
            context_min, context_max = int(df["context_size"].min()), int(df["context_size"].max())
            if context_min < context_max:
                context_range = st.slider("Context Size", context_min, context_max, (context_min, context_max))
            else:
                st.markdown(f"**Context Size:** `{context_min}` (only one value in data)")
                context_range = (context_min, context_max)

            gen_len_min, gen_len_max = int(df["gen_length"].min()), int(df["gen_length"].max())
            if gen_len_min < gen_len_max:
                gen_range = st.slider("Generation Length", gen_len_min, gen_len_max, (gen_len_min, gen_len_max))
            else:
                st.markdown(f"**Generation Length:** `{gen_len_min}` (only one value in data)")
                gen_range = (gen_len_min, gen_len_max)


        # Apply filters
        filtered_df = get_filtered_data(df, selected_models, selected_quants, context_range, gen_range, run_mode)

        if filtered_df.empty:
            st.warning("No data matches the selected filters. Please adjust your selection.")
        else:
            # --- KPI Summary ---
            st.subheader("📊 Performance Metrics (Averages for Filtered Data)")
            kpi_cols = st.columns(5)
            kpi_cols[0].metric("Avg. TPS", f"{filtered_df['tps'].mean():.2f}")
            kpi_cols[1].metric("Avg. TPM", f"{filtered_df['tpm'].mean():.2f}")
            kpi_cols[2].metric("Avg. TTFT (s)", f"{filtered_df['ttft_s'].mean():.2f}")
            kpi_cols[3].metric("Max RAM (MB)", f"{filtered_df['ram_peak_mb'].max():.0f}")
            
            # Handle case where there is no VRAM data
            vram_max = filtered_df['vram_peak_mb'].max()
            kpi_cols[4].metric("Max VRAM (MB)", f"{vram_max:.0f}" if pd.notna(vram_max) and vram_max > 0 else "N/A")


            # --- Chart Visualizations ---
            st.subheader("📈 Token Performance Analysis")
            c1, c2 = st.columns(2)

            with c1:
                fig_tps = px.box(
                    filtered_df, x="model_name", y="tps", color="model_name",
                    labels={'tps': 'Tokens per Second', 'model_name': 'Model'},
                    title="<b>Tokens Per Second (TPS) Distribution</b>"
                )
                st.plotly_chart(fig_tps, use_container_width=True)

            with c2:
                fig_ttft = px.box(
                    filtered_df, x="model_name", y="ttft_s", color="model_name",
                    labels={'ttft_s': 'Time to First Token (seconds)', 'model_name': 'Model'},
                    title="<b>Time to First Token (TTFT) Distribution</b>"
                )
                st.plotly_chart(fig_ttft, use_container_width=True)

            st.subheader("📉 Resource Usage Analysis")
            c3, c4 = st.columns(2)

            with c3:
                fig_ram = px.box(
                    filtered_df, x="model_name", y="ram_peak_mb", color="model_name",
                    labels={'ram_peak_mb': 'Peak RAM Usage (MB)', 'model_name': 'Model'},
                    title="<b>Peak RAM Usage Distribution</b>"
                )
                st.plotly_chart(fig_ram, use_container_width=True)

            with c4:
                # Only show VRAM chart if there's GPU data
                gpu_data = filtered_df[filtered_df['ngl'] != 0]
                if not gpu_data.empty and gpu_data['vram_peak_mb'].notna().any():
                    fig_vram = px.box(
                        gpu_data, x="model_name", y="vram_peak_mb", color="model_name",
                        labels={'vram_peak_mb': 'Peak VRAM Usage (MB)', 'model_name': 'Model'},
                        title="<b>Peak VRAM Usage Distribution (GPU Runs)</b>"
                    )
                    st.plotly_chart(fig_vram, use_container_width=True)
                else:
                    st.info("No GPU data in the current selection to display VRAM usage.")

            # --- Raw Data Viewer ---
            with st.expander("View Filtered Benchmark Data"):
                st.dataframe(filtered_df.sort_values(by=["model_name", "context_size", "gen_length"]))

    else:
        st.error("The uploaded CSV file is empty or could not be processed. Please check the file.")
