import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import requests
import json
import os
from src.config import MODEL_DIR

# --- 1. PAGE CONFIGURATION ---
# Optimize layout for wide-screen monitoring dashboards
st.set_page_config(page_title="Monitor the reactor", page_icon="🏭", layout="wide")

# --- 2. API ENDPOINT RESOLUTION ---
def get_api_url():
    """
    Resolves the API base URL.
    Prioritizes environment variables for Docker/K8s orchestration,
    falling back to localhost for local development.
    """
    return os.environ.get("API_URL", "http://localhost:8000")

API_URL = get_api_url()

# --- 3. OPERATIONAL PARAMETERS ---
# Constants defining simulation sampling and fault persistence logic
TIME_STEP_MINUTES = 3
INJECTION_TIME_MIN = 60
PERSISTENCE_LIMIT = 2

METRICS_PATH = MODEL_DIR / "metrics.json"

def load_performance_metadata():
    """
    In-memory cache for model performance artifacts.
    Provides a fallback mechanism if the evaluation artifact is missing.
    """
    if METRICS_PATH.exists():
        try:
            with open(METRICS_PATH, "r") as f:
                report = json.load(f)

            dynamic_metadata = {}
            # Extract global accuracy as a baseline reference for all fault profiles
            global_acc = report.get("accuracy", 0.0)

            for fault_id in range(1, 21):
                # Data Normalization: The JSON report uses integer-string keys (e.g., "1")
                # while the internal logic might provide floats or integers.
                str_id = str(int(fault_id))

                if str_id in report:
                    metrics = report[str_id]
                    dynamic_metadata[fault_id] = {
                        "desc": f"Fault {fault_id}",
                        "f1": metrics.get("f1-score", 0.0), # Metric produced by sklearn.metrics
                        "accuracy": global_acc,
                        "difficulty": "Dynamic",
                        "comment": f"Recall: {metrics.get('recall', 0.0):.1%}"
                    }
                else:
                    # Graceful degradation if a specific fault class is missing from the artifact
                    dynamic_metadata[fault_id] = {
                        "desc": "N/A",
                        "f1": 0.0,
                        "accuracy": 0.0,
                        "difficulty": "Unknown",
                        "comment": "No data in artifact"
                    }
            return dynamic_metadata

        except (json.JSONDecodeError, KeyError) as e:
            # Alert the user if the artifact is corrupted or malformed
            st.error(f"Artifact Parsing Error: Failed to process {METRICS_PATH.name}. {e}")

    # Fallback mechanism to ensure UI stability even if the model hasn't been evaluated yet
    return {
        i: {
            "desc": "Missing",
            "f1": 0.0,
            "accuracy": 0.0,
            "difficulty": "N/A",
            "comment": "Evaluation artifact not found"
        } for i in range(1, 21)
    }

FAULT_METADATA = load_performance_metadata()

def get_fault_info(code):
    return FAULT_METADATA.get(code, {"desc": "Standard Scenario", "f1": 0.85, "accuracy": 0.88, "difficulty": "Medium", "comment": "Standard performance."})

# --- 4. UI STYLING (CSS) ---
st.markdown("""
    <style>
    .main { background-color: #0E1117; }
    .result-box {
        background-color: #262730;
        border: 1px solid #4B4B4B;
        border-radius: 5px;
        padding: 15px;
        margin-top: 20px;
        color: white;
        text-align: center;
    }
    .result-title { font-size: 0.8em; color: #aaaaaa; margin-bottom: 2px; text-transform: uppercase; letter-spacing: 1px;}
    .result-value { font-size: 1.4em; font-weight: bold; color: #FAFAFA; margin-bottom: 12px; }
    div[data-testid="stMetricValue"] { font-size: 26px; color: #FAFAFA; }
    </style>
""", unsafe_allow_html=True)

# --- 5. REACTOR SYNOPTIC VISUALIZATION ---
def create_reactor_synoptic(current_p, current_t, current_f, base_p, base_t, base_f):
    """
    Generates a Plotly-based reactor schematic with dynamic status indicators (LEDs).
    Alerts are triggered if real-time values deviate beyond a 2% threshold from the baseline.
    """

    # Tolerance threshold for anomaly signaling
    THRESHOLD = 0.02

    # Color logic: Green for nominal operation, Red for out-of-bounds deviation
    def get_color(curr, base):
        if base == 0:
            return "#00CC96"
        dev = abs(curr - base) / base
        return "#FF4B4B" if dev > THRESHOLD else "#00CC96"

    # Evaluate sensor health status
    # Default to nominal color if baseline calibration is not yet established
    c_p = get_color(current_p, base_p) if base_p else "#00CC96"
    c_t = get_color(current_t, base_t) if base_t else "#00CC96"
    c_f = get_color(current_f, base_f) if base_f else "#00CC96"

    fig = go.Figure()

    # 1. Reactor vessel geometry (Rounded rectangle)
    fig.add_shape(type="rect", x0=3, y0=2, x1=7, y1=8, line=dict(color="white", width=2), fillcolor="rgba(255,255,255,0.1)")

    # 2. Piping schematics (Inlet/Outlet lines)
    fig.add_shape(type="line", x0=1, y0=7, x1=3, y1=7, line=dict(color="white", width=2), name="Inlet")
    fig.add_annotation(x=1, y=7, text="Feed", showarrow=False, yshift=10, font=dict(color="#aaa", size=10))

    fig.add_shape(type="line", x0=7, y0=3, x1=9, y1=3, line=dict(color="white", width=2), name="Outlet")
    fig.add_annotation(x=9, y=3, text="Product", showarrow=False, yshift=10, font=dict(color="#aaa", size=10))

    # 3. Dynamic Status Indicators (Scatter points acting as LEDs)
    # Mapping: Pressure (Top), Temperature (Core), Flow (Outlet)

    # Pressure LED
    fig.add_trace(go.Scatter(
        x=[5], y=[8], mode='markers+text',
        marker=dict(size=25, color=c_p, line=dict(width=2, color='white')),
        text=["<b>P</b>"], textposition="middle center", textfont=dict(color='white'),
        hoverinfo='text', hovertext=f"Pressure: {current_p:.1f}"
    ))
    # Temperature LED
    fig.add_trace(go.Scatter(
        x=[5], y=[5], mode='markers+text',
        marker=dict(size=25, color=c_t, line=dict(width=2, color='white')),
        text=["<b>T</b>"], textposition="middle center", textfont=dict(color='white'),
        hoverinfo='text', hovertext=f"Temperature: {current_t:.1f}"
    ))
    # Flow LED
    fig.add_trace(go.Scatter(
        x=[8], y=[3], mode='markers+text',
        marker=dict(size=25, color=c_f, line=dict(width=2, color='white')),
        text=["<b>F</b>"], textposition="middle center", textfont=dict(color='white'),
        hoverinfo='text', hovertext=f"Flow: {current_f:.1f}"
    ))

    # Canvas optimization (Minimalist layout for dashboard integration)
    fig.update_layout(
        title="Reactor Synoptic (Real-time)",
        title_font_size=14,
        width=300, height=250,
        margin=dict(l=10, r=10, t=40, b=10),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(range=[0, 10], showgrid=False, zeroline=False, visible=False),
        yaxis=dict(range=[0, 10], showgrid=False, zeroline=False, visible=False),
        showlegend=False
    )
    return fig

# --- 6. SESSION STATE INITIALIZATION ---
if 'simulation_running' not in st.session_state:
    st.session_state.simulation_running = False
if 'final_report' not in st.session_state:
    st.session_state.final_report = None
if 'final_figs' not in st.session_state:
    st.session_state.final_figs = None

# Baseline calibration values for anomaly comparison
if 'base_values' not in st.session_state:
    st.session_state.base_values = {'p': None, 't': None, 'f': None}

# --- 7. MAIN DASHBOARD LAYOUT ---
st.title("🏭 Monitor the Reactor")

col_left, col_right = st.columns([1, 4])

# === LEFT CONTROL PANEL (SIMULATION ORCHESTRATION) ===
with col_left:
    st.subheader("Control")
    scenario_options = {f"Fault #{i}": i for i in range(1, 21)}
    selected_scenario_name = st.selectbox("Select Scenario", list(scenario_options.keys()), disabled=st.session_state.simulation_running)
    selected_fault_code = scenario_options[selected_scenario_name]

    meta = get_fault_info(selected_fault_code)

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("▶️ START", type="primary", use_container_width=True):
            # Simulation State Reset
            st.session_state.simulation_running = True
            st.session_state.final_report = None
            st.session_state.final_figs = None
            st.session_state.base_values = {'p': None, 't': None, 'f': None}
            st.session_state.anomaly_detected = False
            st.session_state.anomaly_time_min = None
            st.session_state.diagnosis_confirmed = False
            st.session_state.diagnosis_time_min = None
            st.session_state.diag_consecutive_count = 0

    with col_btn2:
        if st.button("⏹️ ABORT", type="primary", use_container_width=True):
            st.session_state.simulation_running = False
            st.session_state.final_report = None
            st.session_state.final_figs = None

    status_txt = "Active" if st.session_state.simulation_running else "Ready"
    st.markdown(f"""<div class="result-box"><div class="result-title">System Status</div><div class="result-value">{status_txt}</div></div>""", unsafe_allow_html=True)

    diff_color = "#28a745" if meta['difficulty'] == "Low" else "#dc3545" if "High" in meta['difficulty'] else "#ffc107"
    st.markdown(f"""<div style="margin-top:10px; text-align:center;"><span style="background-color:{diff_color}; color:white; padding:4px 10px; border-radius:15px; font-size:0.8em;">Complexity : {meta['difficulty']}</span></div>""", unsafe_allow_html=True)

    st.divider()

    # --- SYNOPTIC VIEW (LIVE TELEMETRY) ---
    st.markdown("##### ⚙️ Reactor State")
    synoptic_spot = st.empty() # Synoptic placeholder


# === RIGHT PANEL (DATA VISUALIZATION & ANALYTICS) ===
with col_right:
    # Model Benchmarks
    meta = get_fault_info(selected_fault_code)
    with st.container():
        st.markdown("##### 📋 Performance Profile (Offline)")
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Description", meta['desc'])
        kpi2.metric("F1-Score", f"{meta['f1']:.2f}")
        kpi3.metric("Accuracy", f"{meta['accuracy']:.2f}")
        kpi4.info(f"{meta['comment']}")

    st.divider()

    st.markdown("##### 📈 Fault Diagnosis (Global View)")
    chart_main_spot = st.empty()
    st.markdown("##### 📊 Reactor Sensors (Real-time)")
    feat_c1, feat_c2, feat_c3 = st.columns(3)
    chart_feat1 = feat_c1.empty()
    chart_feat2 = feat_c2.empty()
    chart_feat3 = feat_c3.empty()

    # Persistence of visualization after simulation end
    if not st.session_state.simulation_running and st.session_state.final_figs:
        chart_main_spot.plotly_chart(st.session_state.final_figs['main'], use_container_width=True)
        chart_feat1.plotly_chart(st.session_state.final_figs['f1'], use_container_width=True)
        chart_feat2.plotly_chart(st.session_state.final_figs['f2'], use_container_width=True)
        chart_feat3.plotly_chart(st.session_state.final_figs['f3'], use_container_width=True)
        synoptic_spot.plotly_chart(st.session_state.final_figs['synoptic'], use_container_width=True)

    # Simulation Summary Report
    if st.session_state.final_report:
        st.divider()
        st.markdown("### 📝 Post-Simulation Report")
        rep = st.session_state.final_report
        m1, m2, m3 = st.columns(3)
        m1.metric("Scenario", rep['scenario'])
        m2.metric("⏱️ Detection Delay", f"{rep['detection_delay']:.1f} min" if rep['detection_delay'] else "N/A")
        m3.metric("🎯 Diagnosis Delay", f"{rep['diagnosis_delay']:.1f} min" if rep['diagnosis_delay'] else "N/A")


# ==========================
# 8. EXECUTION ENGINE (STREAMING API MODE)
# ==========================

def get_api_prediction(sensor_data):
    """
    Interfaces with the backend inference engine.
    Sends raw telemetry and retrieves detection/diagnosis results.
    """
    try:
        # Construct payload according to API schema
        payload = {"sensors": sensor_data}
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=0.8)

        if response.status_code == 200:
            return response.json()
    except Exception:
        # Fail-soft: Handle connection timeouts or API unavailability
        return None
    return None

if st.session_state.simulation_running:
    # 1. DATA INGESTION (Simulated telemetry stream)
    try:
        PATH_INPUT = "data/processed/cropped/TEP_Faulty_Testing.parquet"
        df_source = pd.read_parquet(PATH_INPUT)

        # Filter simulation run based on user selection
        simulation_data = df_source[
            (df_source['faultNumber'] == selected_fault_code) &
            (df_source['simulationRun'] == 1)
        ].reset_index(drop=True)

        if simulation_data.empty:
            st.error("Source data not found.")
            st.session_state.simulation_running = False
            st.stop()

    except Exception as e:
        st.error(f"Ingestion Failure: {e}")
        st.session_state.simulation_running = False
        st.stop()

    # Initialize buffer lists for temporal visualization
    history_pression, history_temp, history_debit, history_pred, history_time = [], [], [], [], []
    init_p, init_t, init_f = [], [], []

    # 2. STREAMING LOOP (Row-by-row API inference)
    for index, row in simulation_data.iterrows():
        if not st.session_state.simulation_running:
            break

        # Isolate sensor features (X) from metadata
        sensors_only = row.filter(regex='xmeas_|xmv_').to_dict()

        # --- REAL-TIME INFERENCE CALL ---
        prediction_result = get_api_prediction(sensors_only)

        if prediction_result:
            val_diagnosis = float(prediction_result.get("fault_code", 0))
            is_anomaly = prediction_result.get("is_anomaly", False)
            val_detector = 1.0 if is_anomaly else 0.0
        else:
            val_diagnosis = 0.0
            val_detector = 0.0

        # --- METRIC EXTRACTION ---
        val_press = row.get('xmeas_7', 0)
        val_temp = row.get('xmeas_9', 0)
        val_debit = row.get('xmeas_10', 0)
        val_sample = row.get('sample', index)

        current_time_hours = (val_sample * TIME_STEP_MINUTES) / 60
        current_time_minutes = val_sample * TIME_STEP_MINUTES

        # Hide diagnosis results during initial stabilization window
        display_pred = 0.0 if current_time_hours < 1.0 else val_diagnosis

        # --- BASELINE CALIBRATION (Cold Start Phase) ---
        # Capture average nominal values during the first 5 samples
        if index < 5:
            init_p.append(val_press)
            init_t.append(val_temp)
            init_f.append(val_debit)
        elif index == 5:
            st.session_state.base_values['p'] = sum(init_p)/len(init_p)
            st.session_state.base_values['t'] = sum(init_t)/len(init_t)
            st.session_state.base_values['f'] = sum(init_f)/len(init_f)

        # --- EVENT DETECTION LOGIC ---
        if current_time_minutes > INJECTION_TIME_MIN:
            # Capture first anomaly detection event
            if not st.session_state.anomaly_detected and val_detector > 0.5:
                st.session_state.anomaly_detected = True
                st.session_state.anomaly_time_min = current_time_minutes

            # Confirm diagnosis only after consecutive persistent matches
            if not st.session_state.diagnosis_confirmed:
                if round(val_diagnosis) == selected_fault_code:
                    st.session_state.diag_consecutive_count += 1
                else:
                    st.session_state.diag_consecutive_count = 0
                if st.session_state.diag_consecutive_count >= PERSISTENCE_LIMIT:
                    st.session_state.diagnosis_confirmed = True
                    st.session_state.diagnosis_time_min = current_time_minutes

        history_pression.append(val_press)
        history_temp.append(val_temp)
        history_debit.append(val_debit)
        history_pred.append(display_pred)
        history_time.append(current_time_hours)

        # --- CHART REFRESH CYCLE ---

        fig1 = go.Figure(go.Scatter(x=history_time, y=history_pression, mode='lines', line=dict(color='cyan')))
        fig1.update_layout(height=180, margin=dict(t=30,b=10,l=10,r=10), title="Pressure", xaxis_title="Hours", template="plotly_dark")
        chart_feat1.plotly_chart(fig1, use_container_width=True, key=f"f1_{index}")

        fig2 = go.Figure(go.Scatter(x=history_time, y=history_temp, mode='lines', line=dict(color='orange')))
        fig2.update_layout(height=180, margin=dict(t=30,b=10,l=10,r=10), title="Temperature", xaxis_title="Hours", template="plotly_dark")
        chart_feat2.plotly_chart(fig2, use_container_width=True, key=f"f2_{index}")

        fig3 = go.Figure(go.Scatter(x=history_time, y=history_debit, mode='lines', line=dict(color='#00FF00')))
        fig3.update_layout(height=180, margin=dict(t=30,b=10,l=10,r=10), title="Flow", xaxis_title="Hours", template="plotly_dark")
        chart_feat3.plotly_chart(fig3, use_container_width=True, key=f"f3_{index}")

        fig_main = go.Figure()
        fig_main.add_trace(go.Scatter(x=history_time, y=history_pred, mode='lines', name='Diag', line=dict(color='#FF4B4B', width=2)))
        fig_main.add_vrect(x0=0, x1=1, fillcolor="gray", opacity=0.3, line_width=0, annotation_text="STABILIZATION", annotation_position="top left", annotation_font_color="white")
        fig_main.update_layout(title="Fault Diagnosis Code", height=250, margin=dict(t=30,b=20,l=20,r=20), paper_bgcolor="rgba(0,0,0,0)", template="plotly_dark", xaxis=dict(title="Hours", range=[0, max(1.1, current_time_hours + 0.1)]), yaxis=dict(title="Fault Code", visible=True, automargin=True))
        chart_main_spot.plotly_chart(fig_main, use_container_width=True, key=f"main_{index}")

        # Refresh Synoptic View
        fig_syn = create_reactor_synoptic(val_press, val_temp, val_debit,
                                          st.session_state.base_values.get('p'),
                                          st.session_state.base_values.get('t'),
                                          st.session_state.base_values.get('f'))
        synoptic_spot.plotly_chart(fig_syn, use_container_width=True, key=f"syn_{index}")

    # --- SIMULATION FINALIZATION & REPORTING ---
    if st.session_state.simulation_running:
        d_det, d_diag, det_time_h, diag_time_h = None, None, None, None
        if st.session_state.anomaly_detected:
            d_det = max(0, st.session_state.anomaly_time_min - INJECTION_TIME_MIN)
            det_time_h = st.session_state.anomaly_time_min / 60
        if st.session_state.diagnosis_confirmed:
            d_diag = max(0, st.session_state.diagnosis_time_min - INJECTION_TIME_MIN)
            diag_time_h = st.session_state.diagnosis_time_min / 60

        st.session_state.final_report = {"scenario": selected_scenario_name, "detection_delay": d_det, "diagnosis_delay": d_diag}

        def add_markers_to_fig(fig_to_mark):
            """Appends event markers to charts for post-run analysis."""
            if det_time_h:
                fig_to_mark.add_vline(x=det_time_h, line_width=2, line_dash="dash", line_color="orange", annotation_text="Detection", annotation_position="top right")
            if diag_time_h:
                pos = "bottom right" if (det_time_h and abs(diag_time_h - det_time_h) < 0.1) else "top right"
                fig_to_mark.add_vline(x=diag_time_h, line_width=2, line_dash="solid", line_color="red", annotation_text="Diagnosis", annotation_position=pos)
            return fig_to_mark

        # Persist final state for UI consistency
        st.session_state.final_figs = {'main': add_markers_to_fig(fig_main), 'f1': fig1, 'f2': fig2, 'f3': fig3, 'synoptic': fig_syn}
        st.session_state.simulation_running = False
        st.rerun()
