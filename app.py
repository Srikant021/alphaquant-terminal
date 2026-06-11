# streamlit_app.py
import streamlit as st
from io import BytesIO
import time
import base64
import traceback

# Import your plotting functions (assumes they are in the same repo)
# If you kept the long single-file app, move plotting functions into a module (nifty_tools.py)
# For this example we import from the same file names used earlier.
from import_streamlit_as_st import (  # replace with actual module name if you refactored
    plot_nifty_volatility_st,
    plot_expected_move_st,
    plot_index_divergence_st,
    plot_volatility_cone_st,
    plot_fyers_oi_profile_st,
    plot_vrp_st,
    plot_hurst_regime_st,
    plot_liquidity_sweep_st,
    plot_parkinson_estimator_st,
    generate_fyers_token_streamlit,
)

# ---------- App config ----------
st.set_page_config(page_title="Nifty Live Tools", layout="wide", initial_sidebar_state="expanded")

# ---------- Sidebar ----------
with st.sidebar:
    st.title("Controls")
    st.markdown("**Select tools to show**")
    show_ivr = st.checkbox("IVR & IVP (India VIX)", value=True)
    show_expected = st.checkbox("Expected Move", value=True)
    show_corr = st.checkbox("Index Correlation", value=False)
    show_cone = st.checkbox("Volatility Cone", value=False)
    show_fyers = st.checkbox("FYERS OI Profile", value=False)
    show_vrp = st.checkbox("VRP", value=False)
    show_hurst = st.checkbox("Hurst Exponent", value=False)
    show_liquidity = st.checkbox("Liquidity Sweep", value=False)
    show_parkinson = st.checkbox("Parkinson Estimator", value=False)

    st.markdown("---")
    st.subheader("Refresh & caching")
    auto_refresh = st.checkbox("Auto refresh", value=False)
    refresh_interval = st.selectbox("Interval (minutes)", [1, 2, 5, 10, 30], index=2)
    if st.button("Refresh now"):
        st.cache_data.clear()
        st.experimental_rerun()

    st.markdown("---")
    st.subheader("FYERS / Secrets")
    st.info("For production, set FYERS tokens in Streamlit Secrets (App settings).")
    fyers_token = st.text_input("FYERS Access Token (optional)", value="", type="password")
    fyers_client_id = st.text_input("FYERS Client ID (optional)", value="")

    # If secrets exist, prefer them
    use_secrets = False
    if "FYERS_ACCESS_TOKEN" in st.secrets or "FYERS_CLIENT_ID" in st.secrets:
        use_secrets = st.checkbox("Use Streamlit Secrets for FYERS", value=True)
    st.markdown("---")
    st.caption("Built from your master scripts. Toggle tools to reduce load.")

# ---------- Helper utilities ----------
def show_status(success: bool, msg: str = "", details: str = None):
    if success:
        st.success(msg)
    else:
        st.error(msg)
        if details:
            with st.expander("Error details"):
                st.text(details)

def download_button_for_figure(fig_bytes: BytesIO, label: str, filename: str):
    b64 = base64.b64encode(fig_bytes.getvalue()).decode()
    href = f'<a href="data:file/png;base64,{b64}" download="{filename}">{label}</a>'
    st.markdown(href, unsafe_allow_html=True)

# Auto-refresh logic
if auto_refresh:
    st.experimental_set_query_params(_refresh=int(time.time()))
    time.sleep(1)
    st.experimental_rerun()

# ---------- Main layout: Tabs ----------
tabs = []
if show_ivr: tabs.append("IVR")
if show_expected: tabs.append("Expected Move")
if show_corr: tabs.append("Correlation")
if show_cone: tabs.append("Volatility Cone")
if show_fyers: tabs.append("FYERS OI Profile")
if show_vrp: tabs.append("VRP")
if show_hurst: tabs.append("Hurst")
if show_liquidity: tabs.append("Liquidity Sweep")
if show_parkinson: tabs.append("Parkinson")

if not tabs:
    st.warning("No tools selected. Use the sidebar to enable tools.")
    st.stop()

tab_objs = st.tabs(tabs)

# Map tab name to function and display logic
tab_map = {
    "IVR": ("NIFTY Implied Volatility (IVR & IVP)", plot_nifty_volatility_st),
    "Expected Move": ("NIFTY Expected Move", plot_expected_move_st),
    "Correlation": ("Inter-Index Correlation", plot_index_divergence_st),
    "Volatility Cone": ("Volatility Cone", plot_volatility_cone_st),
    "FYERS OI Profile": ("FYERS OI/Volume Profile", plot_fyers_oi_profile_st),
    "VRP": ("Volatility Risk Premium (VRP)", plot_vrp_st),
    "Hurst": ("Hurst Exponent (Market Regime)", plot_hurst_regime_st),
    "Liquidity Sweep": ("Intraday Liquidity Sweep", plot_liquidity_sweep_st),
    "Parkinson": ("Parkinson Estimator", plot_parkinson_estimator_st),
}

# Iterate tabs and render selected tools
for i, tab_name in enumerate(tabs):
    with tab_objs[i]:
        title, func = tab_map[tab_name]
        st.header(title)

        # Per-tool controls
        cols = st.columns([1, 1, 1, 2])
        with cols[0]:
            run_btn = st.button(f"Run {tab_name}", key=f"run_{tab_name}")
        with cols[1]:
            cached = st.checkbox("Use cache", value=True, key=f"cache_{tab_name}")
        with cols[2]:
            download_fig = st.checkbox("Show download", value=True, key=f"dl_{tab_name}")
        with cols[3]:
            st.write("Last run:")
            last_run_key = f"last_run_{tab_name}"
            last_run = st.session_state.get(last_run_key, "Never")
            st.write(last_run)

        # Decide FYERS credentials to pass if needed
        if tab_name == "FYERS OI Profile":
            if use_secrets:
                access_token = st.secrets.get("FYERS_ACCESS_TOKEN", "")
                client_id = st.secrets.get("FYERS_CLIENT_ID", "")
            else:
                access_token = fyers_token
                client_id = fyers_client_id
        else:
            access_token = None
            client_id = None

        # Run the function (either automatically or via button)
        should_run = run_btn or st.session_state.get(f"auto_run_{tab_name}", False)
        if should_run:
            try:
                # Capture stdout/plots inside the called function (they use st.pyplot)
                # If the function returns a BytesIO (refactored), we can show and offer download.
                result = None
                if tab_name == "FYERS OI Profile":
                    # FYERS function expects tokens and interactive inputs inside; call with tokens
                    func(access_token, client_id)
                    result = None
                else:
                    # Many of your functions directly call st.pyplot; call them and rely on their UI output
                    func()
                    result = None

                st.session_state[last_run_key] = time.strftime("%Y-%m-%d %H:%M:%S")
                show_status(True, f"{title} rendered.")
            except Exception as e:
                tb = traceback.format_exc()
                show_status(False, f"Failed to run {title}: {e}", details=tb)

        # Optionally show a small help / explanation
        with st.expander("About this tool"):
            st.markdown(f"- **Source:** your master scripts (refactored).")
            st.markdown("- **Tip:** Run only the tools you need to reduce API calls and speed up the app.")
            if tab_name == "FYERS OI Profile":
                st.markdown("- **FYERS:** set tokens in Streamlit Secrets for secure usage.")

# ---------- Footer ----------
st.markdown("---")
st.caption("Built from your master financial scripts. Data via yfinance. FYERS integration requires valid credentials.")
