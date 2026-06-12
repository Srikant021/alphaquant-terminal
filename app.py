# streamlit_app.py
# Robust Streamlit launcher for Nifty Live Tools
import streamlit as st
import traceback
import time

st.set_page_config(page_title="Nifty Live Tools (Debuggable)", layout="wide")
st.title("Nifty Live Tools — Debuggable Launcher")

# Safe imports: show errors in UI instead of blank screen
IMPORT_ERROR = None
try:
    import nifty_tools as nt
    from fyers_client import fetch_oi_profile
except Exception as e:
    IMPORT_ERROR = traceback.format_exc()

if IMPORT_ERROR:
    st.error("Import error — app cannot continue. See details below.")
    with st.expander("Import traceback"):
        st.text(IMPORT_ERROR)
    st.stop()

# Sidebar controls
with st.sidebar:
    st.header("Controls")
    tools = st.multiselect(
        "Select tools to show", 
        [
            "IVR & IVP", "Expected Move", "Correlation", "Volatility Cone",
            "VRP", "Hurst", "Liquidity Sweep", "Parkinson", "FYERS OI Profile"
        ], 
        default=["IVR & IVP", "Expected Move"]
    )

    st.markdown("---")
    st.subheader("FYERS credentials")
    st.info("Prefer Streamlit Secrets for production.")
    fyers_token_input = st.text_input("FYERS Access Token (optional)", type="password")
    fyers_client_id_input = st.text_input("FYERS Client ID (optional)")
    
    use_secrets = False
    if "FYERS_ACCESS_TOKEN" in st.secrets or "FYERS_CLIENT_ID" in st.secrets:
        use_secrets = st.checkbox("Use Streamlit Secrets for FYERS", value=True)

    st.markdown("---")
    st.subheader("Cache Settings")
    cache_ttl = st.number_input("Cache TTL (minutes)", min_value=1, max_value=1440, value=10, step=1)
    
    if st.button("Clear Cache Now"):
        st.cache_data.clear()
        st.success("Cache cleared!")

# --- NATIVE STREAMLIT CACHING ---
@st.cache_data(ttl=cache_ttl * 60, show_spinner=False)
def get_plot_buffer(func_name):
    """Fetches and caches the image buffer from the respective nifty_tools function."""
    func_map = {
        "ivr_ivp": nt.plot_nifty_volatility_buf,
        "expected_move": nt.plot_expected_move_buf,
        "correlation": nt.plot_index_divergence_buf,
        "volatility_cone": nt.plot_volatility_cone_buf,
        "vrp": nt.plot_vrp_buf,
        "hurst": nt.plot_hurst_buf,
        "liquidity": nt.plot_liquidity_sweep_buf,
        "parkinson": nt.plot_parkinson_buf,
    }
    
    func = func_map.get(func_name)
    if func:
        # Return bytes so it can be hashed and cached by Streamlit easily
        return func().getvalue() 
    return None

def render_tool(key):
    """Helper to render the cached image and download button."""
    try:
        with st.spinner(f"Computing {key}..."):
            image_bytes = get_plot_buffer(key)
        
        if image_bytes:
            st.image(image_bytes)
            
            st.download_button(
                label=f"Download {key} PNG",
                data=image_bytes,
                file_name=f"{key}.png",
                mime="image/png"
            )
            st.caption(f"Last updated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    except Exception as e:
        st.error(f"Failed to render {key}: {e}")
        with st.expander("Traceback"):
            st.text(traceback.format_exc())

# Render selected tools
if not tools:
    st.warning("No tools selected. Use the sidebar to enable tools.")
else:
    tabs = st.tabs(tools)
    
    # Internal keys matching the func_map
    tool_mapping = {
        "IVR & IVP": "ivr_ivp",
        "Expected Move": "expected_