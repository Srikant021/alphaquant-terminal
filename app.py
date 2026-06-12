# streamlit_app.py
import streamlit as st
import traceback
from io import BytesIO
import time

st.set_page_config(page_title="Nifty Live Tools (Debuggable)", layout="wide")
st.title("Nifty Live Tools — Debuggable Launcher")

# Safe imports
IMPORT_ERROR = None
try:
    import nifty_tools as nt
    from fyers_client import fetch_oi_profile
    # We no longer need the RepeatingTimer scheduler!
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
    tools = st.multiselect("Select tools to show", [
        "IVR & IVP", "Expected Move", "Correlation", "Volatility Cone",
        "VRP", "Hurst", "Liquidity Sweep", "Parkinson", "FYERS OI Profile"
    ], default=["IVR & IVP", "Expected Move"])

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
    # We use this value to control the TTL of our cache
    cache_ttl = st.number_input("Cache TTL (minutes)", min_value=1, max_value=1440, value=10, step=1)
    
    if st.button("Clear Cache Now"):
        st.cache_data.clear()
        st.success("Cache cleared!")

# --- NATIVE STREAMLIT CACHING ---
# This decorator automatically caches the output. If the function is called again 
# within the TTL window (e.g., 10 mins), it instantly returns the cached BytesIO buffer.
@st.cache_data(ttl=cache_ttl * 60, show_spinner=False)
def get_plot_buffer(func_name):
    # Map the string name to the actual function to allow Streamlit to hash the arguments cleanly
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
        return func().getvalue() # Return bytes so it's easily cacheable
    return None

def render_tool(key):
    try:
        with st.spinner(f"Computing {key}..."):
            # This will fetch from cache if within TTL, or compute if expired/new
            image_bytes = get_plot_buffer(key)
        
        if image_bytes:
            st.image(image_bytes)
            
            # Native Streamlit Download Button
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
    for i, tool in enumerate(tools):
        with tabs[i]:
            st.header(tool)
            
            # Map UI selection to the internal keys
            tool_mapping = {
                "IVR & IVP": "ivr_ivp",
                "Expected Move": "expected_move",
                "Correlation": "correlation",
                "Volatility Cone": "volatility_cone",
                "VRP": "vrp",
                "Hurst": "hurst",
                "Liquidity Sweep": "liquidity",
                "Parkinson": "parkinson"
            }
            
            if tool in tool_mapping:
                render_tool(tool_mapping[tool])
                
            elif tool == "FYERS OI Profile":
                st.subheader("FYERS OI / Volume Profile")
                underlying = st.selectbox("Underlying", ["NIFTY", "BANKNIFTY"], index=0)
                expiry = st.text_input("Expiry (e.g., 26JUN)", value="26JUN")
                spot = st.number_input("Spot price (approx)", value=23300.0, step=50.0)
                strike_step = st.selectbox("Strike step", [25, 50, 100, 200], index=1)
                
                if st.button("Fetch FYERS OI Profile"):
                    access_token = st.secrets.get("FYERS_ACCESS_TOKEN", "") if use_secrets else fyers_token_input
                    client_id = st.secrets.get("FYERS_CLIENT_ID", "") if use_secrets else fyers_client_id_input
                    
                    if not access_token or not client_id:
                        st.error("FYERS credentials missing. Set them in Secrets or enter in sidebar.")
                    else:
                        try:
                            df, metric = fetch_oi_profile(access_token, client_id, underlying=underlying, expiry_str=expiry, spot_price=spot, strike_step=strike_step)
                            st.write(f"Metric used: **{metric}**")
                            st.dataframe(df)
                            
                            import matplotlib.pyplot as plt
                            fig, ax = plt.subplots(figsize=(8, 6))
                            ax.barh(df.index, df["Call_Data"], color="#FF3333", alpha=0.8, label=f"Call {metric}")
                            ax.barh(df.index, -df["Put_Data"], color="#00FF00", alpha=0.8, label=f"Put {metric}")
                            ax.axvline(0, color="white", linewidth=1)
                            ax.set_title(f"{underlying} {metric} Profile (Expiry: {expiry})")
                            ax.legend(facecolor="black")
                            st.pyplot(fig)
                        except Exception as e:
                            st.error(f"FYERS fetch failed: {e}")
                            with st.expander("Traceback"):
                                st.text(traceback.format_exc())
            else:
                st.info("Tool not implemented in this build.")