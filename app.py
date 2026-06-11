# streamlit_app.py
# Robust Streamlit launcher for Nifty Live Tools
import streamlit as st
import traceback
from io import BytesIO
import base64
import time

st.set_page_config(page_title="Nifty Live Tools (Debuggable)", layout="wide")

st.title("Nifty Live Tools — Debuggable Launcher")

# Safe imports: show errors in UI instead of blank screen
IMPORT_ERROR = None
try:
    import nifty_tools as nt
    from fyers_client import fetch_oi_profile
    from scheduler import RepeatingTimer
except Exception as e:
    IMPORT_ERROR = traceback.format_exc()

if IMPORT_ERROR:
    st.error("Import error — app cannot continue. See details below.")
    with st.expander("Import traceback"):
        st.text(IMPORT_ERROR)
    st.stop()

# Session cache helpers
if "cache" not in st.session_state:
    st.session_state.cache = {}
if "scheduler" not in st.session_state:
    st.session_state.scheduler = None

def cache_set(key, val):
    st.session_state.cache[key] = val

def cache_get(key):
    return st.session_state.cache.get(key)

def cache_clear():
    st.session_state.cache = {}

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
    st.subheader("Cache & Scheduler")
    precompute_interval = st.number_input("Precompute interval (minutes)", min_value=1, max_value=1440, value=10, step=1)
    start_scheduler = st.button("Start background precompute")
    stop_scheduler = st.button("Stop background precompute")
    clear_cache = st.button("Clear cache now")

if clear_cache:
    cache_clear()
    st.success("Cache cleared")

# Background precompute function (safe, swallows exceptions)
def precompute_all():
    try:
        # heavy plots
        try:
            buf = nt.plot_volatility_cone_buf()
            cache_set("volatility_cone", buf.getvalue())
        except Exception:
            pass
        try:
            buf = nt.plot_vrp_buf()
            cache_set("vrp", buf.getvalue())
        except Exception:
            pass
        try:
            buf = nt.plot_hurst_buf()
            cache_set("hurst", buf.getvalue())
        except Exception:
            pass
    except Exception:
        pass

# Scheduler control
if start_scheduler:
    if st.session_state.scheduler is None:
        rt = RepeatingTimer(int(precompute_interval) * 60, precompute_all)
        rt.start()
        st.session_state.scheduler = rt
        st.success(f"Scheduler started every {precompute_interval} minutes")
    else:
        st.info("Scheduler already running")

if stop_scheduler:
    if st.session_state.scheduler:
        st.session_state.scheduler.stop()
        st.session_state.scheduler = None
        st.success("Scheduler stopped")
    else:
        st.info("No scheduler running")

# Helper to render buffer (from function or cache)
def render_from_func_or_cache(key, func, cache_ttl_seconds=300):
    cached = cache_get(key)
    if cached:
        buf = BytesIO(cached)
        st.image(buf)
        if st.button(f"Download {key} PNG"):
            b64 = base64.b64encode(cached).decode()
            href = f'<a href="data:file/png;base64,{b64}" download="{key}.png">Download {key}.png</a>'
            st.markdown(href, unsafe_allow_html=True)
        st.success(f"Displayed {key} from cache")
        return

    try:
        buf = func()
        cache_set(key, buf.getvalue())
        st.image(buf)
        if st.button(f"Download {key} PNG"):
            b64 = base64.b64encode(buf.getvalue()).decode()
            href = f'<a href="data:file/png;base64,{b64}" download="{key}.png">Download {key}.png</a>'
            st.markdown(href, unsafe_allow_html=True)
        st.success(f"Rendered {key} at {time.strftime('%Y-%m-%d %H:%M:%S')}")
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
            if tool == "IVR & IVP":
                render_from_func_or_cache("ivr_ivp", nt.plot_nifty_volatility_buf)
            elif tool == "Expected Move":
                render_from_func_or_cache("expected_move", nt.plot_expected_move_buf)
            elif tool == "Correlation":
                render_from_func_or_cache("correlation", nt.plot_index_divergence_buf)
            elif tool == "Volatility Cone":
                render_from_func_or_cache("volatility_cone", nt.plot_volatility_cone_buf)
            elif tool == "VRP":
                render_from_func_or_cache("vrp", nt.plot_vrp_buf)
            elif tool == "Hurst":
                render_from_func_or_cache("hurst", nt.plot_hurst_buf)
            elif tool == "Liquidity Sweep":
                render_from_func_or_cache("liquidity", nt.plot_liquidity_sweep_buf)
            elif tool == "Parkinson":
                render_from_func_or_cache("parkinson", nt.plot_parkinson_buf)
            elif tool == "FYERS OI Profile":
                st.subheader("FYERS OI / Volume Profile")
                underlying = st.selectbox("Underlying", ["NIFTY", "BANKNIFTY"], index=0)
                expiry = st.text_input("Expiry (e.g., 26JUN)", value="26JUN")
                spot = st.number_input("Spot price (approx)", value=23300.0, step=50.0)
                strike_step = st.selectbox("Strike step", [25, 50, 100, 200], index=1)
                fetch_btn = st.button("Fetch FYERS OI Profile")
                if fetch_btn:
                    if use_secrets:
                        access_token = st.secrets.get("FYERS_ACCESS_TOKEN", "")
                        client_id = st.secrets.get("FYERS_CLIENT_ID", "")
                    else:
                        access_token = fyers_token_input
                        client_id = fyers_client_id_input
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

st.markdown("---")
st.caption("If the screen is still blank after these changes, check the terminal logs and paste the traceback here and I will debug it line-by-line.")
