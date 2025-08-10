import streamlit as st
import pandas as pd
import os
import json
from xbbg import blp
from datetime import datetime
from st_aggrid import AgGrid, GridOptionsBuilder
import numpy as np
import streamlit.components.v1 as components
import os
import time

# --- Constants and Style ---

PRIMARY_COLOR = "#793690"
LIGHTER_PURPLE = "#B491C8"
ACCENT_BG = "#F3EFFA"
FONT_STACK = "'Inter', 'Segoe UI', Arial, sans-serif"
LOGO_URL = "https://www.marex.com/wp-content/themes/marex-2024/assets/img/logo.svg"
ICON_URL = "https://d3tb6862amv5ed.cloudfront.net/uploads/2024/01/favicon-32x32-1.png"

st.set_page_config(page_title="DV01 Calculator", page_icon=ICON_URL, layout="centered")
st.markdown(f"""
<div style='text-align:center; margin-bottom:0px;'>
    <img src='{LOGO_URL}' height='48' style='margin-bottom:1em'>
    <h2 style='font-size:2.23em; font-weight:500; color:{PRIMARY_COLOR}; letter-spacing: -0.02em;'>DV01 Calculator</h2>
    <div style='font-size:1.13em;color:#555;font-weight:500;letter-spacing: -0.03em;'>Powered by Ben Nathan with assistance from Bloomberg</div>
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<style>
/* --- Minimal CSS for Marex theme --- */

/* DV01 number input: basic border + purple on focus/hover */
.stNumberInput > div > input[type="number"] {{
    border: 1.5px solid #ccc !important;
    border-radius: 5px !important;
    font-family: {FONT_STACK} !important;
    padding: 0.4em 0.75em !important;
    font-size: 1.13em !important;
    outline: none !important;
    transition: border-color 0.3s, box-shadow 0.3s;
}}
.stNumberInput > div > input[type="number"]:focus,
.stNumberInput > div > input[type="number"]:hover {{
    border-color: {LIGHTER_PURPLE} !important;
    box-shadow: 0 0 0 2px {LIGHTER_PURPLE} !important;
}}

/* Calculate button with Marex theme */
.stButton>button {{
    background: {PRIMARY_COLOR} !important;
    color: #fff !important;
    font-family: {FONT_STACK} !important;
    font-size: 1.13em;
    font-weight: 500;
    padding: 0.5em 1.25em;
    border-radius: 5px;
    border: none !important;
    transition: background 0.2s;
}}
.stButton>button:hover {{
    background: {LIGHTER_PURPLE} !important;
    color: #fff !important;
}}

</style>
""", unsafe_allow_html=True)

# --- Futures Configuration ---

FUTURES_CONFIG = {
    "German 2 Year Schatz": {"root": "DU", "default_currency": "EUR Curncy", "contract_size": 100000, "generic_yield_ticker": "GDBR2 Index", "region": "EUR", "duration": 2},
    "Italian 2 Year BTP": {"root": "BTS", "default_currency": "EUR Curncy", "contract_size": 100000, "generic_yield_ticker": "GBTPGR2 Index", "region": "EUR", "duration": 2},
    "Canada 2 Year Bond": {"root": "CV", "default_currency": "CAD Curncy", "contract_size": 100000, "generic_yield_ticker": "GCAN2YR  Index", "region": "CAD", "duration": 2},
    "US 2 Year Treasury": {"root": "TU", "default_currency": "USD Curncy", "contract_size": 200000, "generic_yield_ticker": "USGG2YR Index", "region": "USD", "duration": 2},

    "Australian 3 Year Bond": {"root": "YM", "default_currency": "AUD Curncy", "contract_size": 100000, "generic_yield_ticker": "GACGB3 Index", "region": "AUD", "duration": 3},
    "US 3 Year Treasury": {"root": "3Y", "default_currency": "USD Curncy", "contract_size": 100000, "generic_yield_ticker": "USGG3YR Index", "region": "USD", "duration": 3},
    "German 5 Year Bobl": {"root": "OE", "default_currency": "EUR Curncy", "contract_size": 100000, "generic_yield_ticker": "GDBR5 Index", "region": "EUR", "duration": 5},
    "US 5 Year Treasury": {"root": "FV", "default_currency": "USD Curncy", "contract_size": 100000, "generic_yield_ticker": "USGG5YR Index", "region": "USD", "duration": 5},
    "German 10 Year Bund": {"root": "RX", "default_currency": "EUR Curncy", "contract_size": 100000, "generic_yield_ticker": "GDBR10 Index", "region": "EUR", "duration": 10},
    "French 10 Year OAT": {"root": "OAT", "default_currency": "EUR Curncy", "contract_size": 100000, "generic_yield_ticker": "GFRN10 Index", "region": "EUR", "duration": 10},
    "Italian 10 Year BTP": {"root": "IK", "default_currency": "EUR Curncy", "contract_size": 100000, "generic_yield_ticker": "GBTPGR10 Index", "region": "EUR", "duration": 10},
    "UK 10 Year Gilt": {"root": "G ", "default_currency": "GBP Curncy", "contract_size": 100000, "generic_yield_ticker": "GUKG10 Index", "region": "GBP", "duration": 10},
    "Australian 10 Year Bond": {"root": "XM", "default_currency": "AUD Curncy", "contract_size": 100000, "generic_yield_ticker": "GACGB10 Index", "region": "AUD", "duration": 10},
    "Japanese 10 Year JGB": {"root": "JB", "default_currency": "JPY Curncy", "contract_size": 100000000, "generic_yield_ticker": "GJGB10 Index", "region": "JPY", "duration": 10},
    "Canada 10 Year Bond": {"root": "CN", "default_currency": "CAD Curncy", "contract_size": 100000, "generic_yield_ticker": "GCAN10YR  Index", "region": "CAD", "duration": 10},
    "US 10 Year Treasury": {"root": "TY", "default_currency": "USD Curncy", "contract_size": 100000, "generic_yield_ticker": "USGG7YR Index", "region": "USD", "duration": 10},
    "US Ultra 10 Year Treasury": {"root": "UXY", "default_currency": "USD Curncy", "contract_size": 100000, "generic_yield_ticker": "USGG10YR Index", "region": "USD", "duration": 10},
    "US 20 Year Bond": {"root": "US", "default_currency": "USD Curncy", "contract_size": 100000, "generic_yield_ticker": "USGG20YR Index", "region": "USD", "duration": 20},
    "German 30 Year Buxl": {"root": "UB", "default_currency": "EUR Curncy", "contract_size": 100000, "generic_yield_ticker": "GDBR30 Index", "region": "EUR", "duration": 30},
    "US 30 Year Ultra Bond": {"root": "WN", "default_currency": "USD Curncy", "contract_size": 100000, "generic_yield_ticker": "USGG30YR Index", "region": "USD", "duration": 30},
}

DURATIONS = {
    "US 30 Year Ultra Bond": 30, "US 20 Year Bond": 20, "US 10 Year Treasury": 10, "US Ultra 10 Year Treasury": 10,
    "US 5 Year Treasury": 5, "US 3 Year Treasury": 3, "US 2 Year Treasury": 2,
    "Canada 10 Year Bond": 10, "Canada 2 Year Bond": 2,
    "German 30 Year Buxl": 30, "German 10 Year Bund": 10, "German 5 Year Bobl": 5, "German 2 Year Schatz": 2,
    "French 10 Year OAT": 10, "Italian 10 Year BTP": 10, "Italian 2 Year BTP": 2,
    "UK 10 Year Gilt": 10, "Australian 10 Year Bond": 10, "Australian 3 Year Bond": 3, "Japanese 10 Year JGB": 10
}

# ... Other constants and utils ...

CURRENCY_SYMBOLS = {"USD": "$", "EUR": "€", "GBP": "£", "JPY": "¥", "AUD": "A$", "CAD": "C$"}

MONTH_CODES = {3: 'H', 6: 'M', 9: 'U', 12: 'Z'}
CODE_TO_MONTH = {'H': 'Mar', 'M': 'Jun', 'U': 'Sep', 'Z': 'Dec'}
IMM_MONTHS = [3, 6, 9, 12]
STATIC_CACHE_DIR = ".dv01cache"
CHAIN_CACHE_PATH = os.path.join(STATIC_CACHE_DIR, "chain_cache.json")
OI_CACHE_PATH = os.path.join(STATIC_CACHE_DIR, "oichain_cache.json")
os.makedirs(STATIC_CACHE_DIR, exist_ok=True)

def get_futures_chain_from_bbg(root_ticker):
    df = blp.bds([root_ticker], flds=["FUT_CHAIN"])
    if df.empty or df.shape[1] < 1:
        return []
    col = df.columns[0] if df.shape[1] == 1 else df.columns[1]
    listed = df.reset_index()[col].dropna().tolist()
    return [str(x) for x in listed if isinstance(x, str) and "Comdty" in x]

def load_chain_cache(FUTURES_CONFIG):
    today = datetime.today().strftime("%Y-%m-%d")
    if os.path.exists(CHAIN_CACHE_PATH):
        with open(CHAIN_CACHE_PATH, "r") as f:
            cache = json.load(f)
        if cache.get("date") == today:
            return cache["chains"]

    chains = {}
    for prod, cfg in FUTURES_CONFIG.items():
        root = cfg["root"]
        generic = f"{root}1 Comdty"
        generic_yield = cfg["generic_yield_ticker"]
        FX = cfg["default_currency"]

        df = blp.bds([generic], flds=["FUT_CHAIN"])
        if df.empty or df.shape[1] < 1:
            futures_chain = []
        else:
            col = df.columns[0] if df.shape[1] == 1 else df.columns[1]
            listed = df.reset_index()[col].dropna().tolist()
            futures_chain = [str(x) for x in listed if isinstance(x, str) and "Comdty" in x]

        # Include generic yield ticker and FX ticker in the cached list
        combined_chain = futures_chain.copy()
        if generic_yield not in combined_chain:
            combined_chain.append(generic_yield)
        if FX not in combined_chain:
            combined_chain.append(FX)

        chains[root] = combined_chain

    cache = {"date": today, "chains": chains}
    with open(CHAIN_CACHE_PATH, "w") as f:
        json.dump(cache, f)

    return chains

def dv01_per_lot_from_expiry(expiry_ticker, target_ccy_code, future_name):
    """
    Lookup DV01 per lot for a given expiry and convert to target currency if needed.
    """
    for data in st.session_state.get("oi_vol_cache", {}).values():
        for ex in data.get("expiries", []):
            if ex.get("ticker") == expiry_ticker:
                dv01 = ex.get("dv01_per_lot", None)
                expiry_ccy = ex.get("currency", target_ccy_code)

                if dv01 is None:
                    # fallback calculation
                    try:
                        frsk = ex.get("frsk", 0)
                        root = FUTURES_CONFIG[future_name]["root"]
                        multiplier = 20 if root in ["TU", "3Y"] else 10000 if root == "JB" else 10
                        dv01 = frsk * multiplier
                    except:
                        dv01 = 0

                if expiry_ccy != target_ccy_code:
                    fx = get_exchange_rate(expiry_ccy, target_ccy_code)
                    dv01 *= fx
                return dv01
    st.warning(f"Couldn't find DV01 for {expiry_ticker}")
    return 0.0

# Ensure the flag exists
if "just_calculated" not in st.session_state:
    st.session_state.just_calculated = False


def load_oi_volume_cache(FUTURES_CONFIG, product_chains, max_cache_age_seconds=60):
    today = datetime.today().strftime("%Y-%m-%d")
    OI_CACHE_PATH = os.path.join(".dv01cache", "oichain_cache.json")

    # Check if cache file exists and is recent enough (you can tune max_cache_age_seconds)
    if os.path.exists(OI_CACHE_PATH):
        # Check file modification time
        file_mod_time = os.path.getmtime(OI_CACHE_PATH)
        age_seconds = time.time() - file_mod_time
        if age_seconds <= max_cache_age_seconds:
            # File is fresh enough to trust
            with open(OI_CACHE_PATH, "r") as f:
                cache = json.load(f)
            if cache.get("date") == today:  # Still good to check date matches today
                return cache["data"]

    all_tickers = []
    for lst in product_chains.values():
        all_tickers.extend(lst)
    default_currencies = list({cfg['default_currency'] for cfg in FUTURES_CONFIG.values()})
    all_tickers.extend(default_currencies)
    all_tickers = list(dict.fromkeys(all_tickers))

    if not all_tickers:
        return {}

    # Map tickers to AUD-or-not for special logic
    # Map tickers to risk field and yield field based on region
    risk_field_map = {}
    yield_field_map = {}  # To track which yield field to request per ticker
    
    for prod, cfg in FUTURES_CONFIG.items():
        root = cfg["root"]
        region = cfg.get("region", "")
        for ticker in product_chains.get(root, []):
            # Risk field is always 'fut_cnv_risk_frsk' (based on your snippet) - adjust if needed
            risk_field_map[ticker] = "fut_cnv_risk_frsk"
    
            # Yield field depends on region:
            # AUD tickers get 'YLD_YTM_MID' as yield; else 'CTD_FORWARD_YTM_LAST'
            yield_field_map[ticker] = "YLD_YTM_MID" if region == "AUD" else "CTD_FORWARD_YTM_LAST"
    
    # Base fields always requested regardless of ticker
    base_fields = [
        "PX_LAST", "PX_VOLUME", "CURRENT_CONTRACT_MONTH_YR",
        "FUT_NOTICE_FIRST", "OPEN_INT",
    ]
    
    # Separate tickers by their yield field, so we know which fields to request
    tickers_yld_yield = [t for t, yf in yield_field_map.items() if yf == "YLD_YTM_MID"]
    tickers_ctd_yield = [t for t, yf in yield_field_map.items() if yf == "CTD_FORWARD_YTM_LAST"]
    
    frames = []
    try:
        from xbbg import blp
    
        # Fetch data for tickers that use CTD_FORWARD_YTM_LAST yield field
        if tickers_ctd_yield:
            fields_ctd = base_fields + ["CTD_FORWARD_YTM_LAST", "fut_cnv_risk_frsk"]
            df_ctd = blp.bdp(tickers=tickers_ctd_yield, flds=fields_ctd)
            df_ctd.index = df_ctd.index.astype(str)
            frames.append(df_ctd)
    
        # Fetch data for tickers that use YLD_YTM_MID (AUD tickers)
        if tickers_yld_yield:
            fields_yld = base_fields + ["YLD_YTM_MID", "fut_cnv_risk_frsk"]
            df_yld = blp.bdp(tickers=tickers_yld_yield, flds=fields_yld)
            df_yld.index = df_yld.index.astype(str)
    
            # Rename columns to unify fields downstream:
            # YLD_YTM_MID Ã¢â€ â€™ CTD_FORWARD_YTM_LAST (so yields align)
            if "yld_ytm_mid" in df_yld.columns:
                df_yld = df_yld.rename(columns={"yld_ytm_mid": "ctd_forward_ytm_last"})
    
            frames.append(df_yld)
    
        if frames:
            oivoldf = pd.concat(frames)
        else:
            oivoldf = pd.DataFrame([])
    
    except Exception as e:
        print(f"Bloomberg data retrieval failed: {e}")
        oivoldf = pd.DataFrame([], columns=base_fields + ["CTD_FORWARD_YTM_LAST", "fut_cnv_risk_frsk"])
    
    # Rename risk field column from 'fut_cnv_risk_frsk' 'fut_cnv_risk_ctd' for consistency downstream
    if "fut_cnv_risk_frsk" in oivoldf.columns:
        oivoldf = oivoldf.rename(columns={"fut_cnv_risk_frsk": "fut_cnv_risk_ctd"})

    # Normalize FX rates
    bbg_fx_rates = {}
    for ccy in ["EUR", "GBP", "JPY", "AUD", "CAD"]:
        ticker = f"{ccy} Curncy"
        if ticker in oivoldf.index:
            bbg_fx_rates[ccy] = oivoldf.at[ticker, 'px_last']
        else:
            bbg_fx_rates[ccy] = float('nan')

    def normalize_bbg_fx_rates(bbg_rates):
        normalized = {}
        for ccy, val in bbg_rates.items():
            if ccy in ['EUR', 'GBP', 'AUD']:
                normalized[ccy] = 1 / val if val else float('nan')
            elif ccy in ['JPY', 'CAD']:
                normalized[ccy] = val
            else:
                normalized[ccy] = val
        return normalized

    normalized_fx_rates = normalize_bbg_fx_rates(bbg_fx_rates)

    # Overwrite PX_LAST for FX tickers with normalized values
    for ccy, normalized_val in normalized_fx_rates.items():
        ticker = f"{ccy} Curncy"
        if ticker in oivoldf.index:
            oivoldf.at[ticker, 'px_last'] = normalized_val

    root_expiry_stats = {}
    now = datetime.now()

    for prod, cfg in FUTURES_CONFIG.items():
        root_no_space = cfg["root"].strip()
        explist = product_chains.get(cfg["root"], [])

        expstats = []
        oi_total = 0.0
        for tick in explist:
            if tick in oivoldf.index:
                row = oivoldf.loc[tick]
                oi = row.get("open_int", 0.0)
                vol = row.get("px_volume", 0.0)
                price = row.get("px_last", 0.0)
                frsk = row.get("fut_cnv_risk_ctd", 0.0)
                oi = float(oi) if pd.notna(oi) else 0.0
                vol = float(vol) if pd.notna(vol) else 0.0
                forward_yield = row.get("ctd_forward_ytm_last", 0.0)
            else:
                oi, vol, price, frsk, forward_yield = 0.0, 0.0, 0.0, 0.0, 0.0

            parts = tick.split()
            if root_no_space == "G" and len(parts) >= 2:
                code_year_str = parts[1]
                code = code_year_str[0] if len(code_year_str) >= 1 else ''
                year_str = code_year_str[1:] if len(code_year_str) > 1 else '0'
            else:
                main_ticker = parts[0]
                if main_ticker.startswith(root_no_space):
                    code_year_str = main_ticker[len(root_no_space):]
                else:
                    code_year_str = ""
                code = code_year_str[0] if len(code_year_str) >= 1 else ''
                year_str = code_year_str[1:] if len(code_year_str) > 1 else '0'

            month = {'H': 'Mar', 'M': 'Jun', 'U': 'Sep', 'Z': 'Dec'}.get(code, code)
            try:
                year_num = int(year_str)
            except Exception:
                year_num = 0

            curr_decade = now.year // 10 * 10
            label_year = curr_decade + year_num
            if label_year < now.year:
                label_year += 10
            year_fmt = f"{label_year % 100:02d}"

            expstats.append({
                "ticker": tick,
                "month": month,
                "year": year_fmt,
                "last_price": price,
                "oi": oi,
                "vol": vol,
                "frsk": frsk,
                "forward_yield":forward_yield
            })
            oi_total += oi

        for ex in expstats:
            ex["oi_pct"] = f"{(ex['oi'] / oi_total) * 100:.1f}%" if oi_total else "0.0%"

        root_expiry_stats[cfg["root"]] = {
            "expiries": expstats,
            "default_currency": cfg['default_currency'],
            "generic_yield_ticker": cfg['generic_yield_ticker'],
        }

    with open(OI_CACHE_PATH, "w") as f:
        json.dump({"date": today, "data": root_expiry_stats}, f)

    return root_expiry_stats


def get_ccy_code(ticker):
    # Extracts 3-letter currency code, e.g. 'EUR Curncy' -> 'EUR'
    return ticker.split()[0].upper()

def custom_fx_cross_rate(native_ccy_code, target_ccy_code, fx_usd_rates):
    # If both currencies are the same, return 1
    if native_ccy_code == target_ccy_code:
        return 1.0

    # FX rates are assumed to be normalized to USD, i.e. value = Currency per 1 USD
    # For example, fx_usd_rates['EUR'] = 0.92 means 1 USD = 0.92 EUR

    # Handle target currency USD separately
    if target_ccy_code == "USD":
        # Cross rate = 1 / FX rate of native currency (Currency per USD inverted to get USD per Currency)
        native_fx = fx_usd_rates.get(native_ccy_code)
        if native_fx is None or native_fx == 0:
            return float('nan')
        return 1 / native_fx

    # For other target currencies (e.g., CAD, EUR, GBP, etc.)
    native_fx = fx_usd_rates.get(native_ccy_code)
    target_fx = fx_usd_rates.get(target_ccy_code)
    if native_fx is None or target_fx is None or native_fx == 0 or target_fx == 0:
        return float('nan')

    # Cross rate = (1 / native_fx) * target_fx == (target_currency per USD) / (native_currency per USD)
    return target_fx / native_fx


def convert_df_to_tsv(df):
    return df.to_csv(index=False, sep='\t').encode('utf-8')

def copy_table_to_clipboard_with_feedback(df):
    html_table = df.to_html(index=False, border=1)

    button_style = """
        background-color: #793690;
        color: white;
        border: none;
        border-radius: 5px;
        font-family: 'Inter', 'Segoe UI', Arial, sans-serif;
        font-size: 1.13em;
        padding: 0.5em 1.25em;
        font-weight: 400 !important;  /* Add !important to override defaults */    
        cursor: pointer;
        transition: background-color 0.3s ease;
    """
    button_hover_style = "this.style.backgroundColor = '#B491C8';"
    button_unhover_style = "this.style.backgroundColor = '#793690';"

    help_text = "If you press this button the whole table will be formatted for Bloomberg Chats"

    copy_js = f"""
    <div style="text-align:center; margin-bottom: 1em;">
        <button 
            id="copyBtn"
            style="{button_style}"
            title="{help_text}"
            onmouseover="{button_hover_style}"
            onmouseout="{button_unhover_style}"
            onclick="copyTable()"
        >
            <span style="font-weight:400;">Copy Table to Clipboard</span>
        </button>

    </div>
    <script>
    function copyTable() {{
        var table = document.getElementById('dv01table');
        var btn = document.getElementById('copyBtn');
        if (!table) {{
            alert('Table not found');
            return;
        }}
        var range = document.createRange();
        range.selectNode(table);

        var selection = window.getSelection();
        selection.removeAllRanges();
        selection.addRange(range);

        try {{
            var result = document.execCommand('copy');
            if(result) {{
                btn.innerHTML = 'Copied! Bring back the Liegestuhl ';
                setTimeout(function(){{ btn.innerHTML = 'Copy Table for Bloomberg'; }}, 2000);
            }} else {{
                alert('Copy command was unsuccessful. Please try copying manually.');
            }}
        }} catch(err) {{
            alert('Failed to copy: ' + err);
        }}
        selection.removeAllRanges();
    }}
    </script>
    <div id="dv01table" style="
        position: absolute;
        left: -9999px;
        top: 0;
        ">
        {html_table}
    </div>
    """
    
    components.html(copy_js, height=80)

# --- STYLE PATCH ---
st.markdown(f"""
    <style>
    .stButton > button.refresh {{
        background-color: white !important;
        color: {PRIMARY_COLOR} !important;
        border: 2px solid {PRIMARY_COLOR} !important;
        border-radius: 5px !important;
        font-weight: 500 !important;
        height: 3rem;
        width: 100% !important;
    }}
    .stButton > button.reset {{
        background-color: {ACCENT_BG} !important;
        color: white !important;
        border-radius: 5px !important;
        font-weight: 500 !important;
        height: 3rem;
        width: 100% !important;
    }}
    </style>
""", unsafe_allow_html=True)



def show_cross_product_dv01_matrix(FUTURES_CONFIG, oi_vol_cache, risk_qualifier, input_val, selected_ccy, selectedFutureName, target_dv01_from_first_leg=None):
    rows = []

    # FX normalization
    ccy_last_prices = {}
    for data in oi_vol_cache.values():
        def_ccy = data.get('default_currency')
        if def_ccy:
            found_price = None
            for ex in data.get("expiries", []):
                if ex.get("ticker") == def_ccy:
                    found_price = ex.get("last_price", 1.0)
                    break
            if found_price is None:
                found_price = 1.0
            ccy_last_prices[def_ccy] = found_price

    fx_usd_rates = {get_ccy_code(k): v for k, v in ccy_last_prices.items()}

    # Reference yield
    selected_yield = None
    selected_yield_ticker = FUTURES_CONFIG[selectedFutureName]["generic_yield_ticker"]
    for data in oi_vol_cache.values():
        for ex in data.get("expiries", []):
            if ex.get("ticker") == selected_yield_ticker:
                selected_yield = ex.get("last_price", 0.0)
                break
        if selected_yield is not None:
            break
    if selected_yield is None:
        selected_yield = 0.0

    for product, cfg in FUTURES_CONFIG.items():
        root = cfg['root']
        expiries = oi_vol_cache.get(root, {}).get("expiries", [])
        if not expiries:
            continue

        valid_expiries = [ex for ex in expiries if ex.get('oi', 0) > 0 and not pd.isna(ex.get('frsk'))]
        if not valid_expiries:
            valid_expiries = [ex for ex in expiries if not pd.isna(ex.get('frsk'))]
        if not valid_expiries:
            continue

        front_expiry = max(valid_expiries, key=lambda x: x.get('oi', 0))

        futures_price = front_expiry.get('last_price', 0.0)
        frsk = front_expiry.get('frsk', 0.0)
        futures_ticker = front_expiry.get('ticker', '')
        yield_ticker = cfg.get('generic_yield_ticker', '')
        forward_yield = front_expiry.get('forward_yield', '0.0')

        yield_price = 0.0
        for data in oi_vol_cache.values():
            for ex in data.get("expiries", []):
                if ex.get("ticker") == yield_ticker:
                    yield_price = ex.get("last_price", 0.0)
                    break
            if yield_price > 0:
                break

        native_ccy_code = get_ccy_code(cfg.get('default_currency'))
        target_ccy_code = selected_ccy.upper()
        fx_cross = custom_fx_cross_rate(native_ccy_code, target_ccy_code, fx_usd_rates)

        multiplier = 20 if root in ["TU", "3Y"] else 10000 if root == "JB" else 10
        dv01_per_lot = frsk * multiplier * fx_cross

        if product == selectedFutureName:
            # For first leg: compute DV01 based on input lots
            if risk_qualifier == "Lots":
                lots = input_val
                dv01_target = lots * frsk * fx_cross * multiplier
            else:
                dv01_target = input_val * 1000
                lots = dv01_target / dv01_per_lot if dv01_per_lot else float("nan")
        else:
            if risk_qualifier == "Lots" and target_dv01_from_first_leg is not None:
                # Use DV01 target from first leg to back out lots
                dv01_target = target_dv01_from_first_leg
                lots = dv01_target / dv01_per_lot if dv01_per_lot else float("nan")
            else:
                dv01_target = input_val * 1000
                lots = dv01_target / dv01_per_lot if dv01_per_lot else float("nan")

        rows.append({
            "Product": product,
            "Generic Yield": round(yield_price, 4),
            "Forward Yield": round(forward_yield, 4),
            "Futures": round(futures_price, 2),
            "Lots": round(lots, 2) if not pd.isna(lots) else "N/A",
        })

    if not rows:
        st.warning("No valid DV01 data to display in matrix.")
        return None

    df_matrix = pd.DataFrame(rows)

    # Cross Market Logic
    df_matrix["Cross Market"] = df_matrix["Forward Yield"].apply(lambda y: round(selected_yield - y, 4))
    df_matrix['Generic Yield'] = df_matrix['Generic Yield'].map(lambda x: f"{x:.4f}%")
    df_matrix['Forward Yield'] = df_matrix['Forward Yield'].map(lambda x: f"{x:.4f}%")
    df_matrix['Cross Market'] = df_matrix['Cross Market'].map(lambda x: f"{x:.4f}%")

    return df_matrix


product_chains = load_chain_cache(FUTURES_CONFIG)
oi_vol_cache = load_oi_volume_cache(FUTURES_CONFIG, product_chains)

# Build FX rates from oi_vol_cache
fx_usd_rates = {}
for data in oi_vol_cache.values():
    ccy = data.get('default_currency', '')
    if not ccy:
        continue
    native = ccy.split()[0]
    for ex in data.get('expiries', []):
        if ex.get('ticker') == ccy:
            fx_usd_rates[native] = ex.get('last_price', 1.0)
            break


if "risk_qualifier" not in st.session_state:
    st.session_state.risk_qualifier = "DV01 (000's)"
if "currency" not in st.session_state:
    st.session_state.currency = "USD"
if "last_future" not in st.session_state:
    st.session_state.last_future = None

risk_qualifiers = ["DV01 (000's)", "Lots"]
currencies = ["USD", "EUR", "GBP", "JPY", "AUD", "CAD"]

col1, col2, col3 = st.columns([1, 1, 1])

default_risk = st.session_state.get("risk_qualifier", "DV01 (000's)")
if default_risk not in risk_qualifiers:
    default_risk = "DV01 (000's)"

with col2:
    st.markdown(f"""
        <div style="
            font-family: {FONT_STACK};
            font-weight: 400;
            font-size: 1.35em;
            color: {PRIMARY_COLOR};
            margin-bottom: 0.em;
            letter-spacing: -0.02em;
            text-align: left;
        ">
            Risk Qualifier
        </div>
    """, unsafe_allow_html=True)
    st.session_state.risk_qualifier = st.selectbox(
        "",
        risk_qualifiers,
        index=risk_qualifiers.index(default_risk),
        key="rqdd",
        label_visibility="collapsed"
    )

dv01_label = "DV01 (000's)" if st.session_state.risk_qualifier == "DV01 (000's)" else "Lots"

with col1:
    st.markdown(f"""
        <div style="
            font-family: {FONT_STACK};
            font-weight: 400;
            font-size: 1.35em;
            color: {PRIMARY_COLOR};
            margin-bottom: 0.0em;
            letter-spacing: -0.02em;
            text-align: left;
        ">
            {dv01_label}

        </div>

    """, unsafe_allow_html=True)
    input_val = st.number_input(
        "",
        min_value=0,
        value=10,
        step=1,
        key="dv01txt",
        label_visibility="collapsed",
        format="%d"
    )

with col3:
    st.markdown(f"""
        <div style="
            font-family: {FONT_STACK};
            font-weight: 400;
            font-size: 1.35em;
            color: {PRIMARY_COLOR};
            margin-bottom: 0.0em;
            letter-spacing: -0.02em;
            text-align: left;
        ">
            Currency
        </div>
    """, unsafe_allow_html=True)
    curr_default = st.session_state.get("currency", "USD")
    if curr_default not in currencies:
        curr_default = "USD"
    st.session_state.currency = st.selectbox(
        "",
        currencies,
        index=currencies.index(curr_default),
        key="ccydd",
        label_visibility="collapsed"
    )

risk_qualifier = st.session_state.risk_qualifier
ccy = st.session_state.currency

future_options = [f"{k} ({v['root']} Comdty)" for k,v in FUTURES_CONFIG.items()]
future_keymap = list(FUTURES_CONFIG.keys())
default_idx = future_keymap.index(st.session_state.last_future) if st.session_state.last_future in future_keymap else 0


# --- Setup columns for main future selectors and add leg button ---
col_future, col_expiry, col_add_leg = st.columns([1, 1, 0.32])  # Adjust width for button

# --- Main Future Selector ---
with col_future:
    st.markdown(f"""
        <div style="font-family: {FONT_STACK}; font-weight: 400;
                    font-size: 1.35em; color: {PRIMARY_COLOR};
                    margin-bottom: 0.15em; letter-spacing: -0.02em;
                    text-align: left;">
            Future
        </div>
    """, unsafe_allow_html=True)

    future_disp = st.selectbox(
        "",
        future_options,
        index=default_idx,
        label_visibility="collapsed",
        key="main_future_select"
    )


future_name = future_keymap[future_options.index(future_disp)]

# --- Add Leg Button & Future 2 + Expiry 2 Selectors ---

# Add small CSS for toggle buttons
st.markdown(f"""
<style>
div.stButton > button#addLegBtnToggle {{
    font-size: 0.92rem;
    padding: 0.16em 0.75em;
    min-width: 2.5em;
    border-radius: 5px;
    background-color: #793690;
    color: #fff;
    border: none;
    font-weight: 500;
    letter-spacing: -0.01em;
    box-shadow: none;
    cursor: pointer;
}}
div.stButton > button#addLegBtnToggle:hover {{
    background-color: #B491C8;
}}
</style>
""", unsafe_allow_html=True)

# Initialize session state toggle if missing
if "show_future_2" not in st.session_state:
    st.session_state.show_future_2 = False

# Show "Add Leg" button if second leg not shown
with col_add_leg:
    if not st.session_state.show_future_2:
        st.markdown(f"<div style='padding-top: 0.8em;'>&nbsp;</div>",unsafe_allow_html=True)  # Padding to align button with dropdowns
        if st.button("Add Leg", key="addLegBtnToggle", help="Add Cross Market Leg"):
            st.session_state.show_future_2 = True

# Show Future 2 and Expiry 2 selectors + "Remove" button only if toggle ON
if st.session_state.show_future_2:
    col_future_2, col_expiry_2, col_remove = st.columns([1, 1, 0.32])

    # Initialize related session state variables
    if "last_future_2" not in st.session_state:
        st.session_state.last_future_2 = future_keymap[0]
    if "expiry_2_idx" not in st.session_state:
        st.session_state.expiry_2_idx = 0

    with col_future_2:
        st.markdown(f"""
            <div style="
                font-family: {FONT_STACK};
                font-weight: 400;
                font-size: 1.35em;
                color: {PRIMARY_COLOR};
                margin-bottom: 0.15em;
                letter-spacing: -0.02em;
                text-align: left;
            ">
                Second Leg
            </div>
        """, unsafe_allow_html=True)

        future_2_disp = st.selectbox(
            "",
            future_options,
            index=future_keymap.index(st.session_state.last_future_2),
            key="future2_select",
            label_visibility="collapsed"
        )
        st.session_state.last_future_2 = future_keymap[future_options.index(future_2_disp)]

    # Load expiry details for Future 2
    root_2 = FUTURES_CONFIG[st.session_state.last_future_2]["root"]
    expiry_stats_2 = oi_vol_cache.get(root_2, {}).get("expiries", [])
    expiry_stats_2 = sorted(expiry_stats_2, key=lambda x: -x.get("oi", 0))

    expiry_details_2 = []
    for ex in expiry_stats_2[:3]:
        expiry_details_2.append({
            'Label': f"{ex['month']} {ex['year']}",
            'Ticker': ex['ticker'],
            'FND': "",
            'OI': int(ex.get('oi', 0)),
            'last_price': float(ex.get('last_price', 0)),
            'Vol': int(ex.get('vol', 0)),
            'OI%': ex.get('oi_pct', '0.0%')
        })

    expiry_labels_2 = [
        f"{d['Label']} (OI={int(d['OI']/1000)}k, Vol={int(d['Vol']/1000)}k, {d['OI%']})"
        for d in expiry_details_2
    ]

    with col_expiry_2:
        st.markdown(f"""
            <div style="
                font-family: {FONT_STACK};
                font-weight: 400;
                font-size: 1.35em;
                color: {PRIMARY_COLOR};
                margin-bottom: 0.15em;
                letter-spacing: -0.02em;
                text-align: left;
            ">
                Expiry
            </div>
        """, unsafe_allow_html=True)

        if expiry_labels_2:
            selected_expiry_2_idx = st.selectbox(
                "",
                range(len(expiry_labels_2)),
                format_func=lambda i: expiry_labels_2[i],
                index=st.session_state.expiry_2_idx,
                key="expiry2_select",
                label_visibility="collapsed"
            )
            st.session_state.expiry_2_idx = selected_expiry_2_idx
        else:
            st.warning("No available expiries for this future at this time.")

    # Remove button - toggles off the second leg UI
    with col_remove:
        st.markdown(f"<div style='padding-top: 0.8em;'>&nbsp;</div>", unsafe_allow_html=True)

        if st.button("×", key="removeLegBtn", help="Remove Cross Market Leg"):
            st.session_state.show_future_2 = False


# Keep currency update logic, e.g. if main future changed, adjust session state currency
if st.session_state.last_future != future_name:
    st.session_state.currency = FUTURES_CONFIG[future_name]["default_currency"]
    st.session_state.last_future = future_name

root = FUTURES_CONFIG[future_name]["root"]
expiry_stats = oi_vol_cache.get(root, {}).get("expiries", [])
expiry_stats = sorted(expiry_stats, key=lambda x: -x.get("oi", 0))

expiry_details = []
for ex in expiry_stats[:3]:
    expiry_details.append({
        'Label': f"{ex['month']} {ex['year']}",
        'Ticker': ex['ticker'],
        'FND': "",
        'OI': int(ex.get('oi', 0)),
        'last_price': float(ex.get('last_price', 0)),
        'Vol': int(ex.get('vol', 0)),
        'OI%': ex.get('oi_pct', '0.0%'),
        'forward_yield': ex.get('forward_yield', '0.0%')
    })
if not expiry_details:
    st.warning("No available expiries for this future at this time. Please try another future or try again later.")
    st.stop()

expiry_labels = [
    f"{d['Label']} (OI={int(d['OI']/1000)}k, Vol={int(d['Vol']/1000)}k, {d['OI%']})"
    for d in expiry_details
]

default_expiry_idx = 0

with col_expiry:
    st.markdown(f"""
        <div style="font-family: {FONT_STACK}; font-weight: 400;
                    font-size: 1.35em; color: {PRIMARY_COLOR};
                    margin-bottom: 0.15em; letter-spacing: -0.02em;
                    text-align: left;">
            Expiry
        </div>
    """, unsafe_allow_html=True)

    selected_expiry_idx = st.selectbox(
        "",
        range(len(expiry_labels)),
        format_func=lambda i: expiry_labels[i],
        index=default_expiry_idx,
        key="expdd",
        label_visibility="collapsed"
    )

expiry_ticker = expiry_details[selected_expiry_idx]['Ticker']

all_expiry_tickers = [d['ticker'] for d in expiry_stats]
compare_tickers = []
for p in FUTURES_CONFIG:
    lst = product_chains.get(p, [])
    if lst:
        compare_tickers.append(lst[0])

all_bbg_tickers = list(set(all_expiry_tickers + compare_tickers))

def get_batched_bbg_df(all_tickers):
    fields = ["PX_LAST", "PX_VOLUME", "CURRENT_CONTRACT_MONTH_YR", "FUT_NOTICE_FIRST", "OPEN_INT", "fut_cnv_risk_frsk","CTD_FORWARD_YTM_LAST"]
    try:
        res = blp.bdp(tickers=all_tickers, flds=fields)
        res.index = res.index.astype(str)
    except Exception as e:
        st.error(f"Error querying Bloomberg: {e}")
        res = pd.DataFrame([], index=all_tickers, columns=fields)

    return res

all_bbg_df = get_batched_bbg_df(all_bbg_tickers)

if not st.session_state.get("calculated"):
    col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])
    with col3:
        calc_btn = st.button("Calculate", type="primary")
    if calc_btn:
        st.session_state.calculated = True
        st.session_state.refresh_trigger = True
        st.rerun()
else:
    # 2 centered columns + side spacers
    spacer_l, col1, col2, spacer_r = st.columns([1, 4, 4, 1])

    with col1:
        refresh_clicked = st.button("🔄 Refresh", key="refresh_btn", type="secondary")
        st.markdown("<style>button[data-testid='refresh_btn'] { font-size: 16px; }</style>", unsafe_allow_html=True)

    with col2:
        reset_clicked = st.button("♻️ Reset", key="reset_btn", type="secondary")
        st.markdown("<style>button[data-testid='reset_btn'] { font-size: 16px; }</style>", unsafe_allow_html=True)

    # Apply CSS classes (manually patch via JS-style rebind)
    st.markdown("""
        <script>
        const refresh = window.parent.document.querySelector('button[data-testid="refresh_btn"]');
        if (refresh) refresh.classList.add('refresh');
        const reset = window.parent.document.querySelector('button[data-testid="reset_btn"]');
        if (reset) reset.classList.add('reset');
        </script>
    """, unsafe_allow_html=True)

    # Logic
    if refresh_clicked:
        st.session_state.refresh_trigger = True
    if reset_clicked:
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()


product_cfg = FUTURES_CONFIG[future_name]
native_ccy_ticker = FUTURES_CONFIG[future_name]['default_currency']

def get_exchange_rate(native, target):
    if native == target:
        return 1.0
    native_fx = fx_usd_rates.get(native)
    target_fx = fx_usd_rates.get(target)
    if target == "USD":
        if native_fx is None or native_fx == 0:
            return 1.0
        return 1 / native_fx
    if native == "USD":
        if target_fx is None or target_fx == 0:
            return 1.0
        return target_fx
    if native_fx and target_fx:
        return native_fx / target_fx
    return 1.0


def refresh_oi_vol_cache_if_old(max_age_seconds=60):
    """Refresh the oi_vol_cache in session_state if it's older than max_age_seconds."""
    now = time.time()
    cache_time_key = "oi_vol_cache_time"

    # If cache is missing, or expired
    if ("oi_vol_cache" not in st.session_state or
        cache_time_key not in st.session_state or
        (now - st.session_state[cache_time_key]) > max_age_seconds):

        # Use your existing function to load fresh oi_vol_cache
        st.session_state["oi_vol_cache"] = load_oi_volume_cache(FUTURES_CONFIG, product_chains)

        # Update timestamp
        st.session_state[cache_time_key] = now

def calculate_dv01_per_lot_from_cache(expiry_ticker, target_ccy_code, future_name, fx_usd_rates):
    for data in oi_vol_cache.values():
        for ex in data.get("expiries", []):
            if ex.get("ticker") == expiry_ticker:
                frsk = ex.get("frsk", 0)
                expiry_ccy = ex.get("currency", FUTURES_CONFIG[future_name]["default_currency"])
                native = expiry_ccy.split()[0]
                fx_cross = custom_fx_cross_rate(native, target_ccy_code, fx_usd_rates)

                root = FUTURES_CONFIG[future_name]["root"]
                multiplier = 20 if root in ["TU", "3Y"] else 10000 if root == "JB" else 10
                return frsk * multiplier * fx_cross
    return float('nan')

def get_frsk_from_cache(expiry_ticker):
    for data in oi_vol_cache.values():
        for ex in data.get("expiries", []):
            if ex.get("ticker") == expiry_ticker:
                return ex.get("frsk", 0)
    return 0

def build_summary_row(
    future_name,
    expiry_ticker,
    dv01_per_lot,
    input_val,
    risk_qualifier,
    ccy_sym,
    target_ccy_code,
    override_target_dv01=None
):
    print(f"DEBUG [{future_name}] → override_target_dv01: {override_target_dv01}")  # <-- this stays for confirmation

    if risk_qualifier == "Lots":
        if override_target_dv01 is not None:
            portfolio_dv01 = override_target_dv01
            calc_lots = portfolio_dv01 / dv01_per_lot if dv01_per_lot else float('nan')
        else:
            calc_lots = input_val
            portfolio_dv01 = calc_lots * dv01_per_lot
    elif risk_qualifier == "DV01 (000's)":
        portfolio_dv01 = input_val * 1000
        calc_lots = portfolio_dv01 / dv01_per_lot if dv01_per_lot else float('nan')
    else:
        portfolio_dv01 = float("nan")
        calc_lots = float("nan")

    native_ccy = FUTURES_CONFIG[future_name]["default_currency"].split()[0]
    fx_rate = custom_fx_cross_rate(native_ccy, target_ccy_code, fx_usd_rates)

    generic_yield = next(
        (
            ex['last_price']
            for data in oi_vol_cache.values()
            for ex in data.get("expiries", [])
            if ex.get("ticker") == FUTURES_CONFIG[future_name]["generic_yield_ticker"]
        ),
        0.0,
    )

    forward_yield = next(
        (
            ex.get("forward_yield", 0.0)
            for data in oi_vol_cache.values()
            for ex in data.get("expiries", [])
            if ex.get("ticker") == expiry_ticker
        ),
        0.0,
    )

    return {
        "Future": expiry_ticker,
        "Lots": f"{calc_lots:,.2f}",
        "DV01 Target": f"{ccy_sym}{portfolio_dv01:,.0f}",
        "Exchange Rate": f"1 {native_ccy} = {fx_rate:.4f} {target_ccy_code}",
        "Generic Yield": f"{generic_yield:.4f}%",
        "Forward Yield": f"{forward_yield:.4f}%"
    }

def render_dv01_summary_table(row1, row2=None):
    def fmt(val): return val if isinstance(val, str) else f"{val}"

    def format_dv01_target(dv01_raw):
        try:
            parts = dv01_raw.strip().split()
            if len(parts) != 2:
                return dv01_raw
            amount_part, ccy_code = parts
            symbol = ''.join([c for c in amount_part if not c.isdigit() and c != ','])
            number = ''.join([c for c in amount_part if c.isdigit()])
            if len(number) >= 4:
                truncated = number[:-3]
                return f"{symbol}{truncated}K DV01/{ccy_code}"
            else:
                return dv01_raw
        except Exception:
            return dv01_raw

    # --- Build Header Row Inside Table ---
    header_row = ""
    if row2:
        try:
            leg1_lots = float(row1['Lots'].replace(',', ''))
            leg2_lots = float(row2['Lots'].replace(',', ''))
            leg1_ratio = 1.00
            leg2_ratio = leg2_lots / leg1_lots if leg1_lots else 0.0
            leg1_ticker = row1['Future'].split()[0]
            leg2_ticker = row2['Future'].split()[0]

            target_dv01_k1 = format_dv01_target(row1['DV01 Target'])
            target_dv01_k2 = format_dv01_target(row2['DV01 Target'])

            header_text = (
                f"{leg1_ticker} / {leg2_ticker} = "
                f"{leg1_lots:.2f} / {leg2_lots:.2f} "
                f"(1:{leg2_ratio:.2f} Ratio) → {target_dv01_k1[:-4]}K DV01/{ccy_code}"
            )
            header_row = f"""
            <tr>
                <th colspan="3" style="font-size:1.05em; font-weight:500; color:#793690; text-align:center;
                                      padding: 10px 6px; background-color:#F3EFFA;">
                    {header_text}
                </th>
            </tr>
            """
        except:
            header_row = ""
    else:
        leg1_ticker = row1['Future'].split()[0]
        header_text = f"{leg1_ticker} = {row1['Lots']}"
        header_row = f"""
        <tr>
            <th colspan="3" style="font-size:1.05em; font-weight:500; color:#793690; text-align:center;
                                  padding: 10px 6px; background-color:#F3EFFA;">
                {header_text}
            </th>
        </tr>
        """

    def row(label, v1, v2="–"):
        return f"""
            <tr>
                <td style="font-weight:500; text-align:left; padding:6px 10px;">{label}</td>
                <td style="text-align:left; padding:6px 10px;">{fmt(v1)}</td>
                <td style="text-align:left; padding:6px 10px;">{fmt(v2)}</td>
            </tr>
        """

    target_dv01_k1 = format_dv01_target(row1['DV01 Target'])
    target_dv01_k2 = format_dv01_target(row2['DV01 Target']) if row2 else "–"

    table_html = f"""
    <html>
    <head>
        <style>
            .summary-container {{
                max-width: 700px;
                margin: -10px;
                font-family: {FONT_STACK};
                font-size: 14.5px;
                padding: 4px;
                border: 0px solid #ddd;
                border-radius: 8px;
                background-color: #fff;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                margin-top: 4px;
                font-family: monospace;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 4px;
            }}
            th {{
                background-color: #f2f2f2;
                font-weight: bold;
            }}
            #copyBtn {{
                background-color: {PRIMARY_COLOR};
                color: white;
                border: none;
                padding: 6px 6px;
                font-weight: 500;
                font-size: 14px;
                border-radius: 3px;
                margin-top: 8px;
                cursor: pointer;
                float: right;
                font-family: {FONT_STACK};
            }}
            #copyBtn:hover {{
                background-color: {LIGHTER_PURPLE};
            }}
        </style>
    </head>
    <body>
        <div class="summary-container">
            <div id="copyBlock">
                <table id="dv01summary">
                    <tbody>
                        {header_row}
                        {row("Future:", row1['Future'], row2['Future'] if row2 else "–")}
                        {row("Calculated Lots:", row1['Lots'], row2['Lots'] if row2 else "–")}
                        {row("Target DV01:", target_dv01_k1, target_dv01_k2)}
                        {row("Exchange Rate:", row1['Exchange Rate'], row2['Exchange Rate'] if row2 else "–")}
                        {row("Generic Yield:", row1['Generic Yield'], row2['Generic Yield'] if row2 else "–")}
                        {row("Forward Yield:", row1['Forward Yield'], row2['Forward Yield'] if row2 else "–")}
                    </tbody>
                </table>
            </div>
            <button id="copyBtn" onclick="copyTable()">Copy to Clipboard</button>
        </div>
        <script>
            function copyTable() {{
                var copyTarget = document.getElementById("copyBlock");
                var range = document.createRange();
                range.selectNode(copyTarget);
                window.getSelection().removeAllRanges();
                window.getSelection().addRange(range);
                try {{
                    document.execCommand('copy');
                }} catch (err) {{
                    alert("Failed to copy.");
                }}
                window.getSelection().removeAllRanges();
            }}
        </script>
    </body>
    </html>
    """

    components.html(table_html, height=257)

if st.session_state.get("refresh_trigger", False):
    st.session_state.refresh_trigger = False  # reset flag

    refresh_oi_vol_cache_if_old(60)
    with st.spinner("Calculating..."):
        oi_vol_cache = st.session_state["oi_vol_cache"]

        ccy_code = ccy.split()[0] if " " in ccy else ccy
        ccy_sym = CURRENCY_SYMBOLS.get(ccy_code, ccy_code)

        try:
            # ✅ Compute target DV01
            dv01_lot_1 = calculate_dv01_per_lot_from_cache(expiry_ticker, ccy.upper(), future_name, fx_usd_rates)
            target_ccy_code = ccy.split()[0] if " " in ccy else ccy
            ccy_sym = CURRENCY_SYMBOLS.get(target_ccy_code, target_ccy_code)

            if risk_qualifier == "Lots":
                target_dv01_value = input_val * dv01_lot_1
            else:
                target_dv01_value = input_val * 1000

            # ✅ Generate matrix
            df_matrix = show_cross_product_dv01_matrix(
                FUTURES_CONFIG, oi_vol_cache, risk_qualifier, input_val, ccy, future_name,
                target_dv01_from_first_leg=target_dv01_value
            )
            st.session_state.df_matrix = df_matrix

            # ✅ Build Summary Rows
            row1 = build_summary_row(
                future_name, expiry_ticker, dv01_lot_1, input_val, risk_qualifier, ccy_sym, target_ccy_code
            )

            row2 = None
            if st.session_state.get("show_future_2", False):
                second_future = st.session_state.last_future_2
                expiry_2 = expiry_details_2[st.session_state.expiry_2_idx]
                expiry_ticker_2 = expiry_2["Ticker"]
                dv01_lot_2 = calculate_dv01_per_lot_from_cache(expiry_ticker_2, ccy.upper(), second_future, fx_usd_rates)

                row2 = build_summary_row(
                    second_future,
                    expiry_ticker_2,
                    dv01_lot_2,
                    input_val,
                    risk_qualifier,
                    ccy_sym,
                    target_ccy_code,
                    override_target_dv01=target_dv01_value if risk_qualifier == "Lots" else None
                )

            # ✅ Save summary rows for rendering
            st.session_state.dv01_summary = (row1, row2)
            st.session_state.just_calculated = True

        except Exception as e:
            st.error(f"Error: {e}\nVerify expiry/currency exist in cache and Bloomberg data.")

# Get the 3-letter currency code from the selected currency (e.g. "USD Curncy" "USD")
ccy_code = ccy.split()[0] if " " in ccy else ccy

# Map to symbol, with fallback to currency code
ccy_sym = CURRENCY_SYMBOLS.get(ccy_code, ccy_code)

if "df_matrix" in st.session_state and st.session_state.df_matrix is not None:
    df_matrix = st.session_state.df_matrix

    # 1. SUMMARY (with Streamlit-native copy button)
    try:
        # --- First Leg ---
        dv01_lot_1 = calculate_dv01_per_lot_from_cache(expiry_ticker, ccy.upper(), future_name, fx_usd_rates)
        row1 = build_summary_row(future_name, expiry_ticker, dv01_lot_1, input_val, risk_qualifier, ccy_sym, target_ccy_code)
    
        # --- Second Leg (if enabled) ---
        row2 = None
        if st.session_state.get("show_future_2", False):
            second_future = st.session_state.last_future_2
            expiry_2 = expiry_details_2[st.session_state.expiry_2_idx]
            expiry_ticker_2 = expiry_2["Ticker"]
            dv01_lot_2 = calculate_dv01_per_lot_from_cache(expiry_ticker_2, ccy.upper(), second_future, fx_usd_rates)
            override_target_dv01 = input_val * dv01_lot_1 if risk_qualifier == "Lots" else None
            row2 = build_summary_row(
                second_future,
                expiry_ticker_2,
                dv01_lot_2,
                input_val,
                risk_qualifier,
                ccy_sym,
                target_ccy_code,
                override_target_dv01=override_target_dv01
            )
    
        # Render summary
        render_dv01_summary_table(row1, row2)
    
    except Exception as e:
        st.error(f"Error computing DV01 summary: {e}")

    # 2. MATRIX (table)
    st.markdown("""
        <div style="text-align:center; font-weight:500; font-size:1.4em; 
                    margin-bottom:0.2em; color:#793690;">
            Cross-Product Matrix
        </div>
    """, unsafe_allow_html=True)

    gb = GridOptionsBuilder.from_dataframe(df_matrix)
    gb.configure_default_column(resizable=True, sortable=True, filter=False)
    gb.configure_column(
        "Generic Yield",
        valueFormatter='function(params) { return params.value != null ? params.value.toFixed(4) + " %" : ""; }'
    )
    gb.configure_column(
        "Cross Market",
        valueFormatter='function(params) { return params.value != null ? params.value.toFixed(4) + " %" : ""; }'
    )

    grid_options = gb.build()

    AgGrid(
        df_matrix,
        gridOptions=grid_options,
        theme="ag-theme-material",
        enable_enterprise_modules=False,
        height=650,
        fit_columns_on_grid_load=True,
        allow_unsafe_jscode=True,
        enableRangeSelection=True,
        enableClipboard=True,
        reload_data=False,
        key="dv01_matrix_grid"
    )

    copy_table_to_clipboard_with_feedback(df_matrix)

    # 3. DECKCHAIR IMAGE
    st.markdown(
        """
        <div style="
            display: flex; 
            flex-direction: column; 
            align-items: center; 
            justify-content: center; 
            margin: 4px 0;
        ">
            <img 
                src=https://personalised-deckchairs.co.uk/cdn/shop/products/GermanyFlagDeckchair.jpg?v=1679584547 
                width="300" 
                alt="Germany Flag Deckchair" 
            />
        </div>
        """, 
        unsafe_allow_html=True,
    )


st.markdown("""<hr style="margin-top:10px;">""", unsafe_allow_html=True)
st.caption("Made for Marex | Powered by Bloomberg & Streamlit")



