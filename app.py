import streamlit as st
import cv2
import pytesseract
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import tempfile
from PIL import Image
import numpy as np

st.set_page_config(page_title="CNC Trade Planner", layout="centered")

# 1. Extract table from option chain image
def extract_option_chain(image_path):
    img = cv2.imread(image_path)
    text = pytesseract.image_to_string(img)
    lines = text.split("\n")
    data = [line.split() for line in lines if line.strip()]
    df = pd.DataFrame(data)
    
    # Simplified cleaning (assumes 3 columns: Strike, Call_OI, Put_OI)
    if len(df.columns) >= 3:
        df = df.iloc[:, :3]
        df.columns = ['Strike', 'Call_OI', 'Put_OI']
    else:
        st.error("Failed to detect proper columns. Try with a clearer image.")
    return df

# 2. Identify Support & Resistance
def get_trade_levels(df, spot):
    df['Strike'] = pd.to_numeric(df['Strike'], errors="coerce")
    df['Call_OI'] = pd.to_numeric(df['Call_OI'], errors="coerce")
    df['Put_OI'] = pd.to_numeric(df['Put_OI'], errors="coerce")
    df.dropna(inplace=True)

    support = df.loc[df['Strike'] <= spot].sort_values("Put_OI", ascending=False).iloc[0]['Strike']
    resistance = df.loc[df['Strike'] >= spot].sort_values("Call_OI", ascending=False).iloc[0]['Strike']

    return support, resistance

# 3. Generate CNC Trade Plan Card
def plot_trade_plan(spot, support, resistance):
    stoploss = support - 0.5 * (spot * 0.01)
    targets = [resistance, resistance+20, resistance+40]

    text = f"""
📌 CNC Trade Plan Card

🔵 Spot Price: {spot}
✅ Entry Zone (Buy): {support} – {spot}
🛑 Stop Loss: {stoploss:.2f}
🎯 Targets: {targets[0]}, {targets[1]}, {targets[2]}

⚖️ Risk: {spot-stoploss:.2f} | Reward: {targets[0]-spot}
⭐ Risk-Reward Ratio: {(targets[0]-spot)/(spot-stoploss):.2f}
    """

    fig, ax = plt.subplots(figsize=(7,5))
    box = FancyBboxPatch((0.05,0.05), 0.9, 0.9, boxstyle="round,pad=0.02",
                         linewidth=2, edgecolor="black", facecolor="#f9f9f9")
    ax.add_patch(box)
    ax.text(0.07, 0.95, text, fontsize=11, va="top", ha="left", family="monospace")
    ax.axis("off")
    plt.title("CNC Trade Plan Card", fontsize=14, weight="bold")
    st.pyplot(fig)

# ------------------ Streamlit UI ------------------

st.title("📊 CNC Trade Planner")
st.markdown("Upload option chain image & enter spot price to get your trade plan.")

uploaded_file = st.file_uploader("📷 Upload Option Chain Image", type=["png", "jpg", "jpeg"])
spot_price = st.number_input("💹 Enter Spot Price", min_value=0.0, value=1000.0, step=1.0)

if uploaded_file and spot_price:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        img = Image.open(uploaded_file)
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        cv2.imwrite(tmp_file.name, img_cv)

        df = extract_option_chain(tmp_file.name)
        if not df.empty:
            st.subheader("📄 Extracted Option Chain")
            st.dataframe(df)

            support, resistance = get_trade_levels(df, spot_price)
            st.success(f"🟢 Support: {support} | 🔴 Resistance: {resistance}")

            st.subheader("🧾 CNC Trade Plan Card")
            plot_trade_plan(spot_price, support, resistance)
