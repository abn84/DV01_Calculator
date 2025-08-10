# DV01_Calculator
DV01 Calculator
# 📈 DV01 Calculator – Fixed Income Futures

A professional-grade **DV01 calculator** for fixed income futures, built using [Streamlit](https://streamlit.io/) and [xbbg](https://github.com/matthewgilbert/xbbg), designed for pre-trade analytics and cross-market comparison.

---

## 🧠 Features

- **Bloomberg-integrated**: Fetches live and cached data via `xbbg`
- **Dual-leg support**: Compare risk across two futures contracts
- **Currency normalization**: Converts DV01 into any target currency
- **Lot-based or DV01-based entry**: Toggle risk input method
- **Cross-product matrix**: Instant DV01 comparisons across futures
- **Copy to clipboard**: One-click export of summary/matrix
- **Responsive UI**: Compact layout with Marex branding
- **Easter egg**: A hidden deckchair graphic when risk = beach mode 🏖️

---

## 🚀 Running the App

### Requirements

Python 3.9+ and the following packages (see [`requirements.txt`](./requirements.txt)):

```bash
streamlit
pandas
numpy
xbbg
st-aggrid
