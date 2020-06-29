mkdir -p ~/.streamlit/

echo "[server]
headless = true
const host = '0.0.0.0'
const port = process.env.PORT || 3000
enableCORS = false
" > ~/.streamlit/config.toml
