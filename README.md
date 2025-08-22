# RidePulseProject 🚲

A comprehensive Streamlit application for bikeshare data analytics and machine learning, built on live GBFS (General Bikeshare Feed Specification) data.

## Features

- **🧪 Models Lab** - Interactive ML models for rebalancing, clustering, and anomaly detection
- **📈 Trends** - Historical usage patterns and trend analysis
- **📍 Stations** - Station-level analytics and insights
- **🗺️ Live Map** - Real-time station status visualization
- **🔍 Fun Facts** - Discover interesting bikeshare insights
- **🧠 Quiz** - Test your bikeshare knowledge
- **📖 Story Builder** - Create data-driven narratives

## Streamlit App

### Prerequisites

- Python 3.10+ (recommended)
- pip package manager

### Setup and Run Locally

1. **Clone the repository**
   ```bash
   git clone https://github.com/Adithya-Vasudevan/RidePulseProject.git
   cd RidePulseProject
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run streamlit_app.py
   ```

4. **Open your browser**
   - The app will automatically open at `http://localhost:8501`
   - If not, navigate to the URL shown in the terminal

### Multipage App Structure

This Streamlit app uses the built-in multipage functionality:

- **Main page**: `streamlit_app.py` - Home dashboard with project overview and data status
- **Additional pages**: Files in the `pages/` directory automatically appear in the sidebar
  - `pages/04_Models_Lab.py` - Machine learning models and interactive tools
  - `pages/03_Trends.py` - Trend analysis and historical data
  - `pages/02_Stations.py` - Station-level analytics
  - `pages/08_Live_Map.py` - Real-time map visualization
  - `pages/05_Fun_Facts.py` - Data insights and facts
  - `pages/06_Quiz.py` - Interactive quiz
  - `pages/07_Story_Builder.py` - Narrative creation tool

### Development

The application includes:
- **Live GBFS data integration** - Real-time bikeshare station data
- **Machine learning models** - Built with scikit-learn
- **Interactive visualizations** - Using Plotly and Streamlit
- **Theme-aware UI** - Supports light and dark modes
- **Data caching** - Optimized performance with Streamlit caching

## Deployment to Streamlit Community Cloud

### Prerequisites
- GitHub account
- Streamlit Community Cloud account (free at [share.streamlit.io](https://share.streamlit.io))

### Deployment Steps

1. **Fork or clone this repository** to your GitHub account

2. **Create a new app** on Streamlit Community Cloud:
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Click "New app"
   - Connect your GitHub repository
   - Select branch: `main` (or your preferred branch)
   - **Main file path**: `streamlit_app.py`
   - Choose your app URL

3. **Configure secrets** (if needed):
   - In your deployed app, go to Settings → Secrets
   - Add any required environment variables or API keys
   - **DO NOT** commit secrets to your repository

4. **Deploy**:
   - Click "Deploy!"
   - Your app will be available at `https://your-app-name.streamlit.app`

### Environment Variables

If your app requires environment variables or secrets:
- Use the Streamlit Cloud UI (Settings → Secrets) to configure them
- Access them in your code using `st.secrets`
- Never commit sensitive information to version control

Example secrets configuration in Streamlit Cloud:
```toml
# In the Streamlit Cloud Secrets UI, not in your repo
[api_keys]
some_key = "your-secret-value"

[database]
url = "your-database-url"
```

## Project Structure

```
RidePulseProject/
├── streamlit_app.py          # Main application entry point
├── pages/                    # Streamlit pages (auto-discovered)
│   ├── 02_Stations.py
│   ├── 03_Trends.py
│   ├── 04_Models_Lab.py     # ML models and interactive tools
│   ├── 05_Fun_Facts.py
│   ├── 06_Quiz.py
│   ├── 07_Story_Builder.py
│   └── 08_Live_Map.py
├── utils/                    # Utility modules
│   ├── gbfs.py              # GBFS data fetching
│   ├── badges.py            # User badges system
│   ├── theme.py             # UI theming
│   └── ...
├── .streamlit/              # Streamlit configuration
│   └── config.toml          # App configuration
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test locally with `streamlit run streamlit_app.py`
5. Submit a pull request

## License

This project is open source. Please check the repository for license details.