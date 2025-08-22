# RidePulseProject

A comprehensive bikeshare analytics platform built with Streamlit, providing real-time GBFS (General Bikeshare Feed Specification) data analysis and purpose-built models for operational insights.

## Features

- **Real-time Data Integration**: Live GBFS data from Citi Bike NYC
- **Interactive Analytics**: Purpose-built models for bikeshare operations
- **Multi-page Application**: Organized tools and visualizations
- **Responsive Design**: Theme-aware interface with dark/light mode support

## Streamlit App

### Prerequisites

- Python 3.10+ (recommended)
- pip package manager

### Setup and Run Locally

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Adithya-Vasudevan/RidePulseProject.git
   cd RidePulseProject
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app**:
   ```bash
   streamlit run streamlit_app.py
   ```

4. **Access the application**:
   Open your browser to `http://localhost:8501`

### Multipage App Structure

This application uses Streamlit's multipage feature. Pages located in the `pages/` directory will automatically appear in the sidebar navigation:

- **Models Lab** (`pages/04_Models_Lab.py`): Interactive analytics with machine learning models
- **Other Pages**: Additional analysis tools and visualizations as available

The main `streamlit_app.py` serves as the Home page with project overview and data status.

### Deployment to Streamlit Community Cloud

1. **Create a new app** on [Streamlit Community Cloud](https://share.streamlit.io/)

2. **Configure the app**:
   - Repository: Point to this GitHub repository
   - Branch: `main` (or your target branch)
   - Main file path: `streamlit_app.py`

3. **Environment Configuration**:
   - The app will automatically install dependencies from `requirements.txt`
   - No additional system packages should be needed

4. **Secrets Management** (if needed):
   - Configure any required secrets through the Streamlit Cloud UI
   - Go to your app settings → Secrets
   - Add secrets in TOML format
   - **Never commit secrets to the repository**

5. **Deploy**:
   - Click "Deploy" and wait for the application to build and start
   - The app will be available at your assigned Streamlit Cloud URL

### Local Development Notes

- The app uses automatic OS theme detection (dark/light mode)
- Live GBFS data requires internet connectivity
- Pages under `pages/` are automatically discovered by Streamlit
- Configuration is stored in `.streamlit/config.toml`

### Analytics Components

The application includes several analytics capabilities:

- **🎯 Rebalancing Classifier**: Identifies stations needing bike redistribution
- **🗺️ Station Clustering**: Groups stations by operational characteristics  
- **🔍 Anomaly Detection**: Detects unusual usage patterns
- **🎮 What‑If Simulator**: Interactive scenario modeling

All models operate on real-time GBFS data when available.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test locally with `streamlit run streamlit_app.py`
5. Submit a pull request

## License

This project is open source. Please check the repository for license details.