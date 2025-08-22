# RidePulse - NYC Bike Sharing Analytics

RidePulse is a comprehensive analytics platform for NYC Citi Bike sharing data, featuring real-time station analysis, machine learning predictions, and interactive visualizations.

## Features

- 📊 **Real-time Analytics**: Live GBFS data integration for current station status
- 🤖 **Machine Learning Models**: Rebalancing predictions, station clustering, and anomaly detection  
- 🗺️ **Interactive Maps**: Station locations, trends, and live status
- 📈 **Trend Analysis**: Historical patterns and insights
- 🎮 **Interactive Tools**: What-if simulator, quiz, and story builder

## Streamlit App

### Quick Start

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Locally**:
   ```bash
   streamlit run streamlit_app.py
   ```

3. **Access the App**:
   Open http://localhost:8501 in your browser

### Deployment on Streamlit Community Cloud

1. **Fork this repository** to your GitHub account

2. **Deploy on Streamlit Cloud**:
   - Go to https://share.streamlit.io/
   - Click "New app"
   - Connect your GitHub repository
   - Set the main file path to `streamlit_app.py`
   - Click "Deploy"

3. **Configure Secrets** (if needed):
   - In your Streamlit Cloud dashboard, go to app settings
   - Add any required secrets in the "Secrets" section
   - Use TOML format (see `.streamlit/secrets.toml` template)

### Model Integration

The app automatically loads and trains models using:

- **Data Source**: Live NYC Citi Bike GBFS feeds
- **Models**: 
  - Random Forest Classifier for rebalancing predictions
  - K-Means clustering for station grouping
  - Rule-based fallbacks when ML libraries unavailable
- **Features**: Station capacity, fill level, time context, neighborhood patterns

#### Using Your Own Models

To integrate custom models:

1. **Pre-trained Models**: Place model files (`.pkl`, `.joblib`) in the repository root
2. **Custom Functions**: Create inference functions in `src/inference.py`:
   ```python
   def predict(features: dict) -> dict:
       # Your prediction logic
       return {"prediction": "result", "confidence": 0.95}
   ```
3. **Update streamlit_app.py**: Import and use your functions in the `load_model()` function

#### Large Model Files

For models >100MB:
- Use Git LFS for version control
- Consider remote model storage (S3, Hugging Face Hub)
- Load models from URLs in the `load_model()` function
- Document storage requirements in deployment notes

### Secrets Management

The app supports configuration through Streamlit secrets:

1. **Local Development**: Copy `.streamlit/secrets.toml.template` to `.streamlit/secrets.toml`
2. **Streamlit Cloud**: Add secrets through the web interface
3. **Supported Secrets**:
   - Database connections
   - API keys for external services  
   - Custom model endpoints
   - Authentication tokens

### System Requirements

**Streamlit Cloud**:
- No additional system packages needed for basic functionality
- For advanced ML features, all dependencies are in `requirements.txt`

**Local Development**:
- Python 3.8+
- Dependencies: `pip install -r requirements.txt`

### Customization

The app is designed to be easily customizable:

- **Input Features**: Modify the sidebar input widgets in the `main()` function
- **Model Logic**: Update the `load_model()` and `predict_with_model()` functions
- **UI Theme**: Edit `.streamlit/config.toml` for colors and styling
- **Page Layout**: Adjust the column layout and component arrangement

### Troubleshooting

**Common Issues**:

1. **Import Errors**: Ensure all dependencies are installed: `pip install -r requirements.txt`
2. **Network Issues**: The app gracefully handles GBFS API failures with fallback models
3. **Model Loading**: Check error messages in the "Model Status" expander
4. **Deployment**: Verify all files are in the repository and paths are relative

**Debugging**:
- Enable debug mode: `streamlit run streamlit_app.py --logger.level=debug`
- Check browser console for JavaScript errors
- Use the "Diagnostics" expander for detailed error traces

## Multi-page App Structure

The repository also includes a full multi-page Streamlit application:

- `app.py`: Main overview and entry point
- `pages/`: Individual feature pages (Stations, Trends, Models Lab, etc.)
- `utils/`: Shared utilities (GBFS data, themes, helpers)

To run the full multi-page app:
```bash
streamlit run app.py
```

## Development

### Project Structure

```
├── streamlit_app.py          # Production-ready single-page app
├── app.py                    # Multi-page app main entry  
├── pages/                    # Individual app pages
├── utils/                    # Shared utilities
├── data/                     # Local data storage (gitignored)
├── .streamlit/              # Streamlit configuration
│   ├── config.toml          # Theme and server settings
│   └── secrets.toml         # Secret keys (gitignored)
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes and test locally
4. Submit a pull request

## License

This project is open source. Please refer to the license file for details.