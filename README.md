# NutriSense - AI Soil Intelligence

A precision agriculture application that analyzes soil data and provides AI-powered recommendations for optimal crop yields.

## Features

- **Soil Analysis**: Comprehensive analysis of pH, EC, moisture, NPK, and microbial activity
- **AI Recommendations**: Groq-powered suggestions for crops, fertilizers, and irrigation
- **Visual Dashboard**: Interactive charts and health scoring
- **History Tracking**: SQLite database for analysis history
- **Real-time Insights**: Instant soil health interpretation

## Quick Start

### Local Development

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set API Key** (Optional - for AI features)
   Create `.streamlit/secrets.toml`:
   ```toml
   GROQ_API_KEY = "your_groq_api_key_here"
   ```

3. **Run Application**
   ```bash
   streamlit run app.py
   ```

4. **Access Dashboard**
   Open http://localhost:8501

### Docker Development

1. **Build Container**
   ```bash
   docker build -t nutrisense .
   ```

2. **Run Container**
   ```bash
   docker run -p 8501:8501 nutrisense
   ```

### Dev Container (VS Code)

1. Open in VS Code
2. Install "Dev Containers" extension
3. Command Palette → "Reopen in Container"

## Usage

1. **Input Data**: Enter soil test results in the "Input Data" tab
2. **Dashboard**: View comprehensive analysis and health scores
3. **AI Analysis**: Get crop suggestions and recommendations
4. **History**: Track previous analyses and trends

## Soil Parameters

| Parameter | Optimal Range | Unit |
|-----------|---------------|------|
| pH | 6.5 - 7.5 | - |
| EC | < 0.8 | mS/cm |
| Moisture | 25 - 40 | % |
| Nitrogen | 40 - 80 | mg/kg |
| Phosphorus | 20 - 50 | mg/kg |
| Potassium | 100 - 250 | mg/kg |
| Microbial | 5 - 7 | Index |

## Project Structure

```
├── app.py                 # Main Streamlit application (single-file)
├── data/                  # SQLite databases (auto-created)
├── .streamlit/            # Streamlit configuration
├── .devcontainer/         # Development container setup
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Technology Stack

- **Frontend**: Streamlit
- **Visualization**: Plotly
- **Database**: SQLite
- **AI**: Groq API (Llama 3.1)
- **Validation**: Pydantic
- **Data**: Pandas

## Contributing

1. Fork the repository
2. Create feature branch
3. Make changes
4. Test thoroughly
5. Submit pull request

## License

MIT License - see LICENSE file for details