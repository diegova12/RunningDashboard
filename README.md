# 🏃 Running Performance Analyzer

An advanced analytics dashboard built with [Streamlit](https://streamlit.io/) to track, analyze, and improve your running performance. Seamlessly integrates with Strava to visualize your training data in ways that go beyond the standard app.


## 🚀 Features

- **Overview**: Visualize your running history with interactive charts for distance, pace, and consistency.
- **Personal Records Tracker**: Automatically detects and tracks your best times for 1 Mile, 5K, 10K, Half Marathon, and Marathon.
- **Advanced Analytics**: 
  - Compare performance across different time periods (e.g., this month vs. last month)
  - GitHub-style **Training Consistency Calendar** heatmap
  - Distance vs. Pace correlation analysis
- **AI-Powered Predictions**:
  - **Race Time Predictor**: Estimates your potential race times using the Riegel formula based on current fitness.
  - **Performance Forecasting**: Uses linear regression to forecast your future pace trends over the next 30-90 days.
  - **Optimal Training Paces**: Calculates personalized training zones (Recovery, Tempo, Interval, etc.) based on recent performance.
- **Training Plan Generator**: Generates a customized weekly training structure based on your specific goal race and time target.

## 🛠️ Technology Stack

- **Frontend**: Streamlit, Plotly
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: Scikit-learn (Linear Regression)
- **Database**: SQLite, SQLAlchemy
- **API Integration**: Strava API

## 🏁 Getting Started

### Prerequisites

- Python 3.9+
- A Strava account
- Strava API Credentials (see [Setup Guide](#strava-api-setup))

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/RunningPerformanceAnalyzer.git
   cd RunningPerformanceAnalyzer
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configuration**
   - Copy the example environment file:
     ```bash
     cp .env.example .env
     ```
   - Open `.env` and add your Strava Client ID and Client Secret.

### Strava API Setup

1. Go to [Strava API Settings](https://www.strava.com/settings/api).
2. Create an application (Category: "Performance Analysis").
3. Copy your `Client ID` and `Client Secret` into your `.env` file.
4. On first run, the app will guide you through the OAuth authorization process.

### Running the App

```bash
streamlit run app.py
```

The dashboard will launch in your default web browser at `http://localhost:8501`.

## 📁 Project Structure

```
RunningPerformanceAnalyzer/
├── app.py                  # Main Streamlit application
├── data/                   # Data storage (SQLite db, raw CSVs)
├── src/                    # Source code
│   ├── api/                # Strava API integration
│   └── database/           # Database models
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```
