import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
from pathlib import Path
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Add src directory to path for imports
sys.path.append(str(Path(__file__).parent / 'src'))

from database.models import Activity, get_session

# Page configuration
st.set_page_config(
    page_title="Running Performance Analyzer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Helper Functions

@st.cache_data
def load_data():
    """Load activities from database and convert to DataFrame"""
    session = get_session()
    activities = session.query(Activity).all()
    session.close()
    
    data = []
    for activity in activities:
        data.append({
            'id': activity.activity_id,
            'name': activity.name,
            'date': activity.start_date,
            'distance_miles': activity.distance_miles,
            'distance_km': activity.distance_km,
            'pace_min_per_mile': activity.average_pace_min_per_mile,
            'pace_min_per_km': activity.average_pace_min_per_km,
            'time_hours': activity.moving_time_hours,
            'time_minutes': activity.moving_time_minutes,
            'time_seconds': activity.moving_time_seconds,
            'elevation_meters': activity.elevation_gain_meters,
            'avg_hr': activity.average_heartrate,
        })
    
    df = pd.DataFrame(data)
    if len(df) > 0:
        df['date'] = pd.to_datetime(df['date'])
        df['week'] = df['date'].dt.isocalendar().week
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day_of_week'] = df['date'].dt.dayofweek
    return df


def refresh_data_from_strava():
    """Fetch new data from Strava and update database"""
    try:
        from api.strava_fetch import get_all_activities, activities_to_dataframe
        from database.models import Activity, get_session
        from sqlalchemy.exc import IntegrityError
        
        activities = get_all_activities(activity_type='Run')
        
        if not activities:
            return False, "No activities found", 0
        
        df = activities_to_dataframe(activities)
        
        # Save to CSV
        csv_path = Path('data/raw/strava_activities.csv')
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
        
        # Load into database
        session = get_session()
        session.query(Activity).delete()
        session.commit()
        
        added = 0
        for _, row in df.iterrows():
            try:
                activity = Activity(
                    activity_id=str(row['id']),
                    name=row.get('name'),
                    activity_type=row.get('type', 'Run'),
                    start_date=pd.to_datetime(row['start_date']),
                    distance_meters=row.get('distance'),
                    distance_miles=row.get('distance_miles'),
                    distance_km=row.get('distance_km'),
                    moving_time_seconds=row.get('moving_time'),
                    moving_time_minutes=row.get('moving_time_minutes'),
                    moving_time_hours=row.get('moving_time_hours'),
                    average_pace_min_per_mile=row.get('average_pace_min_per_mile'),
                    average_pace_min_per_km=row.get('average_pace_min_per_km'),
                    average_speed=row.get('average_speed'),
                    max_speed=row.get('max_speed'),
                    elevation_gain_meters=row.get('total_elevation_gain'),
                    average_heartrate=row.get('average_heartrate'),
                    max_heartrate=row.get('max_heartrate'),
                )
                session.add(activity)
                session.commit()
                added += 1
            except IntegrityError:
                session.rollback()
        
        session.close()
        return True, f"Successfully refreshed {added} activities", added
        
    except Exception as e:
        return False, f"Error: {str(e)}", 0


def calculate_prs(df):
    """Calculate personal records at standard distances"""
    standard_distances = {
        '1 Mile': (0.9, 1.1),
        '2 Mile': (1.9, 2.1),
        '3 Mile': (2.9, 3.05),
        '5K': (3.06, 3.3),
        '10K': (5.8, 6.5),
        'Half Marathon': (12.5, 13.5),
        'Marathon': (25.5, 26.5)
    }
    
    prs = {}
    for distance_name, (min_miles, max_miles) in standard_distances.items():
        matching = df[(df['distance_miles'] >= min_miles) & (df['distance_miles'] <= max_miles)]
        if len(matching) > 0:
            best = matching.loc[matching['time_seconds'].idxmin()]
            prs[distance_name] = {
                'time': best['time_seconds'],
                'pace': best['pace_min_per_mile'],
                'date': best['date'],
                'name': best['name']
            }
    
    return prs


def calculate_training_zones(df):
    """Calculate training pace zones based on recent performances"""
    # Use best 5K time in last 90 days or overall best
    recent_90 = df[df['date'] >= (datetime.now() - timedelta(days=90))]
    
    if len(recent_90) > 0:
        # Find best 5K equivalent pace
        best_pace = recent_90['pace_min_per_mile'].min()
    else:
        best_pace = df['pace_min_per_mile'].min()
    
    zones = {
        'Easy Run': (best_pace * 1.2, best_pace * 1.4),
        'Marathon Pace': (best_pace * 1.1, best_pace * 1.15),
        'Tempo Run': (best_pace * 1.0, best_pace * 1.05),
        'Threshold': (best_pace * 0.95, best_pace * 1.0),
        'Interval': (best_pace * 0.85, best_pace * 0.95),
    }
    
    return zones, best_pace


def predict_race_times(df):
    """Predict race times using Riegel formula and current fitness"""
    prs = calculate_prs(df)
    
    predictions = {}
    
    # Use best recent performance as baseline
    if '5K' in prs:
        baseline_distance = 3.1  # miles
        baseline_time = prs['5K']['time']
    elif '10K' in prs:
        baseline_distance = 6.2
        baseline_time = prs['10K']['time']
    else:
        # Use average of recent runs
        recent = df[df['date'] >= (datetime.now() - timedelta(days=30))]
        if len(recent) > 0:
            baseline_distance = recent['distance_miles'].mean()
            baseline_time = recent['time_seconds'].mean()
        else:
            return {}
    
    # Riegel formula: T2 = T1 * (D2/D1)^1.06
    target_distances = {
        '5K': 3.1,
        '10K': 6.2,
        'Half Marathon': 13.1,
        'Marathon': 26.2
    }
    
    for race_name, race_distance in target_distances.items():
        predicted_time = baseline_time * (race_distance / baseline_distance) ** 1.06
        predictions[race_name] = predicted_time
    
    return predictions


def forecast_performance(df):
    """Forecast future performance using linear regression"""
    if len(df) < 10:
        return None
    
    # Use data from last 6 months
    recent = df[df['date'] >= (datetime.now() - timedelta(days=180))].copy()
    recent = recent.sort_values('date')
    
    if len(recent) < 5:
        return None
    
    # Convert dates to numeric (days since first run)
    recent['days_since_start'] = (recent['date'] - recent['date'].min()).dt.days
    
    # Fit model for pace improvement
    X = recent['days_since_start'].values.reshape(-1, 1)
    y = recent['pace_min_per_mile'].values
    
    model = LinearRegression()
    model.fit(X, y)
    
    # Predict next 90 days
    future_days = np.arange(recent['days_since_start'].max(), 
                            recent['days_since_start'].max() + 90, 7)
    future_pace = model.predict(future_days.reshape(-1, 1))
    
    return {
        'current_pace': y[-1],
        'predicted_pace_30d': future_pace[4] if len(future_pace) > 4 else None,
        'predicted_pace_90d': future_pace[-1] if len(future_pace) > 0 else None,
        'trend': 'improving' if model.coef_[0] < 0 else 'declining'
    }


def recommend_training_plan(df, goal_distance, goal_time_seconds, weeks_available):
    """Generate a personalized training plan based on execution history"""
    # 1. Analyze Current Fitness (Last 4 weeks)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=28)
    recent_runs = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
    
    current_weekly_avg = 0
    current_long_run = 0
    
    if not recent_runs.empty:
        # Calculate actual weekly average from last 4 weeks
        total_recent_miles = recent_runs['distance_miles'].sum()
        current_weekly_avg = total_recent_miles / 4.0
        current_long_run = recent_runs['distance_miles'].max()
    
    # 2. Define Goal Requirements
    if goal_distance >= 26:
        peak_mileage = max(40, current_weekly_avg * 1.5)
        peak_long_run = 20
        taper_weeks = 3
    elif goal_distance >= 13:
        peak_mileage = max(25, current_weekly_avg * 1.4)
        peak_long_run = 12
        taper_weeks = 2
    elif goal_distance >= 6:
        peak_mileage = max(20, current_weekly_avg * 1.3)
        peak_long_run = 8
        taper_weeks = 1
    else:
        peak_mileage = max(15, current_weekly_avg * 1.2)
        peak_long_run = 5
        taper_weeks = 1
        
    # 3. Build Weekly Schedule
    plan_weeks = []
    
    # We need to bridge current_weekly_avg to peak_mileage
    build_weeks = weeks_available - taper_weeks
    if build_weeks < 1:
        build_weeks = 1
        
    mileage_diff = peak_mileage - current_weekly_avg
    weekly_increment = mileage_diff / build_weeks if build_weeks > 0 else 0
    
    current_planned_mileage = current_weekly_avg
    
    for w in range(1, weeks_available + 1):
        week_type = "Build"
        
        if w > build_weeks:
            # Taper Phase
            week_type = "Taper"
            drop_factor = (w - build_weeks) / (taper_weeks + 1)
            weekly_mileage = peak_mileage * (1 - (0.3 * drop_factor))
        else:
            # Build Phase with recovery weeks
            if w % 4 == 0 and w != build_weeks:
                week_type = "Recovery"
                weekly_mileage = current_planned_mileage * 0.75
            else:
                current_planned_mileage += weekly_increment
                weekly_mileage = current_planned_mileage
        
        # Long run logic
        long_run = min(peak_long_run, current_long_run + (w * 1.0))
        if long_run > weekly_mileage * 0.5:
             long_run = weekly_mileage * 0.5
             
        plan_weeks.append({
            "Week": w,
            "Phase": week_type,
            "Mileage": round(weekly_mileage, 1),
            "Long Run": round(long_run, 1),
            "Focus": f"{week_type} - Long run {round(long_run, 1)}mi"
        })
        
    return {
        "current_status": {
            "avg_weekly": round(current_weekly_avg, 1),
            "max_long_run": round(current_long_run, 1)
        },
        "goal_targets": {
            "peak_weekly": round(peak_mileage, 1),
            "peak_long_run": peak_long_run
        },
        "schedule": pd.DataFrame(plan_weeks)
    }


# Main App

# Title
st.title("Running Performance Analyzer")
st.markdown("Advanced analytics dashboard for tracking and improving your running performance")
st.markdown("---")

# Load data
df = load_data()

if len(df) == 0:
    st.error("No running data found. Please load data from Strava first.")
    st.stop()

# Sidebar
st.sidebar.header("Filters & Actions")

# Refresh button
st.sidebar.subheader("Data Management")
if st.sidebar.button("Refresh from Strava"):
    with st.spinner("Fetching new activities from Strava..."):
        success, message, count = refresh_data_from_strava()
        
        if success:
            st.cache_data.clear()
            st.sidebar.success(f"{message}")
            st.rerun()
        else:
            st.sidebar.error(f"{message}")

st.sidebar.markdown("---")

# Date range filter
st.sidebar.subheader("Date Range")
min_date = df['date'].min().date()
max_date = df['date'].max().date()

date_range = st.sidebar.date_input(
    "Select date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

# Apply filter
if len(date_range) == 2:
    start_date, end_date = date_range
    filtered_df = df[(df['date'].dt.date >= start_date) & (df['date'].dt.date <= end_date)]
else:
    filtered_df = df

st.sidebar.info(f"Showing {len(filtered_df)} of {len(df)} runs")

# Main Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Overview", 
    "Personal Records", 
    "Analytics", 
    "AI Predictions",
    "Training Plans"
])

# Tab 1: Overview
with tab1:
    st.header("Overview")
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Runs", len(filtered_df))
    
    with col2:
        st.metric("Total Miles", f"{filtered_df['distance_miles'].sum():.1f}")
    
    with col3:
        st.metric("Total Hours", f"{filtered_df['time_hours'].sum():.1f}")
    
    with col4:
        avg_pace = filtered_df['pace_min_per_mile'].mean()
        st.metric("Avg Pace", f"{avg_pace:.2f} min/mi")
    
    st.markdown("---")
    
    # Recent Activity
    st.subheader("Recent Activity")
    recent_df = filtered_df.sort_values('date', ascending=False).head(10)
    st.dataframe(
        recent_df[['date', 'name', 'distance_miles', 'pace_min_per_mile', 'time_minutes']],
        width="stretch"
    )
    
    st.markdown("---")
    
    # Miles Over Time
    st.subheader("Miles Over Time")
    df_sorted = filtered_df.sort_values('date')
    fig = px.line(df_sorted, x='date', y='distance_miles', 
                  title='Distance by Run',
                  labels={'distance_miles': 'Miles', 'date': 'Date'})
    st.plotly_chart(fig, width="stretch")
    
    st.markdown("---")
    
    # Distribution charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Pace Distribution")
        fig_pace = px.histogram(
            filtered_df, 
            x='pace_min_per_mile',
            nbins=20,
            title='Distribution of Running Pace',
            labels={'pace_min_per_mile': 'Pace (min/mile)'}
        )
        st.plotly_chart(fig_pace, width="stretch")
    
    with col2:
        st.subheader("Distance Distribution")
        fig_dist = px.histogram(
            filtered_df,
            x='distance_miles',
            nbins=15,
            title='Distribution of Run Distance',
            labels={'distance_miles': 'Distance (miles)'}
        )
        st.plotly_chart(fig_dist, width="stretch")
    
    st.markdown("---")
    
    # Scatter plot
    st.subheader("Distance vs Pace")
    valid_scatter_data = filtered_df.dropna(subset=['distance_miles', 'pace_min_per_mile'])
    
    if len(valid_scatter_data) > 0:
        fig_scatter = px.scatter(
            valid_scatter_data,
            x='distance_miles',
            y='pace_min_per_mile',
            hover_data=['name', 'date'],
            title='How does distance affect your pace?',
            labels={'distance_miles': 'Distance (miles)', 'pace_min_per_mile': 'Pace (min/mile)'},
            trendline='ols'
        )
        fig_scatter.update_traces(marker=dict(size=10, opacity=0.7))
        st.plotly_chart(fig_scatter, width="stretch")
    
    st.markdown("---")
    
    # Monthly mileage
    st.subheader("Monthly Mileage")
    filtered_df_copy = filtered_df.copy()
    filtered_df_copy['year_month'] = filtered_df_copy['date'].dt.to_period('M').astype(str)
    monthly_miles = filtered_df_copy.groupby('year_month')['distance_miles'].sum().reset_index()
    
    fig_monthly = px.bar(
        monthly_miles,
        x='year_month',
        y='distance_miles',
        title='Total Miles by Month',
        labels={'year_month': 'Month', 'distance_miles': 'Miles'}
    )
    st.plotly_chart(fig_monthly, width="stretch")

# Tab 2: Personal Records
with tab2:
    st.header("Personal Records")
    
    prs = calculate_prs(df)
    
    if len(prs) > 0:
        st.success(f"Found PRs at {len(prs)} distances!")
        
        # Display PRs
        for distance, record in prs.items():
            with st.expander(f"{distance}", expanded=True):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    minutes = int(record['time'] // 60)
                    seconds = int(record['time'] % 60)
                    st.metric("Time", f"{minutes}:{seconds:02d}")
                
                with col2:
                    st.metric("Pace", f"{record['pace']:.2f} min/mi")
                
                with col3:
                    st.metric("Date", record['date'].strftime('%Y-%m-%d'))
                
                st.info(f"Run: {record['name']}")
    else:
        st.warning("No personal records found at standard distances. Keep running!")
    
    st.markdown("---")
    
    # Training Zones
    st.subheader("Your Training Zones")
    zones, best_pace = calculate_training_zones(df)
    
    st.info(f"Based on your best pace of {best_pace:.2f} min/mile")
    
    for zone_name, (slow, fast) in zones.items():
        st.write(f"**{zone_name}**: {fast:.2f} - {slow:.2f} min/mile")

# Tab 3: Analytics
with tab3:
    st.header("Advanced Analytics")
    
    # Compare time periods
    st.subheader("Compare Time Periods")
    
    col1, col2 = st.columns(2)
    
    # This month vs last month
    now = datetime.now()
    this_month_start = now.replace(day=1)
    last_month_end = this_month_start - timedelta(days=1)
    last_month_start = last_month_end.replace(day=1)
    
    this_month_df = df[(df['date'] >= this_month_start)]
    last_month_df = df[(df['date'] >= last_month_start) & (df['date'] < this_month_start)]
    
    with col1:
        st.markdown("### This Month")
        st.metric("Runs", len(this_month_df))
        st.metric("Miles", f"{this_month_df['distance_miles'].sum():.1f}")
        st.metric("Avg Pace", f"{this_month_df['pace_min_per_mile'].mean():.2f}" if len(this_month_df) > 0 else "N/A")
    
    with col2:
        st.markdown("### Last Month")
        st.metric("Runs", len(last_month_df))
        st.metric("Miles", f"{last_month_df['distance_miles'].sum():.1f}")
        st.metric("Avg Pace", f"{last_month_df['pace_min_per_mile'].mean():.2f}" if len(last_month_df) > 0 else "N/A")
    
    # Calculate changes
    if len(last_month_df) > 0 and len(this_month_df) > 0:
        miles_change = ((this_month_df['distance_miles'].sum() - last_month_df['distance_miles'].sum()) / 
                       last_month_df['distance_miles'].sum() * 100)
        st.info(f"Month-over-month mileage change: {miles_change:+.1f}%")
    
    st.markdown("---")
    
    # Training Consistency Heatmap
    st.subheader("Training Consistency Heatmap")
    
    # Create calendar heatmap data (GitHub-style)
    df_cal = df.copy()
    df_cal['date_only'] = df_cal['date'].dt.date
    daily_miles = df_cal.groupby('date_only')['distance_miles'].sum().reset_index()
    
    # Create a complete date range for last 180 days
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=179)
    date_range_full = pd.date_range(start=start_date, end=end_date, freq='D')
    daily_miles_full = pd.DataFrame({'date_only': date_range_full.date})
    daily_miles_full = daily_miles_full.merge(daily_miles, on='date_only', how='left')
    daily_miles_full['distance_miles'] = daily_miles_full['distance_miles'].fillna(0)
    
    # Add week and day info
    daily_miles_full['date_only'] = pd.to_datetime(daily_miles_full['date_only'])
    daily_miles_full['day_of_week'] = daily_miles_full['date_only'].dt.dayofweek  # 0=Monday, 6=Sunday
    daily_miles_full['week_num'] = ((daily_miles_full['date_only'] - daily_miles_full['date_only'].min()).dt.days // 7)
    
    # Create pivot table for heatmap (rows = days of week, columns = weeks)
    pivot_data = daily_miles_full.pivot(index='day_of_week', columns='week_num', values='distance_miles')
    
    # Create hover text with dates and miles
    hover_dates = daily_miles_full.pivot(index='day_of_week', columns='week_num', values='date_only')
    hover_text = []
    for i in range(len(pivot_data)):
        row_text = []
        for j in range(len(pivot_data.columns)):
            date_val = hover_dates.iloc[i, j]
            miles_val = pivot_data.iloc[i, j]
            if pd.notna(date_val) and pd.notna(miles_val):
                date_str = date_val.strftime('%Y-%m-%d')
                row_text.append(f"{date_str}<br>{miles_val:.1f} miles")
            else:
                row_text.append("")
        hover_text.append(row_text)
    
    # Create calendar-style heatmap
    day_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    
    fig_heatmap = go.Figure(data=go.Heatmap(
        z=pivot_data.values,
        x=pivot_data.columns,
        y=day_labels,
        hovertext=hover_text,
        hoverinfo='text',
        colorscale=[
            [0, '#ebedf0'],      # No activity
            [0.2, '#c6e48b'],    # Light activity
            [0.4, '#7bc96f'],    # Moderate activity
            [0.6, '#239a3b'],    # Good activity
            [1.0, '#196127']     # High activity
        ],
        colorbar=dict(title="Miles"),
        showscale=True
    ))
    
    fig_heatmap.update_layout(
        title='Training Consistency Calendar (Last 180 Days)',
        xaxis=dict(title='Week', showgrid=False),
        yaxis=dict(title='', showgrid=False),
        height=300,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    st.plotly_chart(fig_heatmap, width="stretch")
    
    st.markdown("---")
    
    # Weekly mileage trend
    st.subheader("Weekly Mileage Trend")
    
    weekly_miles = df.groupby([df['date'].dt.year, df['date'].dt.isocalendar().week])['distance_miles'].sum().reset_index()
    weekly_miles.columns = ['year', 'week', 'miles']
    weekly_miles['year_week'] = weekly_miles['year'].astype(str) + '-W' + weekly_miles['week'].astype(str)
    
    fig_weekly = px.line(
        weekly_miles.tail(20),
        x='year_week',
        y='miles',
        title='Weekly Mileage (Last 20 Weeks)',
        labels={'year_week': 'Week', 'miles': 'Miles'}
    )
    st.plotly_chart(fig_weekly, width="stretch")

# Tab 4: AI Predictions
with tab4:
    st.header("AI-Powered Predictions")
    
    # Race time predictions
    st.subheader("Race Time Predictions")
    st.info("Based on your current fitness using the Riegel formula")
    
    predictions = predict_race_times(df)
    
    if len(predictions) > 0:
        cols = st.columns(4)
        
        for idx, (race_name, predicted_seconds) in enumerate(predictions.items()):
            with cols[idx % 4]:
                minutes = int(predicted_seconds // 60)
                seconds = int(predicted_seconds % 60)
                hours = minutes // 60
                minutes = minutes % 60
                
                if hours > 0:
                    time_str = f"{hours}:{minutes:02d}:{seconds:02d}"
                else:
                    time_str = f"{minutes}:{seconds:02d}"
                
                st.metric(race_name, time_str)
    else:
        st.warning("Need more running data to make predictions")
    
    st.markdown("---")
    
    # Performance forecast
    st.subheader("Performance Forecast")
    
    forecast = forecast_performance(df)
    
    if forecast:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Current Avg Pace", f"{forecast['current_pace']:.2f} min/mi")
        
        with col2:
            if forecast['predicted_pace_30d']:
                change_30 = forecast['predicted_pace_30d'] - forecast['current_pace']
                st.metric("Predicted (30 days)", 
                         f"{forecast['predicted_pace_30d']:.2f} min/mi",
                         f"{change_30:.2f}")
        
        with col3:
            if forecast['predicted_pace_90d']:
                change_90 = forecast['predicted_pace_90d'] - forecast['current_pace']
                st.metric("Predicted (90 days)", 
                         f"{forecast['predicted_pace_90d']:.2f} min/mi",
                         f"{change_90:.2f}")
        
        if forecast['trend'] == 'improving':
            st.success("Your pace is trending faster! Keep up the great work!")
        else:
            st.info("Your pace is steady. Consider adding speed work to improve.")
    else:
        st.warning("Need at least 10 runs in the last 6 months for forecasting")
    
    st.markdown("---")
    
    st.markdown("---")

# Tab 5: Training Plans
with tab5:
    st.header("Training Plan Generator")
    
    st.info("Get a personalized training plan based on your goal race and current fitness")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        goal_race = st.selectbox(
            "Goal Race Distance",
            ["5K (3.1 mi)", "10K (6.2 mi)", "Half Marathon (13.1 mi)", "Marathon (26.2 mi)"]
        )
    
    with col2:
        goal_hours = st.number_input("Goal Hours", min_value=0, max_value=10, value=0)
        goal_minutes = st.number_input("Goal Minutes", min_value=0, max_value=59, value=25)
        goal_seconds = st.number_input("Goal Seconds", min_value=0, max_value=59, value=0)
    
    with col3:
        weeks_to_race = st.number_input("Weeks Until Race", min_value=4, max_value=52, value=12)
    
    if st.button("Generate Training Plan", type="primary"):
        # Parse goal distance
        goal_distance_map = {
            "5K (3.1 mi)": 3.1,
            "10K (6.2 mi)": 6.2,
            "Half Marathon (13.1 mi)": 13.1,
            "Marathon (26.2 mi)": 26.2
        }
        
        goal_distance = goal_distance_map[goal_race]
        goal_time_seconds = goal_hours * 3600 + goal_minutes * 60 + goal_seconds
        
        plan = recommend_training_plan(df, goal_distance, goal_time_seconds, weeks_to_race)
        
        st.success("Personalized Training Plan Generated!")
        
        # Display Plan Overview
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Current Weekly Base", f"{plan['current_status']['avg_weekly']} mi")
        with col2:
            st.metric("Peak Weekly Goal", f"{plan['goal_targets']['peak_weekly']} mi")
        with col3:
            st.metric("Target Long Run", f"{plan['goal_targets']['peak_long_run']} mi")
            
        st.markdown("---")
        
        # Visualizing the Progression
        st.subheader("Weekly Mileage Progression")
        schedule_df = plan['schedule']
        
        # Create a bar chart for mileage
        fig = px.bar(
            schedule_df, 
            x='Week', 
            y='Mileage',
            color='Phase',
            title='Road to Race Day',
            hover_data=['Long Run', 'Focus'],
            color_discrete_map={
                "Build": "#196127",    # Green
                "Recovery": "#c6e48b", # Light Green
                "Taper": "#0366d6"     # Blue
            }
        )
        st.plotly_chart(fig, width="stretch")
        
        # Table View
        with st.expander("View Detailed Schedule"):
            st.dataframe(
                schedule_df.style.format({
                    "Mileage": "{:.1f}",
                    "Long Run": "{:.1f}"
                }),
                width="stretch"
            )