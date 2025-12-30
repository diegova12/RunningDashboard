import requests
import time
import pandas as pd
from pathlib import Path
from strava_auth import get_valid_access_token

# Fetch activities from Strava API. Returns as list of activity dictionaries.
def get_activities(per_page=30, page=1):
    # Get a valid access token
    access_token = get_valid_access_token()
    
    if not access_token:
        print("Unable to get a valid access token.")
        return []
    
    # Set up the request
    url = "https://www.strava.com/api/v3/athlete/activities"
    
    # Bearer is the type of authentication. Every API request needs this header.
    headers = {
        'Authorization': f'Bearer {access_token}'   
    }
    
    # Query parameters for pagination. Pagination means the API splits results into "pages" to avoid sending too much data at one
    params = {
        'per_page': per_page,
        'page': page
    }
    
    # Make the request
    print("Fetching activities from Strava API...")
    # Get request to Strava API
    response = requests.get(url, headers=headers, params=params)
    
    if response.status_code == 200:
        activities = response.json()
        print(f"Fetched {len(activities)} activities.")
        return activities
    else:
        print(f"Error fetching activities: {response.status_code}")
        print(response.text)
        return []
    
def get_all_activities(activity_type='Run'):
    all_activites = []
    page = 1
    
    print(f"Fetching all activities of type: {activity_type}")
    
    while True:
        # Fetch this page
        activities = get_activities(per_page=200, page=page)
        
        # If no activities returned, break the loop
        if not activities:
            break
        
        # Filter activities by type
        filtered = [a for a in activities if a.get('type') == activity_type]
        # Adds all items from one list to another
        all_activites.extend(filtered)
        
        print(f"Page {page}: Found {len(filtered)} '{activity_type}' activities.")
        
        # If we got less than 200, we are done
        if len(activities) < 200:
            break
        
        page += 1
        
        # To avoid hitting rate limits, pause briefly
        time.sleep(0.5)
        
    print(f"Total '{activity_type}' activities so far: {len(all_activites)}")
    return all_activites
    
# Convert list of activity dicts to pandas DataFrame
def activities_to_dataframe(activities):
    if not activities:
        return pd.DataFrame()  # Return empty DataFrame if no activities
    
    df = pd.DataFrame(activities)
    
    columns_to_keep = [
        'id',
        'name',
        'type',
        'start_date',
        'distance',
        'moving_time',
        'elapsed_time',
        'total_elevation_gain',
        'average_speed',
        'max_speed',
        'average_heartrate',
        'max_heartrate',
        'calories',
    ]
    
    # Keep only the columns that exist in the DataFrame
    columns_to_keep = [col for col in columns_to_keep if col in df.columns]
    df = df[columns_to_keep]
    
    if 'distance' in df.columns:
        df['distance_miles'] = df['distance'] / 1609.34  # Convert meters to miles
        df['distance_km'] = df['distance'] / 1000  # Convert meters to kilometers
    
    if 'moving_time' in df.columns:
        df['moving_time_minutes'] = df['moving_time'] / 60  # Convert seconds to minutes
        df['moving_time_hours'] = df['moving_time'] / 3600  # Convert seconds to hours
    
    if 'moving_time' in df.columns and 'distance' in df.columns:
        df['average_pace_min_per_mile'] = (df['moving_time'] / 60) / df['distance_miles']  # min/mile
        df['average_pace_min_per_km'] = (df['moving_time'] / 60) / df['distance_km']  # min/km
    
    if 'start_date' in df.columns:
        df['start_date'] = pd.to_datetime(df['start_date'])

    return df

# Save activities to CSV
def save_activities_to_csv(activities, filename='data/raw/strava_activities.csv'):
    # Convert to DataFrame
    df = activities_to_dataframe(activities)

    if df.empty:
        print("No activities to save.")
        return
    
    # Create directory if it doesn't exist
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV
    df.to_csv(filename, index=False)
    
    print(f"Activities saved to {filename}")
    print(f" Columns saved: {list(df.columns)}")
    
    return df
    
if __name__ == "__main__":
    import sys
    
    if '--all' in sys.argv:
        # Fetch ALL activities
        activities = get_all_activities(activity_type='Run')
        
        if activities:
            # Save to CSV
            df = save_activities_to_csv(activities)
            
            # Show summary
            print("\nSummary Statistics:")
            print(f"   Total runs: {len(df)}")
            print(f"   Total distance: {df['distance_miles'].sum():.1f} miles")
            print(f"   Total time: {df['moving_time_hours'].sum():.1f} hours")
            print(f"   Average pace: {df['average_pace_min_per_mile'].mean():.2f} min/mile")
            print(f"   Date range: {df['start_date'].min().date()} to {df['start_date'].max().date()}")
    
    else:
        # Just fetch first 10 for testing
        activities = get_activities(per_page=10, page=1)
        
        if activities:
            print(f"\nSample of your first activity:\n")
            first = activities[0]
            print(f"Name: {first.get('name')}")
            print(f"Type: {first.get('type')}")
            print(f"Distance: {first.get('distance', 0) / 1609.34:.2f} miles")
            print(f"Time: {first.get('moving_time', 0) / 60:.1f} minutes")
            print(f"Date: {first.get('start_date')}")
            
            print(f"\nRun with --all to fetch and save all activities")