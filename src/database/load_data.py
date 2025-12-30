import pandas as pd
from pathlib import Path
from models import Activity, get_engine, get_session, create_tables
from sqlalchemy.exc import IntegrityError

# Load activities from CSV into database
def load_activities_from_csv(csv_path='data/raw/strava_activities.csv', clear_existing=False):
    print(f"Loading activities from {csv_path} into database...")
    print("=" * 50)
    
    if not Path(csv_path).exists():
        print(f"CSV file {csv_path} does not exist. Please fetch activities first.")
        return
    
    # Read CSV
    print("Reading CSV file...")
    df = pd.read_csv(csv_path)
    print(f"Total activities in CSV: {len(df)}")
    
    # Initialize database
    engine = get_engine()
    create_tables(engine)
    session = get_session(engine)
    
    # Clear existing data if specified
    if clear_existing:
        print("Clearing existing activities from database...")
        session.query(Activity).delete()
        session.commit()
        print("Existing activities cleared.")
        
    # Load activities into database
    added = 0
    skipped = 0
    errors = 0
    
    for idx, row in df.iterrows():
        try:
            # Create Activity object
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
            
            # Staging the activity
            session.add(activity)
            # Write to database
            session.commit()
            added += 1
            
        # Handle duplicate entries
        except IntegrityError:
            # Cancel the staged changes.
            session.rollback()
            skipped += 1
        
        except Exception as e:
            session.rollback()
            print(f"Error adding activity ID {row['id']}: {e}")
            errors += 1
        
    session.close()
    
    print("\n" + "=" * 50)
    print("Load Summary:")
    print(f"Added: {added}")
    print(f"Skipped (duplicates): {skipped}")
    print(f"Errors: {errors}")
    print(f"Total in database: {added + skipped}")
    print("=" * 50)
    
    if added > 0:
        print("\nData successfully loaded into database!")


def verify_database():
    """
    Query the database and show summary statistics
    """
    print("\nDatabase Verification\n")
    print("=" * 50)
    
    session = get_session()
    
    # Count total activities
    total = session.query(Activity).count()
    
    if total == 0:
        print("No activities found in database")
        session.close()
        return
    
    print(f"Found {total} activities in database\n")
    
    # Get all activities
    activities = session.query(Activity).all()
    
    # Convert to list of dictionaries for easier analysis
    data = []
    for activity in activities:
        data.append({
            'date': activity.start_date,
            'name': activity.name,
            'distance_miles': activity.distance_miles,
            'pace': activity.average_pace_min_per_mile,
            'time_hours': activity.moving_time_hours,
        })
    
    df = pd.DataFrame(data)
    
    # Show summary statistics
    print("Summary Statistics:")
    print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"Total distance: {df['distance_miles'].sum():.1f} miles")
    print(f"Total time: {df['time_hours'].sum():.1f} hours")
    print(f"Average pace: {df['pace'].mean():.2f} min/mile")
    print(f"Average distance: {df['distance_miles'].mean():.2f} miles")
    
    # Show most recent 5 runs
    print("\nMost Recent 5 Runs:")
    recent = session.query(Activity).order_by(Activity.start_date.desc()).limit(5).all()
    
    for i, run in enumerate(recent, 1):
        print(f"  {i}. {run.start_date.date()} - {run.name}")
        print(f"     {run.distance_miles:.2f} mi @ {run.average_pace_min_per_mile:.2f} min/mi")
    
    # Show longest run
    print("\nLongest Run:")
    longest = session.query(Activity).order_by(Activity.distance_miles.desc()).first()
    print(f"  {longest.name} - {longest.distance_miles:.2f} miles")
    print(f"  Date: {longest.start_date.date()}")
    
    # Show fastest run (best pace)
    print("\nFastest Run:")
    fastest = session.query(Activity).order_by(Activity.average_pace_min_per_mile).first()
    print(f"  {fastest.name} - {fastest.average_pace_min_per_mile:.2f} min/mile")
    print(f"  Date: {fastest.start_date.date()}")
    print(f"  Distance: {fastest.distance_miles:.2f} miles")
    
    print("\n" + "=" * 50)
    
    session.close()

if __name__ == "__main__":
    import sys
    
    # Check command line arguments
    verify_only = '--verify' in sys.argv or '-v' in sys.argv
    clear = '--clear' in sys.argv or '-c' in sys.argv
    
    if verify_only:
        # Just verify, don't load
        verify_database()
    else:
        # Load data
        if clear:
            print("\n WARNING: This will delete all existing data!")
            confirm = input("Are you sure? (yes/no): ").lower()
            if confirm != 'yes':
                print("Cancelled.")
                sys.exit()
        
        load_activities_from_csv(clear_existing=clear)
        
        # Verify after loading
        verify_database()