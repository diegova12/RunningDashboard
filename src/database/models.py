from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

# Base class for the models
Base = declarative_base()

# Main activities table
class Activity(Base):
    __tablename__ = 'activities'
    
    # Primary key
    id = Column(Integer, primary_key=True)
    
    # Strava activity ID. nullable=False to ensure field is always populated
    activity_id = Column(String, unique=True, nullable=False)
    
    name = Column(String)
    activity_type = Column(String, default='Run')
    
    # Date/Time
    start_date = Column(DateTime)
    
    # Distance
    distance_meters = Column(Float)
    distance_miles = Column(Float)
    distance_km = Column(Float)
    
    # Time
    moving_time_seconds = Column(Float)
    moving_time_minutes = Column(Float)
    moving_time_hours = Column(Float)
    
    # Speed/Pace
    average_pace_min_per_mile = Column(Float)
    average_pace_min_per_km = Column(Float)
    average_speed = Column(Float)
    max_speed = Column(Float)
    
    # Elevation
    elevation_gain_feet = Column(Float)
    elevation_gain_meters = Column(Float)
    
    # Heart Rate
    average_heartrate = Column(Float)
    max_heartrate = Column(Float)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
# Database Helper Functions
def get_engine(database_url=None):
    if database_url is None:
        database_url = os.getenv("DATABASE_URL", "sqlite:///data/running_data.db")
    
    return create_engine(database_url, echo=False)

def create_tables(engine=None):
    if engine is None:
        engine = get_engine()
    Base.metadata.create_all(engine)
    print("Database tables created.")
    
def get_session(engine=None):
    if engine is None:
        engine = get_engine()
    Session = sessionmaker(bind=engine)
    return Session()

if __name__ == "__main__":
    print("\n🗄️  Initializing Database\n")
    print("=" * 50)
    
    engine = get_engine()
    create_tables(engine)
    
    print(f"\nDatabase created at: data/running_data.db")
    print("Table created: activities") 
    