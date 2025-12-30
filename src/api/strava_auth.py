import os
import requests # For making HTTP requests
import json
import time
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()
CLIENT_ID = os.getenv("STRAVA_CLIENT_ID")
CLIENT_SECRET = os.getenv("STRAVA_CLIENT_SECRET")

# Creates the URL needed to visit to authorize the app
def get_authorization_url():
    # Base Strava auth URL
    base_url = "https://www.strava.com/oauth/authorize"
    
    # Parameters for the authorization URL
    params = {
        'client_id': CLIENT_ID,
        'response_type': 'code',  # We want a code back
        'redirect_uri': 'http://localhost',  # Where Strava sends you after
        'approval_prompt': 'force',  # Always show the authorization screen
        'scope': 'activity:read_all'  # We want to read all activities
    }
    
    # Build the full URL
    url = f"{base_url}?client_id={params['client_id']}&response_type={params['response_type']}&redirect_uri={params['redirect_uri']}&approval_prompt={params['approval_prompt']}&scope={params['scope']}"
    
    return url

# Exchange the authorization code for an access token
def exchange_code_for_token(authorization_code):
    # Strava's token endpoint
    token_url = "https://www.strava.com/oauth/token"
    
    # Data we need to send to get the token
    payload = {
        'client_id': CLIENT_ID,
        'client_secret': CLIENT_SECRET,
        'code': authorization_code,
        'grant_type': 'authorization_code'
    }
    
    # Make the POST request to get the token
    print("Exchanging code for access token...")
    response = requests.post(token_url, data=payload)
    
    # Check if the request was successful
    if response.status_code == 200:
        token_data = response.json() # Converts Strava's response from JSON to a Python dictionary
        print("Access token obtained successfully.")
        save_token(token_data)  # Save the token data for future use. Expires after 6 hrs.
        return token_data
    else:
        print(f"Error obtaining access token: {response.status_code}")
        print(response.text)
        return None
    

# Add Token Saving functionality
TOKEN_FILE = 'data/strava_token.json'

def save_token(token_data):
    Path('data').mkdir(exist_ok=True)  # Ensure the data folder exists
    
    # Save the token data to a JSON file
    with open(TOKEN_FILE, 'w') as f:
        json.dump(token_data, f, indent=2) # Writes the Python dictionary as a JSON file
    
    print(("Token data saved to", TOKEN_FILE))
    
def load_token():
    try:
        with open(TOKEN_FILE, 'r') as f:
            return json.load(f) # Reads the JSON file and converts it back to a Python dictionary
    except FileNotFoundError:
        return None
    
    
# Refresh the access token
def refresh_access_token():
    # Load the existing token data
    token_data = load_token()
    
    # Check for a token file
    if not token_data:
        print("No token data found. Please authenticate first.")
        return None
    
    # Check for a refresh token
    if 'refresh_token' not in token_data:
        print("No refresh token found. Please authenticate first.")
        return None
    
    print("Refreshing access token...")
    
    # Strava's token endpoint
    token_url = "https://www.strava.com/oauth/token"
    
    # Different payload for refreshing the token
    payload = {
        'client_id': CLIENT_ID,
        'client_secret': CLIENT_SECRET,
        'grant_type': 'refresh_token',
        'refresh_token': token_data['refresh_token']
    }
    
    # Make the POST request to refresh the token
    response = requests.post(token_url, data=payload)
    
    if response.status_code == 200:
        new_token_data = response.json()
        print("Access token refreshed successfully.")
        save_token(new_token_data)  # Save the new token data
        return new_token_data
    else:
        print(f"Error refreshing access token: {response.status_code}")
        print(response.text)
        return None
    
 
# Function used in App to get a valid access token   
def get_valid_access_token():
    # Load existing token data
    token_data = load_token()
    
    if not token_data:
        print("No token data found. Please authenticate first.")
        return None
    
    # Check if the token has expired
    current_time = time.time()
    expires_at = token_data['expires_at']
    
    if expires_at < current_time + 300: # 300 seconds buffer
        print("Access token has expired or is about to expire. Refreshing...")
        token_data = refresh_access_token()
        if not token_data:
            return None
    
    return token_data['access_token']

if __name__ == "__main__":
    import sys
    
    # Check if user wants to test token refresh
    if '--test-refresh' in sys.argv:
        print("\nTesting token refresh...\n")
        token = get_valid_access_token()
        if token:
            print(f" Got valid token: {token[:20]}...")
        sys.exit()  
    
    # Otherwise, run normal authentication
    print("\n Strava Authentication\n")
    print("=" * 50)
    
    # Check if we already have a valid token
    existing_token = load_token()
    if existing_token:
        print(" You already have a token saved.")
        print(f"   Expires at: {existing_token['expires_at']}")
        
        choice = input("\nDo you want to re-authenticate? (y/n): ").lower()
        if choice != 'y':
            print("Using existing token.")
            sys.exit()  
    
    # Step 1: Get the authorization URL
    auth_url = get_authorization_url()
    print("\nVisit this URL in your browser:")
    print(f"\n{auth_url}\n")
    print("Click 'Authorize'")
    print("Copy the 'code' from the URL you're redirected to")
    print("\nExample: http://localhost/?code=XXXXX&scope=...")
    print("Copy everything after 'code=' and before '&scope'\n")
    
    # Step 2: Get the code from user
    auth_code = input("Paste the authorization code here: ").strip()
    
    if not auth_code:
        print("No code provided")
    else:
        # Step 3: Exchange it for a token
        token_data = exchange_code_for_token(auth_code)
        
        if token_data:
            print("\n✅ Authentication complete!")
            print(f"\nYou can now use the API!")