"""
Kite API Authentication Helper
==============================
Simplified script to get Kite Connect access token.

Usage:
    python scripts/kite_auth_helper.py
"""

from kiteconnect import KiteConnect
import os

def main():
    print("="*60)
    print("Kite Connect Authentication Helper")
    print("="*60)
    print()

    # Get API credentials
    api_key = input("Enter your Kite API Key: ").strip()
    api_secret = input("Enter your Kite API Secret: ").strip()

    if not api_key or not api_secret:
        print("\nError: API Key and Secret are required!")
        return

    # Initialize KiteConnect
    kite = KiteConnect(api_key=api_key)

    # Generate login URL
    login_url = kite.login_url()

    print("\n" + "="*60)
    print("Step 1: Login to Kite")
    print("="*60)
    print("\n1. Open this URL in your browser:")
    print(f"\n   {login_url}")
    print("\n2. Login with your Zerodha credentials")
    print("3. After login, you'll be redirected to a URL like:")
    print("   http://127.0.0.1/?request_token=XXXXXX&action=login&status=success")
    print("\n4. Copy the 'request_token' value from the URL")
    print()

    # Get request token from user
    request_token = input("Enter the request_token from URL: ").strip()

    if not request_token:
        print("\nError: Request token is required!")
        return

    try:
        # Generate session
        print("\nGenerating session...")
        data = kite.generate_session(request_token, api_secret=api_secret)
        access_token = data["access_token"]

        print("\n" + "="*60)
        print("✓ Authentication Successful!")
        print("="*60)
        print(f"\nAccess Token: {access_token}")
        print(f"\nUser ID: {data.get('user_id', 'N/A')}")
        print(f"User Name: {data.get('user_name', 'N/A')}")
        print(f"Email: {data.get('email', 'N/A')}")

        # Save to environment file
        env_file = '.env.kite'
        with open(env_file, 'w') as f:
            f.write(f"KITE_API_KEY={api_key}\n")
            f.write(f"KITE_ACCESS_TOKEN={access_token}\n")

        print(f"\n✓ Credentials saved to: {env_file}")

        print("\n" + "="*60)
        print("Next Steps")
        print("="*60)
        print("\n1. Set environment variables:")
        print(f"   export KITE_API_KEY={api_key}")
        print(f"   export KITE_ACCESS_TOKEN={access_token}")
        print()
        print("2. Or load from .env file:")
        print("   source .env.kite")
        print()
        print("3. Fetch historical data:")
        print("   python scripts/kite_data_collector.py \\")
        print("       --symbols NIFTY100 \\")
        print("       --start-date 2021-01-01 \\")
        print("       --end-date 2024-12-31 \\")
        print("       --interval 5minute")
        print()
        print("⚠️  Note: Access token expires daily. Re-run this script tomorrow.")
        print()

    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nPlease check:")
        print("1. API Key and Secret are correct")
        print("2. Request token is valid (it expires quickly)")
        print("3. You copied the entire request_token value")


if __name__ == '__main__':
    main()
