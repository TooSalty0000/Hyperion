"""CLI script to set the Polaris API password.

Usage:
    python -m polaris.api.setup_password
"""

import asyncio
import getpass
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


async def main():
    # Load environment
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")

    print("Polaris API — Password Setup")
    print("=" * 40)
    print()

    password = getpass.getpass("Enter new password: ")
    if not password:
        print("Error: Password cannot be empty.")
        sys.exit(1)

    confirm = getpass.getpass("Confirm password: ")
    if password != confirm:
        print("Error: Passwords do not match.")
        sys.exit(1)

    if len(password) < 8:
        print("Warning: Password is shorter than 8 characters.")

    # Connect to Redis and store the hash
    import redis.asyncio as redis_async

    client = await redis_async.from_url(redis_url, encoding="utf-8", decode_responses=True)

    try:
        from polaris.api.auth import set_password

        success = await set_password(password, client)
        if success:
            print()
            print("Password set successfully.")
            print()
            print("Make sure POLARIS_APP_SECRET is set in your .env file.")
            print("You can generate one with: python -c \"import secrets; print(secrets.token_hex(32))\"")
        else:
            print("Error: Failed to set password. Check that pwdlib[bcrypt] is installed.")
            sys.exit(1)
    finally:
        await client.aclose()


if __name__ == "__main__":
    asyncio.run(main())
