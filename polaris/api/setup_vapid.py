"""CLI script to generate VAPID keys for Web Push notifications.

Usage:
    python -m polaris.api.setup_vapid
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def main():
    print("Polaris API — VAPID Key Generation")
    print("=" * 40)
    print()

    try:
        from py_vapid import Vapid
    except ImportError:
        print("Error: py-vapid is not installed.")
        print("Install it with: pip install py-vapid")
        sys.exit(1)

    vapid = Vapid()
    vapid.generate_keys()

    # Get the keys in the format needed for environment variables
    raw_private = vapid.private_pem()
    raw_public = vapid.public_key

    # Application server key (base64url-encoded)
    import base64
    from cryptography.hazmat.primitives.serialization import (
        Encoding,
        PublicFormat,
    )

    public_bytes = vapid.public_key.public_bytes(
        encoding=Encoding.X962,
        format=PublicFormat.UncompressedPoint,
    )
    public_b64 = base64.urlsafe_b64encode(public_bytes).rstrip(b"=").decode()

    # Private key as base64url (for pywebpush)
    from cryptography.hazmat.primitives.serialization import (
        NoEncryption,
        PrivateFormat,
    )

    private_numbers = vapid.private_key.private_numbers()
    private_bytes = private_numbers.private_value.to_bytes(32, byteorder="big")
    private_b64 = base64.urlsafe_b64encode(private_bytes).rstrip(b"=").decode()

    print("Add these to your .env file:")
    print()
    print(f"VAPID_PRIVATE_KEY={private_b64}")
    print(f"VAPID_PUBLIC_KEY={public_b64}")
    print("VAPID_SUBJECT=mailto:you@example.com")
    print()
    print("The public key is also used in your frontend for PushManager.subscribe().")


if __name__ == "__main__":
    main()
