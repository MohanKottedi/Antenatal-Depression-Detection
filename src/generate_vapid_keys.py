"""Generate a local VAPID key pair for browser push notifications."""

import base64
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec


def main() -> None:
    output_path = Path("vapid_private.pem")
    if output_path.exists():
        raise FileExistsError(
            f"{output_path} already exists. Move it safely or choose another path."
        )

    private_key = ec.generate_private_key(ec.SECP256R1())
    output_path.write_bytes(
        private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    public_numbers = private_key.public_key().public_numbers()
    public_bytes = (
        b"\x04"
        + public_numbers.x.to_bytes(32, "big")
        + public_numbers.y.to_bytes(32, "big")
    )
    public_key = base64.urlsafe_b64encode(public_bytes).rstrip(b"=").decode("ascii")
    print(f"VAPID_PUBLIC_KEY={public_key}")
    print(f"VAPID_PRIVATE_KEY={output_path.resolve()}")
    print("Keep the private key file secret and do not commit it.")


if __name__ == "__main__":
    main()
