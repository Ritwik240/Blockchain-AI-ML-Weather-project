import hashlib
import json
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives import serialization


def sha256_hash(data):

    if isinstance(data, dict):
        data = json.dumps(data, sort_keys=True).encode()

    if isinstance(data, str):
        data = data.encode()

    return hashlib.sha256(data).hexdigest()


def hash_file(file_path):

    with open(file_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def generate_keypair():

    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key()

    private_bytes = private_key.private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption()
    )

    public_bytes = public_key.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw
    )

    return private_bytes.hex(), public_bytes.hex()


def sign_data(private_key_hex, data):

    private_key = Ed25519PrivateKey.from_private_bytes(bytes.fromhex(private_key_hex))

    signature = private_key.sign(data.encode())

    return signature.hex()


def verify_signature(public_key_hex, data, signature_hex):

    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    public_key = Ed25519PublicKey.from_public_bytes(bytes.fromhex(public_key_hex))

    try:
        public_key.verify(bytes.fromhex(signature_hex), data.encode())
        return True
    except:
        return False