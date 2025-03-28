
import hashlib
import json
import secrets
import base64
from typing import Union

class CryptoTool:
    @staticmethod
    def generate_salt(length: int = 16) -> bytes:
        """Generate cryptographically secure random salt."""
        return secrets.token_bytes(length)
    
    @staticmethod
    def hash_password(password: str, salt: bytes = None) -> dict:
        """Hash password with SHA-256 and optional salt."""
        if salt is None:
            salt = CryptoTool.generate_salt()
        
        # Combine password and salt
        salted_password = password.encode() + salt
        
        # Hash using SHA-256
        hashed_password = hashlib.sha256(salted_password).hexdigest()
        
        return {
            'salt': base64.b64encode(salt).decode(),
            'hashed_password': hashed_password
        }
    
    @staticmethod
    def verify_password(input_password: str, stored_salt: Union[str, bytes], 
                        stored_hash: str) -> bool:
        """Verify password against stored hash."""
        if isinstance(stored_salt, str):
            stored_salt = base64.b64decode(stored_salt)
        
        verification_result = CryptoTool.hash_password(input_password, stored_salt)
        return verification_result['hashed_password'] == stored_hash

def main():
    # Example usage
    password = "MySecurePassword123!"
    crypto_result = CryptoTool.hash_password(password)
    print("Password Hashing Result:")
    print(json.dumps(crypto_result, indent=2))
    
    # Verification test
    CryptoTool.verify_password(
        password, 
        crypto_result['salt'], 
        crypto_result['hashed_password']
    )
# Verification test is already handled inside the main() function.

if __name__ == '__main__':
    main()
