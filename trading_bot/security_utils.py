import os
import base64
import secrets
import hashlib
import logging
import hmac
import re
import json
from typing import Dict, Any, Optional, List, Tuple, Union
from datetime import datetime

logger = logging.getLogger("SecurityUtils")

# Constants
HASH_ALGORITHM = "sha256"
HASH_ITERATIONS = 100000  # OWASP recommended minimum
SALT_LENGTH = 32  # bytes
KEY_LENGTH = 32  # bytes

def generate_secure_token(length: int = 32) -> str:
    """Generate a cryptographically secure random token.
    
    Args:
        length: Length of the token in bytes before base64 encoding
        
    Returns:
        Base64 encoded token string
    """
    token_bytes = secrets.token_bytes(length)
    return base64.urlsafe_b64encode(token_bytes).decode('utf-8').rstrip('=')

def hash_password(password: str, salt: bytes = None) -> Tuple[bytes, bytes]:
    """Hash a password securely using PBKDF2.
    
    Args:
        password: Password string to hash
        salt: Optional salt bytes. If None, a random salt will be generated
        
    Returns:
        Tuple of (password_hash, salt)
    """
    if salt is None:
        salt = os.urandom(SALT_LENGTH)
        
    password_bytes = password.encode('utf-8')
    
    # Use PBKDF2 with appropriate parameters
    password_hash = hashlib.pbkdf2_hmac(
        HASH_ALGORITHM,
        password_bytes,
        salt,
        HASH_ITERATIONS,
        KEY_LENGTH
    )
    
    return (password_hash, salt)

def verify_password(password: str, stored_hash: bytes, salt: bytes) -> bool:
    """Verify a password against a stored hash.
    
    Args:
        password: Password string to verify
        stored_hash: Previously hashed password
        salt: Salt used for hashing
        
    Returns:
        True if password matches, False otherwise
    """
    password_hash, _ = hash_password(password, salt)
    
    # Use constant-time comparison to prevent timing attacks
    return hmac.compare_digest(password_hash, stored_hash)

def format_password_for_storage(password_hash: bytes, salt: bytes) -> str:
    """Format password hash and salt for storage in database.
    
    Args:
        password_hash: Hashed password bytes
        salt: Salt bytes
        
    Returns:
        String in format: algorithm$iterations$salt$hash
    """
    salt_b64 = base64.b64encode(salt).decode('utf-8')
    hash_b64 = base64.b64encode(password_hash).decode('utf-8')
    
    return f"{HASH_ALGORITHM}${HASH_ITERATIONS}${salt_b64}${hash_b64}"

def parse_stored_password(stored_password: str) -> Tuple[str, int, bytes, bytes]:
    """Parse a stored password string back into components.
    
    Args:
        stored_password: Password string in format: algorithm$iterations$salt$hash
        
    Returns:
        Tuple of (algorithm, iterations, salt, hash)
    """
    try:
        parts = stored_password.split('$')
        if len(parts) != 4:
            raise ValueError("Invalid stored password format")
            
        algorithm = parts[0]
        iterations = int(parts[1])
        salt = base64.b64decode(parts[2])
        password_hash = base64.b64decode(parts[3])
        
        return (algorithm, iterations, salt, password_hash)
    except Exception as e:
        logger.error(f"Error parsing stored password: {e}")
        raise ValueError("Invalid stored password format")

def get_secure_env_var(name: str, default: str = None) -> Optional[str]:
    """Get an environment variable securely.
    
    Args:
        name: Name of environment variable
        default: Default value if variable is not set
        
    Returns:
        Environment variable value or default
    """
    value = os.environ.get(name, default)
    
    # Log access but not actual value
    if value is not None:
        masked_value = "*" * 8
        logger.debug(f"Accessed env var {name}: {masked_value}")
    else:
        logger.debug(f"Accessed env var {name}: not set")
        
    return value

def get_secure_env_var_json(name: str, default: Dict[str, Any] = None) -> Dict[str, Any]:
    """Get an environment variable and parse it as JSON.
    
    Args:
        name: Name of environment variable
        default: Default value if variable is not set or invalid
        
    Returns:
        Parsed JSON as dict or default
    """
    if default is None:
        default = {}
        
    value = get_secure_env_var(name)
    if value is None:
        return default
        
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        logger.error(f"Failed to parse env var {name} as JSON")
        return default

def validate_password_strength(password: str) -> Tuple[bool, str]:
    """Validate password strength based on OWASP recommendations.
    
    Args:
        password: Password to validate
        
    Returns:
        Tuple of (valid, reason) where reason is None for valid passwords
    """
    # Check length
    if len(password) < 12:
        return (False, "Password must be at least 12 characters long")
        
    # Check for character classes
    has_uppercase = bool(re.search(r'[A-Z]', password))
    has_lowercase = bool(re.search(r'[a-z]', password))
    has_digit = bool(re.search(r'\d', password))
    has_special = bool(re.search(r'[^A-Za-z0-9]', password))
    
    character_classes = sum([has_uppercase, has_lowercase, has_digit, has_special])
    
    if character_classes < 3:
        return (False, "Password must contain at least 3 of the following: uppercase, lowercase, digits, special characters")
        
    # Check for common patterns
    if re.search(r'(.)\1\1', password):  # Three or more repeated characters
        return (False, "Password contains too many repeated characters")
        
    if re.search(r'(?:password|admin|user|login)', password, re.IGNORECASE):
        return (False, "Password contains common terms")
        
    # Check for sequential characters
    sequences = ['abcdefghijklmnopqrstuvwxyz', '0123456789', 'qwertyuiop', 'asdfghjkl', 'zxcvbnm']
    for seq in sequences:
        for i in range(len(seq) - 2):
            if seq[i:i+3].lower() in password.lower() or seq[i:i+3].lower()[::-1] in password.lower():
                return (False, "Password contains sequential characters")
    
    return (True, "")

def generate_hmac_signature(data: str, secret_key: str) -> str:
    """Generate an HMAC signature for data.
    
    Args:
        data: Data to sign
        secret_key: Secret key used for signing
        
    Returns:
        Base64 encoded HMAC signature
    """
    key_bytes = secret_key.encode('utf-8')
    data_bytes = data.encode('utf-8')
    
    signature = hmac.new(key_bytes, data_bytes, hashlib.sha256).digest()
    return base64.b64encode(signature).decode('utf-8')

def verify_hmac_signature(data: str, signature: str, secret_key: str) -> bool:
    """Verify an HMAC signature.
    
    Args:
        data: Original data that was signed
        signature: Base64 encoded signature to verify
        secret_key: Secret key used for signing
        
    Returns:
        True if signature is valid, False otherwise
    """
    expected = generate_hmac_signature(data, secret_key)
    
    # Use constant time comparison to prevent timing attacks
    return hmac.compare_digest(expected, signature)

def generate_api_key_with_expiry(user_id: str, expiry_date: datetime = None) -> Dict[str, str]:
    """Generate an API key with optional expiry.
    
    Args:
        user_id: User ID to associate with the key
        expiry_date: Optional expiry date for the key
        
    Returns:
        Dict with api_key, secret, and expiry information
    """
    api_key = generate_secure_token(16)  # shorter for usability
    api_secret = generate_secure_token(32)  # longer for security
    
    expiry_str = expiry_date.isoformat() if expiry_date else None
    
    # Create a signature that binds user_id to the key
    signature_data = f"{user_id}:{api_key}:{expiry_str or 'none'}"
    signature = generate_hmac_signature(signature_data, api_secret)
    
    return {
        "api_key": api_key,
        "api_secret": api_secret,
        "user_id": user_id,
        "expiry": expiry_str,
        "signature": signature
    }

def redact_sensitive_data(data: Dict[str, Any], sensitive_fields: List[str] = None) -> Dict[str, Any]:
    """Redact sensitive data from a dictionary.
    
    Args:
        data: Dictionary of data to redact
        sensitive_fields: List of field names to redact
        
    Returns:
        Dictionary with sensitive fields redacted
    """
    if sensitive_fields is None:
        sensitive_fields = [
            "password", "api_key", "api_secret", "token", "secret",
            "credit_card", "ssn", "social_security", "account_number"
        ]
        
    redacted = data.copy()
    
    for field in sensitive_fields:
        for key in list(redacted.keys()):
            if field.lower() in key.lower():
                if isinstance(redacted[key], str) and redacted[key]:
                    redacted[key] = "******"
            
            # Handle nested dictionaries
            if isinstance(redacted[key], dict):
                redacted[key] = redact_sensitive_data(redacted[key], sensitive_fields)
                
            # Handle lists of dictionaries
            if isinstance(redacted[key], list):
                redacted[key] = [
                    redact_sensitive_data(item, sensitive_fields) if isinstance(item, dict) else item 
                    for item in redacted[key]
                ]
    
    return redacted

def sanitize_html(text: str) -> str:
    """Remove HTML tags from a string to prevent XSS attacks.
    
    Args:
        text: Text that might contain HTML
        
    Returns:
        Sanitized text with HTML tags removed
    """
    if not text or not isinstance(text, str):
        return ""
        
    # Remove all HTML tags
    sanitized = re.sub(r'<[^>]*>', '', text)
    
    # Also escape potential script-related entities
    sanitized = sanitized.replace('&', '&amp;')
    sanitized = sanitized.replace('<', '&lt;')
    sanitized = sanitized.replace('>', '&gt;')
    sanitized = sanitized.replace('"', '&quot;')
    sanitized = sanitized.replace("'", '&#x27;')
    
    return sanitized

def validate_email(email: str) -> bool:
    """Validate an email address using a regular expression.
    
    Args:
        email: Email address to validate
        
    Returns:
        True if email is valid, False otherwise
    """
    # Use a relatively permissive pattern for real-world emails
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

# Example usage:
# password = "MySecurePassword123!"
# password_hash, salt = hash_password(password)
# stored = format_password_for_storage(password_hash, salt)
# 
# # Later, to verify:
# alg, iters, salt, hashed = parse_stored_password(stored)
# is_valid = verify_password("MySecurePassword123!", hashed, salt) 