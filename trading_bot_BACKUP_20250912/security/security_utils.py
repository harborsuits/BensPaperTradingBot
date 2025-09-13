import json
import re
import logging

logger = logging.getLogger(__name__)

def redact_sensitive_data(data, sensitive_keys=None):
    """
    Redact sensitive data from logs and outputs
    
    Args:
        data: Data object to redact
        sensitive_keys: List of sensitive key names to redact
        
    Returns:
        Redacted copy of the data
    """
    if sensitive_keys is None:
        sensitive_keys = [
            'api_key', 'apikey', 'key', 'secret', 'password', 'token',
            'credential', 'auth', 'private'
        ]
    
    # Make a deep copy to avoid modifying original
    if isinstance(data, dict):
        result = {}
        for key, value in data.items():
            # Check if key contains sensitive information
            if any(sensitive_term in key.lower() for sensitive_term in sensitive_keys):
                if isinstance(value, str):
                    if len(value) > 8:
                        result[key] = value[:4] + '****' + value[-4:]
                    else:
                        result[key] = '********'
                else:
                    result[key] = '[REDACTED]'
            else:
                # Recursive redaction for nested structures
                if isinstance(value, (dict, list)):
                    result[key] = redact_sensitive_data(value, sensitive_keys)
                else:
                    result[key] = value
        return result
    elif isinstance(data, list):
        return [redact_sensitive_data(item, sensitive_keys) for item in data]
    else:
        return data

def sanitize_input(input_str):
    """
    Sanitize input strings to prevent injection attacks
    
    Args:
        input_str: String to sanitize
        
    Returns:
        Sanitized string
    """
    if not input_str or not isinstance(input_str, str):
        return input_str
    
    # Remove potentially dangerous patterns
    sanitized = re.sub(r'<script.*?>.*?</script>', '', input_str, flags=re.DOTALL | re.IGNORECASE)
    sanitized = re.sub(r'javascript:', '', sanitized, flags=re.IGNORECASE)
    sanitized = re.sub(r'on\w+=["\'].*?["\']', '', sanitized, flags=re.IGNORECASE)
    
    return sanitized 