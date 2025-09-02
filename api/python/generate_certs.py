#!/usr/bin/env python3
"""
Generate self-signed SSL certificates for development use.
DO NOT use these certificates in production!
"""

import os
from OpenSSL import crypto
from datetime import datetime, timedelta
import ipaddress
import socket

def get_local_ip():
    """Get local IP address to include in certificate"""
    try:
        # Create a socket connection to an external server to get local IP
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.connect(("8.8.8.8", 80))
        ip_address = sock.getsockname()[0]
        sock.close()
        return ip_address
    except Exception:
        return "127.0.0.1"  # Fallback to localhost

def generate_self_signed_cert(cert_file="ssl/cert.pem", key_file="ssl/key.pem"):
    """
    Generate a self-signed certificate and private key.
    
    Args:
        cert_file: Path to save the certificate
        key_file: Path to save the private key
    """
    # Create SSL directory if it doesn't exist
    os.makedirs(os.path.dirname(cert_file), exist_ok=True)
    
    # Get hostname and local IP for certificate
    hostname = socket.gethostname()
    ip_address = get_local_ip()
    
    # Create a key pair
    key = crypto.PKey()
    key.generate_key(crypto.TYPE_RSA, 4096)
    
    # Create a self-signed certificate
    cert = crypto.X509()
    cert.get_subject().CN = hostname
    cert.set_serial_number(1000)
    cert.gmtime_adj_notBefore(0)
    cert.gmtime_adj_notAfter(365 * 24 * 60 * 60)  # 1 year validity
    cert.set_issuer(cert.get_subject())
    cert.set_pubkey(key)
    
    # Add Subject Alternative Names (SANs)
    sans = [
        f"DNS:{hostname}",
        f"DNS:localhost",
        f"IP:{ip_address}",
        f"IP:127.0.0.1"
    ]
    san_extension = crypto.X509Extension(
        b"subjectAltName", 
        False, 
        ", ".join(sans).encode()
    )
    cert.add_extensions([san_extension])
    
    # Sign the certificate with the private key
    cert.sign(key, 'sha256')
    
    # Write the certificate and private key to files
    with open(cert_file, "wb") as f:
        f.write(crypto.dump_certificate(crypto.FILETYPE_PEM, cert))
    
    with open(key_file, "wb") as f:
        f.write(crypto.dump_privatekey(crypto.FILETYPE_PEM, key))
    
    print(f"Generated self-signed certificate: {cert_file}")
    print(f"Generated private key: {key_file}")
    print("\nIMPORTANT: This certificate is for development use only!")
    print("          Do not use in production environments.")

if __name__ == "__main__":
    generate_self_signed_cert() 