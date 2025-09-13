"""
Security module for the marketplace API.

Provides:
- Code signing and verification
- Sandbox execution environment
- API key management
- Security checks for marketplace components
"""

import os
import uuid
import base64
import hashlib
import hmac
import json
import subprocess
import threading
import time
import logging
import tempfile
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from pathlib import Path
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)

class CodeSigner:
    """
    Signs code components using cryptographic signatures to verify authenticity.
    """
    
    def __init__(self, keys_file: str):
        """
        Initialize the code signer.
        
        Args:
            keys_file: Path to the keys file
        """
        self.keys_file = keys_file
        self._load_keys()
    
    def _load_keys(self):
        """Load signing keys from file"""
        try:
            with open(self.keys_file, 'r') as f:
                keys = json.load(f)
                self.user_id = keys.get("user_id")
                self.signing_key = keys.get("signing_key")
                
                if not self.signing_key:
                    raise ValueError("No signing key found")
        except Exception as e:
            logger.error(f"Failed to load signing keys: {e}")
            raise RuntimeError(f"Failed to load signing keys: {e}")
    
    def sign(self, file_path: str) -> str:
        """
        Sign a file using the signing key.
        
        Args:
            file_path: Path to the file to sign
            
        Returns:
            Base64-encoded signature
        """
        try:
            # Read file content
            with open(file_path, 'rb') as f:
                content = f.read()
                
            return self.sign_data(content)
        except Exception as e:
            logger.error(f"Signing failed: {e}")
            raise RuntimeError(f"Failed to sign file: {e}")
    
    def sign_data(self, data: bytes) -> str:
        """
        Sign data using the signing key.
        
        Args:
            data: Bytes to sign
            
        Returns:
            Base64-encoded signature
        """
        try:
            # Decode hex signing key
            key_bytes = bytes.fromhex(self.signing_key)
            
            # Create signature
            signature = hmac.new(key_bytes, data, hashlib.sha256).digest()
            
            # Encode as base64
            return base64.b64encode(signature).decode('utf-8')
        except Exception as e:
            logger.error(f"Data signing failed: {e}")
            raise RuntimeError(f"Failed to sign data: {e}")


class CodeVerifier:
    """
    Verifies code component signatures to ensure authenticity.
    """
    
    def __init__(self, trusted_keys_file: str):
        """
        Initialize the code verifier.
        
        Args:
            trusted_keys_file: Path to the trusted keys file
        """
        self.trusted_keys_file = trusted_keys_file
        self.trusted_publishers = {}
        self._load_trusted_keys()
    
    def _load_trusted_keys(self):
        """Load trusted keys from file"""
        try:
            with open(self.trusted_keys_file, 'r') as f:
                trusted = json.load(f)
                self.trusted_publishers = trusted.get("trusted_publishers", {})
        except Exception as e:
            logger.error(f"Failed to load trusted keys: {e}")
            # Continue with empty trusted publishers list
            pass
    
    def verify(self, file_path: str, signature: str, publisher_id: Optional[str] = None) -> bool:
        """
        Verify a file's signature.
        
        Args:
            file_path: Path to the file to verify
            signature: Base64-encoded signature to verify
            publisher_id: Optional ID of the publisher
            
        Returns:
            True if verification succeeds, False otherwise
        """
        try:
            # Read file content
            with open(file_path, 'rb') as f:
                content = f.read()
                
            return self.verify_data(content, signature, publisher_id)
        except Exception as e:
            logger.error(f"Verification failed: {e}")
            return False
    
    def verify_data(self, data: bytes, signature: str, publisher_id: Optional[str] = None) -> bool:
        """
        Verify data signature.
        
        Args:
            data: Bytes to verify
            signature: Base64-encoded signature to verify
            publisher_id: Optional ID of the publisher
            
        Returns:
            True if verification succeeds, False otherwise
        """
        try:
            # If publisher ID is provided, verify against that publisher's key
            if publisher_id and publisher_id in self.trusted_publishers:
                return self._verify_with_key(data, signature, self.trusted_publishers[publisher_id])
            
            # Otherwise, try all trusted keys
            for pub_id, key in self.trusted_publishers.items():
                if self._verify_with_key(data, signature, key):
                    return True
            
            return False
        except Exception as e:
            logger.error(f"Data verification failed: {e}")
            return False
    
    def _verify_with_key(self, data: bytes, signature: str, key: str) -> bool:
        """
        Verify data with a specific key.
        
        Args:
            data: Bytes to verify
            signature: Base64-encoded signature to verify
            key: Signing key as hex string
            
        Returns:
            True if verification succeeds, False otherwise
        """
        try:
            # Decode key and signature
            key_bytes = bytes.fromhex(key)
            sig_bytes = base64.b64decode(signature)
            
            # Compute expected signature
            expected_sig = hmac.new(key_bytes, data, hashlib.sha256).digest()
            
            # Compare in constant time to prevent timing attacks
            return hmac.compare_digest(sig_bytes, expected_sig)
        except Exception as e:
            logger.error(f"Key verification failed: {e}")
            return False


class Sandbox:
    """
    Secure sandbox environment for executing marketplace components.
    
    Provides isolation and resource limits to prevent malicious code execution.
    """
    
    def __init__(self, temp_dir: Optional[str] = None):
        """
        Initialize the sandbox.
        
        Args:
            temp_dir: Optional custom temporary directory
        """
        self.temp_dir = temp_dir or tempfile.mkdtemp(prefix="trading_sandbox_")
        self.process = None
        self.execution_id = None
        
        # Create temp directory if it doesn't exist
        os.makedirs(self.temp_dir, exist_ok=True)
        
        logger.info(f"Sandbox initialized with temp dir: {self.temp_dir}")
    
    def execute(self, component_path: str, input_data: Dict[str, Any], timeout: int = 30) -> Dict[str, Any]:
        """
        Execute a component in the sandbox.
        
        Args:
            component_path: Path to the component file
            input_data: Input data for the component
            timeout: Execution timeout in seconds
            
        Returns:
            Execution results
        """
        try:
            # Generate execution ID
            self.execution_id = str(uuid.uuid4())
            
            # Create sandbox environment
            sandbox_dir = os.path.join(self.temp_dir, self.execution_id)
            os.makedirs(sandbox_dir, exist_ok=True)
            
            # Copy component to sandbox
            sandbox_component = os.path.join(sandbox_dir, os.path.basename(component_path))
            with open(component_path, 'rb') as src, open(sandbox_component, 'wb') as dst:
                dst.write(src.read())
            
            # Create input file
            input_file = os.path.join(sandbox_dir, "input.json")
            with open(input_file, 'w') as f:
                json.dump(input_data, f)
            
            # Create output file path
            output_file = os.path.join(sandbox_dir, "output.json")
            
            # Create sandbox execution script
            sandbox_script = self._create_sandbox_script(sandbox_component, input_file, output_file)
            script_path = os.path.join(sandbox_dir, "sandbox_script.py")
            
            with open(script_path, 'w') as f:
                f.write(sandbox_script)
            
            # Execute in subprocess with timeout
            result = self._execute_sandbox_script(script_path, timeout)
            
            # Read output if process completed successfully
            if result["exit_code"] == 0 and os.path.exists(output_file):
                try:
                    with open(output_file, 'r') as f:
                        result["output"] = json.load(f)
                except json.JSONDecodeError:
                    # If output file isn't valid JSON, read as string
                    with open(output_file, 'r') as f:
                        result["output"] = f.read()
            
            return result
            
        except Exception as e:
            logger.error(f"Sandbox execution failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "execution_id": self.execution_id
            }
        finally:
            # Cleanup
            self._cleanup(sandbox_dir)
    
    def _create_sandbox_script(self, component_path: str, input_file: str, output_file: str) -> str:
        """
        Create a Python script to execute the component in a sandbox.
        
        Args:
            component_path: Path to the component file
            input_file: Path to the input data file
            output_file: Path to write output data
            
        Returns:
            Python script as string
        """
        script = """import os
import sys
import json
import importlib.util
import traceback
from io import StringIO
import resource

# Set resource limits
resource.setrlimit(resource.RLIMIT_CPU, (20, 20))  # CPU time in seconds
resource.setrlimit(resource.RLIMIT_NOFILE, (50, 50))  # Number of open files
resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024, 512 * 1024 * 1024))  # Memory limit: 512MB

# Capture stdout and stderr
original_stdout = sys.stdout
original_stderr = sys.stderr
sys.stdout = captured_stdout = StringIO()
sys.stderr = captured_stderr = StringIO()

try:
    # Load component module
    spec = importlib.util.spec_from_file_location("component", "{0}")
    component = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(component)
    
    # Load input data
    with open("{1}", "r") as f:
        input_data = json.load(f)
    
    # Execute component
    if hasattr(component, "execute"):
        result = component.execute(input_data)
    else:
        result = {{
            "error": "Component does not have an execute function"
        }}
    
    # Add stdout and stderr to result
    result["stdout"] = captured_stdout.getvalue()
    result["stderr"] = captured_stderr.getvalue()
    
    # Write result to output file
    with open("{2}", "w") as f:
        json.dump(result, f)
    
    sys.exit(0)
except Exception as e:
    # Write error to output file
    with open("{2}", "w") as f:
        error_data = {{
            "error": str(e),
            "traceback": traceback.format_exc(),
            "stdout": captured_stdout.getvalue(),
            "stderr": captured_stderr.getvalue()
        }}
        json.dump(error_data, f)
    
    sys.exit(1)
finally:
    # Restore stdout and stderr
    sys.stdout = original_stdout
    sys.stderr = original_stderr
""".format(component_path, input_file, output_file)
        
        return script
    
    def _execute_sandbox_script(self, script_path: str, timeout: int) -> Dict[str, Any]:
        """
        Execute the sandbox script in a subprocess.
        
        Args:
            script_path: Path to the sandbox script
            timeout: Execution timeout in seconds
            
        Returns:
            Execution result
        """
        result = {
            "success": False,
            "exit_code": None,
            "execution_id": self.execution_id,
            "timeout": timeout,
            "output": None,
            "error": None
        }
        
        try:
            start_time = time.time()
            
            # Execute script in subprocess
            self.process = subprocess.Popen(
                [sys.executable, script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Wait for process to complete with timeout
            stdout, stderr = self.process.communicate(timeout=timeout)
            
            # Update result
            result["exit_code"] = self.process.returncode
            result["success"] = (self.process.returncode == 0)
            result["execution_time"] = time.time() - start_time
            result["stdout"] = stdout.decode('utf-8', errors='replace')
            result["stderr"] = stderr.decode('utf-8', errors='replace')
            
            return result
            
        except subprocess.TimeoutExpired:
            # Kill process if it times out
            if self.process:
                self.process.kill()
                stdout, stderr = self.process.communicate()
                
                result["error"] = "Execution timed out"
                result["exit_code"] = -1
                result["stdout"] = stdout.decode('utf-8', errors='replace')
                result["stderr"] = stderr.decode('utf-8', errors='replace')
            
            return result
        except Exception as e:
            result["error"] = str(e)
            return result
    
    def _cleanup(self, sandbox_dir: str):
        """
        Clean up sandbox directory.
        
        Args:
            sandbox_dir: Path to sandbox directory
        """
        try:
            # For security, we don't delete the entire directory
            # but only clear sensitive files
            if os.path.exists(sandbox_dir):
                for filename in os.listdir(sandbox_dir):
                    file_path = os.path.join(sandbox_dir, filename)
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
        except Exception as e:
            logger.error(f"Sandbox cleanup failed: {e}")


class SecurityManager:
    """
    Manages security features for the marketplace API.
    
    Responsible for:
    - Code signing and verification
    - API key management
    - User identification
    """
    
    def __init__(self, keys_path: str):
        """
        Initialize the security manager.
        
        Args:
            keys_path: Path to store security keys
        """
        self.keys_path = keys_path
        self.user_keys_file = os.path.join(keys_path, "user_keys.json")
        self.trusted_keys_file = os.path.join(keys_path, "trusted_keys.json")
        
        # Create keys directory if it doesn't exist
        os.makedirs(keys_path, exist_ok=True)
        
        # Initialize user and trusted keys
        self._init_keys()
        
        # Create code verifier and signer instances
        self.code_verifier = CodeVerifier(self.trusted_keys_file)
        self.code_signer = CodeSigner(self.user_keys_file)
        
        logger.info("Security manager initialized")
    
    def _init_keys(self):
        """Initialize user and trusted keys files if they don't exist"""
        # User keys
        if not os.path.exists(self.user_keys_file):
            default_keys = {
                "user_id": str(uuid.uuid4()),
                "signing_key": self._generate_key().hex(),
                "api_key": self._generate_key().hex()
            }
            
            with open(self.user_keys_file, 'w') as f:
                json.dump(default_keys, f, indent=2)
        
        # Trusted keys
        if not os.path.exists(self.trusted_keys_file):
            default_trusted = {"trusted_publishers": {}}
            
            with open(self.trusted_keys_file, 'w') as f:
                json.dump(default_trusted, f, indent=2)
    
    def _generate_key(self, size: int = 32) -> bytes:
        """Generate a random key of specified size"""
        return os.urandom(size)
    
    def get_user_id(self) -> str:
        """
        Get the current user ID.
        
        Returns:
            User ID string
        """
        try:
            with open(self.user_keys_file, 'r') as f:
                keys = json.load(f)
                return keys.get("user_id", str(uuid.uuid4()))
        except:
            # Generate new ID if file is corrupted or missing
            user_id = str(uuid.uuid4())
            self._init_keys()
            return user_id
    
    def generate_api_key(self) -> str:
        """
        Generate a new API key for the user.
        
        Returns:
            New API key string
        """
        try:
            with open(self.user_keys_file, 'r') as f:
                keys = json.load(f)
            
            # Generate new API key
            api_key = self._generate_key().hex()
            keys["api_key"] = api_key
            
            with open(self.user_keys_file, 'w') as f:
                json.dump(keys, f, indent=2)
            
            return api_key
            
        except Exception as e:
            logger.error(f"Failed to generate API key: {e}")
            raise RuntimeError(f"API key generation failed: {e}")
    
    def sign_component(self, component_path: str) -> str:
        """
        Sign a component using the user's signing key.
        
        Args:
            component_path: Path to the component file
        
        Returns:
            Signature string
        """
        return self.code_signer.sign(component_path)
    
    def verify_component(self, component_path: str, signature: str, publisher_id: Optional[str] = None) -> bool:
        """
        Verify a component's signature.
        
        Args:
            component_path: Path to the component file
            signature: Signature to verify
            publisher_id: Optional ID of the publisher
        
        Returns:
            True if verification succeeds, False otherwise
        """
        return self.code_verifier.verify(component_path, signature, publisher_id)
    
    def verify_component_data(self, component_data: bytes, signature: str, publisher_id: Optional[str] = None) -> bool:
        """
        Verify component data's signature.
        
        Args:
            component_data: Component data
            signature: Signature to verify
            publisher_id: Optional ID of the publisher
        
        Returns:
            True if verification succeeds, False otherwise
        """
        return self.code_verifier.verify_data(component_data, signature, publisher_id)
    
    def add_trusted_publisher(self, publisher_id: str, signing_key: str) -> bool:
        """
        Add a trusted publisher.
        
        Args:
            publisher_id: ID of the publisher
            signing_key: Publisher's signing key
        
        Returns:
            True if added successfully, False otherwise
        """
        try:
            with open(self.trusted_keys_file, 'r') as f:
                trusted = json.load(f)
            
            trusted.setdefault("trusted_publishers", {})[publisher_id] = signing_key
            
            with open(self.trusted_keys_file, 'w') as f:
                json.dump(trusted, f, indent=2)
            
            # Reload code verifier
            self.code_verifier = CodeVerifier(self.trusted_keys_file)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to add trusted publisher: {e}")
            return False
    
    def remove_trusted_publisher(self, publisher_id: str) -> bool:
        """
        Remove a trusted publisher.
        
        Args:
            publisher_id: ID of the publisher to remove
        
        Returns:
            True if removed successfully, False otherwise
        """
        try:
            with open(self.trusted_keys_file, 'r') as f:
                trusted = json.load(f)
            
            if publisher_id in trusted.get("trusted_publishers", {}):
                del trusted["trusted_publishers"][publisher_id]
                
                with open(self.trusted_keys_file, 'w') as f:
                    json.dump(trusted, f, indent=2)
                
                # Reload code verifier
                self.code_verifier = CodeVerifier(self.trusted_keys_file)
                
                return True
            else:
                return False
            
        except Exception as e:
            logger.error(f"Failed to remove trusted publisher: {e}")
            return False
