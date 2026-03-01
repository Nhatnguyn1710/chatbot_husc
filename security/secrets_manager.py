"""
🔐 SECRETS MANAGER MODULE
==========================
Module quản lý bảo mật API keys cho HUSC RAG Chatbot.

Features:
- Mã hóa AES-256 cho API keys
- Validate format API keys tự động

"""

import os
import re
import base64
import hashlib
import secrets
import json
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache

# ==============================================================================
# OPTIONAL: Cryptography for AES encryption
# ==============================================================================
try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    Fernet = None  # type: ignore


# ==============================================================================
# CONFIGURATION
# ==============================================================================

@dataclass
class SecretsConfig:
    """Configuration for Secrets Manager."""
    # Environment
    env: str = field(default_factory=lambda: os.getenv("APP_ENV", "development"))
    
    # Encryption settings
    encryption_enabled: bool = field(default_factory=lambda: os.getenv("ENCRYPT_SECRETS", "false").lower() in ("1", "true", "yes"))
    master_password_env: str = "MASTER_PASSWORD"
    salt_length: int = 16
    iterations: int = 100000
    
    # API Key validation patterns
    gemini_key_pattern: str = r"^AIza[A-Za-z0-9_-]{35,}$"
    
    # Secrets file paths
    secrets_dir: str = field(default_factory=lambda: os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "secrets"
    ))
    encrypted_file: str = "secrets.enc"
    
    # Key rotation
    key_max_age_days: int = 90  # Recommend rotating keys after this period


# ==============================================================================
# SECRETS MANAGER
# ==============================================================================

class SecretsManager:
    """
    Quản lý secrets và API keys một cách bảo mật.
    
    Usage:
        secrets = SecretsManager()
        api_key = secrets.get_gemini_key()
        
        # Validate key
        if secrets.validate_gemini_key(api_key):
            print("Key is valid!")
    """
    
    _instance: Optional["SecretsManager"] = None
    _lock = None
    
    def __new__(cls, config: Optional[SecretsConfig] = None):
        """Singleton pattern để đảm bảo chỉ có một instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, config: Optional[SecretsConfig] = None):
        if self._initialized:
            return
            
        self.config = config or SecretsConfig()
        self._fernet: Optional[Any] = None
        self._cached_secrets: Dict[str, str] = {}
        self._last_load_time: Optional[datetime] = None
        self._initialized = True
        
        # Initialize encryption if enabled and available
        if self.config.encryption_enabled and CRYPTO_AVAILABLE:
            self._init_encryption()
    
    def _init_encryption(self) -> bool:
        """Initialize Fernet encryption with master password."""
        if not CRYPTO_AVAILABLE:
            print("⚠️ Cryptography library not installed. Encryption disabled.")
            return False
            
        master_password = os.getenv(self.config.master_password_env)
        if not master_password:
            print(f"⚠️ {self.config.master_password_env} not set. Encryption disabled.")
            return False
        
        try:
            # Derive encryption key from master password
            salt = self._get_or_create_salt()
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=self.config.iterations,
            )
            key = base64.urlsafe_b64encode(kdf.derive(master_password.encode()))
            self._fernet = Fernet(key)
            return True
        except Exception as e:
            print(f"❌ Failed to initialize encryption: {e}")
            return False
    
    def _get_or_create_salt(self) -> bytes:
        """Get or create salt for key derivation."""
        salt_file = os.path.join(self.config.secrets_dir, ".salt")
        
        if os.path.exists(salt_file):
            with open(salt_file, "rb") as f:
                return f.read()
        
        # Create new salt
        os.makedirs(self.config.secrets_dir, exist_ok=True)
        salt = secrets.token_bytes(self.config.salt_length)
        
        with open(salt_file, "wb") as f:
            f.write(salt)
        
        # Set restrictive permissions (owner only)
        try:
            os.chmod(salt_file, 0o600)
        except:
            pass
            
        return salt
    
    # ==========================================================================
    # API KEY RETRIEVAL
    # ==========================================================================
    
    def get_gemini_key(self, validate: bool = True) -> str:
        """
        Lấy Gemini API key.
        
        Args:
            validate: Có validate format key không (default: True)
        
        Returns:
            API key string
            
        Raises:
            ValueError: Nếu key không hợp lệ hoặc không tìm thấy
        """
        # Try encrypted storage first
        if self.config.encryption_enabled and self._fernet:
            key = self._get_encrypted_secret("GEMINI_API_KEY")
            if key:
                if validate and not self.validate_gemini_key(key):
                    raise ValueError("❌ Gemini API key format không hợp lệ")
                return key
        
        # Fallback to environment variable
        key = os.getenv("GEMINI_API_KEY")
        if not key:
            raise ValueError("❌ Thiếu GEMINI_API_KEY trong .env hoặc encrypted storage")
        
        if validate and not self.validate_gemini_key(key):
            raise ValueError("❌ Gemini API key format không hợp lệ (phải bắt đầu bằng 'AIza')")
        
        return key
    
    def get_secret(self, key_name: str, default: Optional[str] = None) -> Optional[str]:
        """
        Lấy secret theo tên.
        
        Args:
            key_name: Tên của secret
            default: Giá trị mặc định nếu không tìm thấy
            
        Returns:
            Secret value hoặc default
        """
        # Try encrypted storage first
        if self.config.encryption_enabled and self._fernet:
            value = self._get_encrypted_secret(key_name)
            if value:
                return value
        
        # Fallback to environment variable
        return os.getenv(key_name, default)
    
    def _get_encrypted_secret(self, key_name: str) -> Optional[str]:
        """Get secret from encrypted storage."""
        if not self._fernet:
            return None
            
        encrypted_file = os.path.join(
            self.config.secrets_dir, 
            self.config.encrypted_file
        )
        
        if not os.path.exists(encrypted_file):
            return None
        
        try:
            with open(encrypted_file, "rb") as f:
                encrypted_data = f.read()
            
            decrypted = self._fernet.decrypt(encrypted_data)
            secrets_dict = json.loads(decrypted.decode())
            return secrets_dict.get(key_name)
        except Exception as e:
            print(f"⚠️ Failed to read encrypted secret: {e}")
            return None
    
    # ==========================================================================
    # API KEY VALIDATION
    # ==========================================================================
    
    def validate_gemini_key(self, api_key: str) -> bool:
        """
        Validate Gemini API key format.
        
        Args:
            api_key: API key to validate
            
        Returns:
            True nếu format hợp lệ
        """
        if not api_key:
            return False
        
        # Check format
        if not re.match(self.config.gemini_key_pattern, api_key):
            return False
        
        # Check minimum length
        if len(api_key) < 39:
            return False
        
        return True
    
    def validate_all_keys(self) -> Dict[str, bool]:
        """
        Validate tất cả API keys.
        
        Returns:
            Dict với key name và validation status
        """
        results = {}
        
        # Validate Gemini key
        try:
            gemini_key = self.get_gemini_key(validate=False)
            results["GEMINI_API_KEY"] = self.validate_gemini_key(gemini_key)
        except ValueError:
            results["GEMINI_API_KEY"] = False
        
        return results
    
    # ==========================================================================
    # ENCRYPTION OPERATIONS
    # ==========================================================================
    
    def encrypt_and_store(self, secrets_dict: Dict[str, str]) -> bool:
        """
        Encrypt và lưu secrets vào file.
        
        Args:
            secrets_dict: Dict chứa secrets cần lưu
            
        Returns:
            True nếu thành công
        """
        if not CRYPTO_AVAILABLE:
            print("❌ Cryptography library not installed")
            return False
            
        if not self._fernet:
            if not self._init_encryption():
                return False
        
        try:
            # Create secrets directory
            os.makedirs(self.config.secrets_dir, exist_ok=True)
            
            # Encrypt
            json_data = json.dumps(secrets_dict, indent=2)
            encrypted = self._fernet.encrypt(json_data.encode())
            
            # Save
            encrypted_file = os.path.join(
                self.config.secrets_dir,
                self.config.encrypted_file
            )
            
            with open(encrypted_file, "wb") as f:
                f.write(encrypted)
            
            # Set restrictive permissions
            try:
                os.chmod(encrypted_file, 0o600)
            except:
                pass
            
            print(f"✅ Secrets encrypted and saved to {encrypted_file}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to encrypt secrets: {e}")
            return False
    
    def decrypt_secrets(self) -> Optional[Dict[str, str]]:
        """
        Decrypt và đọc tất cả secrets.
        
        Returns:
            Dict chứa secrets hoặc None
        """
        if not self._fernet:
            return None
            
        encrypted_file = os.path.join(
            self.config.secrets_dir,
            self.config.encrypted_file
        )
        
        if not os.path.exists(encrypted_file):
            return None
        
        try:
            with open(encrypted_file, "rb") as f:
                encrypted_data = f.read()
            
            decrypted = self._fernet.decrypt(encrypted_data)
            return json.loads(decrypted.decode())
        except Exception as e:
            print(f"❌ Failed to decrypt secrets: {e}")
            return None
    
    # ==========================================================================
    # SECURITY UTILITIES
    # ==========================================================================
    
    def mask_key(self, key: str, visible_chars: int = 8) -> str:
        """
        Mask API key để hiển thị an toàn.
        
        Args:
            key: API key cần mask
            visible_chars: Số ký tự hiển thị ở đầu và cuối
            
        Returns:
            Masked key string
        """
        if not key or len(key) <= visible_chars * 2:
            return "****"
        
        return f"{key[:visible_chars]}...{key[-visible_chars:]}"
    
    def generate_secure_token(self, length: int = 32) -> str:
        """
        Generate secure random token.
        
        Args:
            length: Độ dài token (bytes)
            
        Returns:
            Secure token string (base64 encoded)
        """
        return secrets.token_urlsafe(length)
    
    def hash_for_comparison(self, value: str) -> str:
        """
        Hash value để so sánh an toàn.
        
        Args:
            value: String cần hash
            
        Returns:
            SHA256 hash
        """
        return hashlib.sha256(value.encode()).hexdigest()
    
    # ==========================================================================
    # ENVIRONMENT HELPERS
    # ==========================================================================
    
    @property
    def is_production(self) -> bool:
        """Check if running in production."""
        return self.config.env.lower() in ("production", "prod")
    
    @property
    def is_development(self) -> bool:
        """Check if running in development."""
        return self.config.env.lower() in ("development", "dev")
    
    def get_environment_info(self) -> Dict[str, Any]:
        """
        Get environment information.
        
        Returns:
            Dict với environment details
        """
        return {
            "environment": self.config.env,
            "encryption_enabled": self.config.encryption_enabled,
            "encryption_available": CRYPTO_AVAILABLE,
            "secrets_dir_exists": os.path.exists(self.config.secrets_dir),
            "gemini_key_set": bool(os.getenv("GEMINI_API_KEY")),
            "master_password_set": bool(os.getenv(self.config.master_password_env)),
        }
    
    # ==========================================================================
    # KEY ROTATION HELPERS
    # ==========================================================================
    
    def check_key_age(self) -> Dict[str, Any]:
        """
        Check age of stored keys.
        
        Returns:
            Dict với key age information
        """
        result = {
            "needs_rotation": False,
            "keys": {}
        }
        
        # Check encrypted file modification time
        encrypted_file = os.path.join(
            self.config.secrets_dir,
            self.config.encrypted_file
        )
        
        if os.path.exists(encrypted_file):
            mtime = os.path.getmtime(encrypted_file)
            age_days = (datetime.now().timestamp() - mtime) / (24 * 3600)
            
            result["keys"]["encrypted_store"] = {
                "age_days": int(age_days),
                "last_modified": datetime.fromtimestamp(mtime).isoformat(),
                "needs_rotation": age_days > self.config.key_max_age_days
            }
            
            if age_days > self.config.key_max_age_days:
                result["needs_rotation"] = True
        
        return result


# ==============================================================================
# SINGLETON ACCESSOR
# ==============================================================================

_secrets_manager: Optional[SecretsManager] = None


def get_secrets_manager(config: Optional[SecretsConfig] = None) -> SecretsManager:
    """
    Get Secrets Manager singleton instance.
    
    Args:
        config: Optional configuration
        
    Returns:
        SecretsManager instance
    """
    global _secrets_manager
    
    if _secrets_manager is None:
        _secrets_manager = SecretsManager(config)
    
    return _secrets_manager


def reset_secrets_manager() -> None:
    """Reset singleton instance (for testing)."""
    global _secrets_manager
    _secrets_manager = None
    SecretsManager._instance = None


# ==============================================================================
# CLI UTILITIES
# ==============================================================================

def setup_initial_secrets() -> None:
    """
    Interactive setup for initial secrets.
    Chạy: python secrets_manager.py setup
    """
    print("\n🔐 HUSC RAG Chatbot - Secrets Setup")
    print("=" * 50)
    
    # Get Gemini API key
    print("\n1️⃣ Gemini API Key")
    print("   Lấy key tại: https://makersuite.google.com/app/apikey")
    api_key = input("   Nhập Gemini API key: ").strip()
    
    if not api_key:
        print("❌ API key không được để trống!")
        return
    
    # Validate
    secrets = get_secrets_manager()
    if not secrets.validate_gemini_key(api_key):
        print("⚠️ API key format không hợp lệ (phải bắt đầu bằng 'AIza')")
        confirm = input("   Vẫn tiếp tục? (y/n): ").strip().lower()
        if confirm != 'y':
            return
    
    # Option to encrypt
    print("\n2️⃣ Encryption")
    encrypt = input("   Mã hóa secrets? (y/n): ").strip().lower() == 'y'
    
    if encrypt:
        if not CRYPTO_AVAILABLE:
            print("❌ Cài đặt cryptography: pip install cryptography")
            return
        
        master_password = input("   Nhập master password: ").strip()
        if len(master_password) < 8:
            print("❌ Master password phải >= 8 ký tự!")
            return
        
        os.environ["MASTER_PASSWORD"] = master_password
        os.environ["ENCRYPT_SECRETS"] = "true"
        
        # Re-init with encryption
        reset_secrets_manager()
        secrets = get_secrets_manager()
        
        # Store encrypted
        secrets.encrypt_and_store({"GEMINI_API_KEY": api_key})
        
        print(f"\n✅ Setup hoàn tất!")
        print(f"   - Secrets đã được mã hóa trong thư mục 'secrets/'")
        print(f"   - Thêm vào .env: MASTER_PASSWORD={master_password}")
        print(f"   - Thêm vào .env: ENCRYPT_SECRETS=true")
    else:
        # Just save to .env example
        print(f"\n✅ API Key đã được validate!")
        print(f"   - Thêm vào .env: GEMINI_API_KEY={api_key}")
    
    print("\n⚠️ QUAN TRỌNG: KHÔNG commit file .env và thư mục secrets/ lên Git!")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "setup":
        setup_initial_secrets()
    else:
        # Test mode
        print("\n🔐 Secrets Manager - Test Mode")
        print("=" * 50)
        
        secrets = get_secrets_manager()
        
        print("\n📋 Environment Info:")
        info = secrets.get_environment_info()
        for k, v in info.items():
            print(f"   {k}: {v}")
        
        print("\n🔑 Key Validation:")
        validation = secrets.validate_all_keys()
        for k, v in validation.items():
            status = "✅" if v else "❌"
            print(f"   {k}: {status}")
        
        print("\n   Chạy 'python secrets_manager.py setup' để cấu hình secrets")
