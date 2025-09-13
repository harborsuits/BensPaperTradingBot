"""
Configuration Models

Pydantic models for BensBot configuration schema validation and typing.
"""

import os
from typing import List, Dict, Any, Optional, Union
from datetime import time
from enum import Enum
import re
from pathlib import Path

from pydantic import (
    BaseModel, 
    Field, 
    validator, 
    root_validator, 
    EmailStr, 
    confloat, 
    conint,
    DirectoryPath,
    FilePath
)


class Timezone(str, Enum):
    """Valid timezones for trading"""
    NEW_YORK = "America/New_York"
    CHICAGO = "America/Chicago"
    LOS_ANGELES = "America/Los_Angeles"
    LONDON = "Europe/London"
    TOKYO = "Asia/Tokyo"
    SYDNEY = "Australia/Sydney"
    UTC = "UTC"


class LogLevel(str, Enum):
    """Valid logging levels"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class ReportFrequency(str, Enum):
    """Report generation frequency"""
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


class ReportFormat(str, Enum):
    """Report file format"""
    CSV = "csv"
    JSON = "json"
    HTML = "html"
    PDF = "pdf"


class ReportDestination(str, Enum):
    """Report destination"""
    FILE = "file"
    EMAIL = "email"
    BOTH = "both"


class EmergencyCriteria(str, Enum):
    """Emergency shutdown criteria"""
    MAX_DRAWDOWN = "max_drawdown_exceeded"
    MAX_DAILY_LOSS = "max_daily_loss_exceeded"
    CONSECUTIVE_LOSSES = "consecutive_losses_exceeded" 
    API_ERROR_THRESHOLD = "api_error_threshold_exceeded"
    SYSTEM_ERROR_THRESHOLD = "system_error_threshold_exceeded"


class TradeTimeModel(BaseModel):
    """Trading hours configuration"""
    start: str = Field(..., description="Trading day start time in HH:MM format")
    end: str = Field(..., description="Trading day end time in HH:MM format") 
    timezone: Timezone = Field(..., description="Timezone for trading hours")
    
    _start_time: Optional[time] = None
    _end_time: Optional[time] = None
    
    @validator('start', 'end')
    def validate_time_format(cls, v):
        """Validate time string format (HH:MM)"""
        if not re.match(r'^([0-1]?[0-9]|2[0-3]):[0-5][0-9]$', v):
            raise ValueError(f"Invalid time format: {v}. Must be in HH:MM format")
        return v
    
    def get_start_time(self) -> time:
        """Convert start time string to datetime.time object"""
        if self._start_time is None:
            hours, minutes = map(int, self.start.split(':'))
            self._start_time = time(hour=hours, minute=minutes)
        return self._start_time
    
    def get_end_time(self) -> time:
        """Convert end time string to datetime.time object"""
        if self._end_time is None: 
            hours, minutes = map(int, self.end.split(':'))
            self._end_time = time(hour=hours, minute=minutes)
        return self._end_time


class SmtpSettings(BaseModel):
    """SMTP server settings for email notifications"""
    server: str
    port: conint(gt=0, lt=65536)
    username: str
    password: str
    tls: bool = True


class EmailAlerts(BaseModel):
    """Email alert configuration"""
    enabled: bool = False
    email: Optional[EmailStr] = None
    smtp_settings: Optional[SmtpSettings] = None
    
    @validator('smtp_settings')
    def validate_smtp_settings(cls, v, values):
        """Validate SMTP settings if email alerts are enabled"""
        if values.get('enabled', False) and not v:
            raise ValueError("SMTP settings are required when email alerts are enabled")
        return v
    
    @validator('email')
    def validate_email(cls, v, values):
        """Validate email if alerts are enabled"""
        if values.get('enabled', False) and not v:
            raise ValueError("Email address is required when email alerts are enabled")
        return v


class SmsApiSettings(BaseModel):
    """API settings for SMS service"""
    account_sid: str
    auth_token: str
    from_number: str


class SmsAlerts(BaseModel):
    """SMS alert configuration"""
    enabled: bool = False
    phone_number: Optional[str] = None
    service_provider: Optional[str] = None
    api_settings: Optional[SmsApiSettings] = None
    
    @validator('phone_number')
    def validate_phone_number(cls, v, values):
        """Validate phone number format"""
        if values.get('enabled', False):
            if not v:
                raise ValueError("Phone number is required when SMS alerts are enabled")
            if not re.match(r'^\+[1-9]\d{1,14}$', v):
                raise ValueError("Phone number must be in E.164 format (e.g., +14155552671)")
        return v
    
    @validator('service_provider')
    def validate_service_provider(cls, v, values):
        """Validate service provider"""
        if values.get('enabled', False):
            if not v:
                raise ValueError("Service provider is required when SMS alerts are enabled")
            if v not in ["twilio", "aws_sns"]:
                raise ValueError("Service provider must be 'twilio' or 'aws_sns'")
        return v
    
    @validator('api_settings')
    def validate_api_settings(cls, v, values):
        """Validate API settings if SMS alerts are enabled"""
        if values.get('enabled', False) and not v:
            raise ValueError("API settings are required when SMS alerts are enabled")
        return v


class AppAlerts(BaseModel):
    """Mobile app alert configuration"""
    enabled: bool = False
    app_token: Optional[str] = None
    device_id: Optional[str] = None
    
    @validator('app_token', 'device_id')
    def validate_app_fields(cls, v, values, field):
        """Validate app token and device ID"""
        if values.get('enabled', False) and not v:
            raise ValueError(f"{field.name} is required when app alerts are enabled")
        return v


class NotificationSettings(BaseModel):
    """Notification settings for alerts"""
    email_alerts: EmailAlerts = EmailAlerts()
    sms_alerts: SmsAlerts = SmsAlerts()
    app_alerts: AppAlerts = AppAlerts()


class ReportGeneration(BaseModel):
    """Report generation configuration"""
    enabled: bool = True
    frequency: ReportFrequency = ReportFrequency.DAILY
    format: ReportFormat = ReportFormat.HTML
    destination: ReportDestination = ReportDestination.FILE


class PerformanceTracking(BaseModel):
    """Performance tracking settings"""
    metrics_calculation_interval: conint(ge=60) = 3600
    report_generation: ReportGeneration = ReportGeneration()


class CircuitBreakers(BaseModel):
    """Circuit breaker settings for risk management"""
    max_drawdown_percent: confloat(ge=1, le=50) = 15.0
    max_daily_loss_percent: confloat(ge=0.1, le=20) = 5.0
    consecutive_loss_count: conint(ge=1, le=20) = 5


class EmergencyShutdown(BaseModel):
    """Emergency shutdown configuration"""
    enabled: bool = True
    criteria: List[EmergencyCriteria] = [
        EmergencyCriteria.MAX_DRAWDOWN,
        EmergencyCriteria.MAX_DAILY_LOSS
    ]


class SystemSafeguards(BaseModel):
    """System safeguard settings"""
    circuit_breakers: CircuitBreakers = CircuitBreakers()
    emergency_shutdown: EmergencyShutdown = EmergencyShutdown()


class ConfigModel(BaseModel):
    """Main configuration model for BensBot trading system"""
    enable_market_regime_system: bool = Field(..., description="Whether to enable the market regime detection system")
    market_regime_config_path: str = Field(..., description="Path to the market regime configuration file")
    watched_symbols: List[str] = Field(..., description="List of symbols to monitor and trade")
    data_dir: str = Field(..., description="Directory to store data files")
    trading_hours: TradeTimeModel = Field(..., description="Trading hours configuration")
    initial_capital: confloat(ge=100, le=10000000) = Field(..., description="Initial capital for trading")
    risk_per_trade: confloat(ge=0.001, le=0.1) = Field(..., description="Maximum risk per trade as a decimal")
    max_open_positions: conint(ge=1, le=100) = Field(..., description="Maximum number of open positions at any time")
    broker_config_path: str = Field("config/broker_config.json", description="Path to broker configuration file")
    market_data_config_path: str = Field("config/market_data_config.json", description="Path to market data configuration file")
    log_level: LogLevel = Field(LogLevel.INFO, description="Logging level")
    notification_settings: NotificationSettings = Field(default_factory=NotificationSettings, description="Notification settings for alerts")
    performance_tracking: PerformanceTracking = Field(default_factory=PerformanceTracking, description="Performance tracking settings")
    system_safeguards: SystemSafeguards = Field(default_factory=SystemSafeguards, description="System safeguard settings")
    
    @validator('watched_symbols')
    def validate_watched_symbols(cls, v):
        """Validate watched symbols format"""
        if not v:
            raise ValueError("At least one symbol must be specified")
        
        for symbol in v:
            if not re.match(r'^[A-Z]+$', symbol):
                raise ValueError(f"Invalid symbol: {symbol}. Must contain only uppercase letters")
        
        return v
    
    @validator('market_regime_config_path', 'broker_config_path', 'market_data_config_path')
    def validate_path_format(cls, v):
        """Validate path format"""
        if not re.match(r'^[\w./\-]+$', v):
            raise ValueError(f"Invalid path format: {v}")
        return v
    
    @root_validator
    def check_path_exists(cls, values):
        """Check if required files and directories exist"""
        # Check data_dir exists or can be created
        data_dir = values.get('data_dir')
        if data_dir and not os.path.exists(data_dir):
            try:
                os.makedirs(data_dir, exist_ok=True)
            except Exception as e:
                raise ValueError(f"Cannot create data directory: {data_dir} - {str(e)}")
        
        # Check config paths
        for path_key in ['market_regime_config_path', 'broker_config_path', 'market_data_config_path']:
            if path_key in values and values[path_key]:
                path = values[path_key]
                # If file doesn't exist, we'll just warn (not error) since it might be created later
                if not os.path.exists(path):
                    print(f"Warning: Config file does not exist: {path}")
        
        return values
    
    class Config:
        """Pydantic model configuration"""
        validate_assignment = True
        validate_all = True
        extra = "forbid"
        json_encoders = {
            time: lambda t: t.strftime("%H:%M")
        }


def create_default_config() -> ConfigModel:
    """Create a default configuration model"""
    return ConfigModel(
        enable_market_regime_system=True,
        market_regime_config_path="config/market_regime_config.json",
        watched_symbols=["AAPL", "MSFT", "GOOGL", "AMZN", "SPY", "QQQ"],
        data_dir="data",
        trading_hours={
            "start": "09:30",
            "end": "16:00",
            "timezone": "America/New_York"
        },
        initial_capital=10000,
        risk_per_trade=0.02,
        max_open_positions=5
    )
