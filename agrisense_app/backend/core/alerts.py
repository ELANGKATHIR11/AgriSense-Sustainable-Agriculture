"""
Alert and Notification System
SMS, WhatsApp, Email alerts with automation triggers
"""
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from enum import Enum

from pydantic import BaseModel

from ..config.optimization import settings

logger = logging.getLogger(__name__)

# Optional dependencies
try:
    from twilio.rest import Client as TwilioClient
    TWILIO_AVAILABLE = True
except ImportError:
    TWILIO_AVAILABLE = False
    TwilioClient = None  # type: ignore


class AlertLevel(str, Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AlertType(str, Enum):
    """Types of alerts"""
    TEMPERATURE = "temperature"
    MOISTURE = "moisture"
    TANK_LEVEL = "tank_level"
    PEST_DETECTION = "pest_detection"
    DISEASE_DETECTION = "disease_detection"
    IRRIGATION_FAILURE = "irrigation_failure"
    DEVICE_OFFLINE = "device_offline"
    SECURITY_BREACH = "security_breach"


class Alert(BaseModel):
    """Alert model"""
    alert_id: str
    alert_type: AlertType
    level: AlertLevel
    title: str
    message: str
    device_id: Optional[str] = None
    timestamp: datetime
    data: Optional[Dict[str, Any]] = None
    actions_taken: List[str] = []


class AlertManager:
    """
    Manages alerts and notifications across multiple channels.
    Supports SMS, WhatsApp, Email with configurable thresholds.
    """
    
    def __init__(self):
        self.twilio_client = None
        self.initialize_twilio()
        self.alert_history: List[Alert] = []
        
    def initialize_twilio(self):
        """Initialize Twilio client for SMS/WhatsApp"""
        if not TWILIO_AVAILABLE:
            logger.warning("Twilio not available. SMS/WhatsApp alerts disabled.")
            return
        
        if not settings.twilio_account_sid or not settings.twilio_auth_token:
            logger.info("Twilio credentials not configured. SMS/WhatsApp alerts disabled.")
            return
        
        try:
            self.twilio_client = TwilioClient(
                settings.twilio_account_sid,
                settings.twilio_auth_token
            )
            logger.info("Twilio client initialized for SMS/WhatsApp alerts")
        except Exception as e:
            logger.error(f"Failed to initialize Twilio: {e}")
    
    async def send_alert(
        self,
        alert_type: AlertType,
        level: AlertLevel,
        title: str,
        message: str,
        device_id: Optional[str] = None,
        data: Optional[Dict[str, Any]] = None,
        recipients: Optional[List[str]] = None
    ) -> Alert:
        """
        Send alert through configured channels.
        
        Args:
            alert_type: Type of alert
            level: Severity level
            title: Alert title
            message: Alert message
            device_id: Associated device ID
            data: Additional context data
            recipients: List of phone numbers/emails (optional)
            
        Returns:
            Alert object with delivery status
        """
        import uuid
        
        alert = Alert(
            alert_id=str(uuid.uuid4()),
            alert_type=alert_type,
            level=level,
            title=title,
            message=message,
            device_id=device_id,
            timestamp=datetime.utcnow(),
            data=data
        )
        
        logger.info(
            f"Alert triggered: {alert_type.value} - {level.value}",
            extra={"custom_fields": {
                "alert_id": alert.alert_id,
                "device_id": device_id,
                "level": level.value
            }}
        )
        
        # Send via configured channels
        if settings.enable_sms_alerts and level in [AlertLevel.CRITICAL, AlertLevel.EMERGENCY]:
            await self._send_sms(alert, recipients)
        
        if settings.enable_whatsapp_alerts and level in [AlertLevel.CRITICAL, AlertLevel.EMERGENCY]:
            await self._send_whatsapp(alert, recipients)
        
        if settings.enable_email_alerts:
            await self._send_email(alert, recipients)
        
        # Store in history
        self.alert_history.append(alert)
        if len(self.alert_history) > 1000:
            self.alert_history = self.alert_history[-1000:]
        
        return alert
    
    async def _send_sms(self, alert: Alert, recipients: Optional[List[str]]):
        """Send SMS alert via Twilio"""
        if not self.twilio_client:
            logger.warning("SMS not configured, skipping")
            alert.actions_taken.append("sms_skipped_not_configured")
            return
        
        try:
            sms_body = f"[{alert.level.value.upper()}] {alert.title}\n{alert.message}"
            if alert.device_id:
                sms_body += f"\nDevice: {alert.device_id}"
            
            # Use provided recipients or default
            phone_numbers = recipients or []  # Add default numbers from config
            
            for phone in phone_numbers:
                message = self.twilio_client.messages.create(
                    body=sms_body[:160],  # SMS character limit
                    from_=settings.twilio_phone_number,
                    to=phone
                )
                logger.info(f"SMS sent: {message.sid}")
                alert.actions_taken.append(f"sms_sent_to_{phone}")
        
        except Exception as e:
            logger.error(f"Failed to send SMS: {e}")
            alert.actions_taken.append("sms_failed")
    
    async def _send_whatsapp(self, alert: Alert, recipients: Optional[List[str]]):
        """Send WhatsApp alert via Twilio"""
        if not self.twilio_client:
            logger.warning("WhatsApp not configured, skipping")
            alert.actions_taken.append("whatsapp_skipped_not_configured")
            return
        
        try:
            wa_body = f"*{alert.title}*\n\n{alert.message}"
            if alert.device_id:
                wa_body += f"\n\n📱 Device: {alert.device_id}"
            wa_body += f"\n\n⏰ {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')} UTC"
            
            phone_numbers = recipients or []
            
            for phone in phone_numbers:
                message = self.twilio_client.messages.create(
                    body=wa_body,
                    from_=f"whatsapp:{settings.twilio_phone_number}",
                    to=f"whatsapp:{phone}"
                )
                logger.info(f"WhatsApp sent: {message.sid}")
                alert.actions_taken.append(f"whatsapp_sent_to_{phone}")
        
        except Exception as e:
            logger.error(f"Failed to send WhatsApp: {e}")
            alert.actions_taken.append("whatsapp_failed")
    
    async def _send_email(self, alert: Alert, recipients: Optional[List[str]]):
        """Send email alert"""
        # Implement email sending (SMTP, SendGrid, etc.)
        logger.info(f"Email alert: {alert.title}")
        alert.actions_taken.append("email_logged")
        # TODO: Implement actual email sending
    
    def get_recent_alerts(self, limit: int = 50) -> List[Alert]:
        """Get recent alerts"""
        return self.alert_history[-limit:]


# Singleton instance
alert_manager = AlertManager()


# ===== THRESHOLD-BASED ALERT TRIGGERS =====

async def check_temperature_alert(device_id: str, temperature: float):
    """Check if temperature exceeds critical threshold"""
    if temperature >= settings.alert_temperature_critical:
        await alert_manager.send_alert(
            alert_type=AlertType.TEMPERATURE,
            level=AlertLevel.CRITICAL,
            title="Critical Temperature Alert",
            message=f"Temperature reached {temperature}°C (Critical: >{settings.alert_temperature_critical}°C)",
            device_id=device_id,
            data={"temperature": temperature}
        )


async def check_moisture_alert(device_id: str, moisture: float):
    """Check if soil moisture is critically low"""
    if moisture <= settings.alert_moisture_critical:
        await alert_manager.send_alert(
            alert_type=AlertType.MOISTURE,
            level=AlertLevel.CRITICAL,
            title="Critical Soil Moisture Alert",
            message=f"Soil moisture at {moisture}% (Critical: <{settings.alert_moisture_critical}%)",
            device_id=device_id,
            data={"moisture": moisture}
        )


async def check_tank_level_alert(tank_level: float):
    """Check if water tank level is low"""
    if tank_level <= settings.alert_tank_level_low:
        await alert_manager.send_alert(
            alert_type=AlertType.TANK_LEVEL,
            level=AlertLevel.WARNING,
            title="Low Water Tank Level",
            message=f"Water tank at {tank_level}% (Low threshold: {settings.alert_tank_level_low}%)",
            data={"tank_level": tank_level}
        )


async def trigger_emergency_irrigation(device_id: str, moisture: float, reason: str):
    """Trigger emergency irrigation and send alert"""
    if not settings.enable_auto_irrigation:
        logger.info(f"Auto-irrigation disabled. Manual action required for {device_id}")
        return
    
    await alert_manager.send_alert(
        alert_type=AlertType.IRRIGATION_FAILURE,
        level=AlertLevel.EMERGENCY,
        title="Emergency Irrigation Triggered",
        message=f"Auto-irrigation activated: {reason}. Moisture: {moisture}%",
        device_id=device_id,
        data={"moisture": moisture, "reason": reason}
    )
    
    # Trigger valve control (would integrate with IoT backend)
    logger.info(f"Emergency irrigation triggered for device {device_id}")
