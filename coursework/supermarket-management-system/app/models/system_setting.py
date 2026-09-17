from datetime import datetime

from app import db


class SystemSetting(db.Model):
    """System setting table."""
    __tablename__ = 'system_settings'

    setting_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    setting_key = db.Column(db.String(80), unique=True, nullable=False)
    setting_value = db.Column(db.String(300), nullable=False)
    description = db.Column(db.String(300))
    updated_at = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now)

