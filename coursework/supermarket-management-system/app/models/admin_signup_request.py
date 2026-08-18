from datetime import datetime

from app import db


class AdminSignupRequest(db.Model):
    """管理员注册申请表"""
    __tablename__ = 'admin_signup_requests'

    request_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    status = db.Column(db.String(20), nullable=False, default='pending')
    reviewed_by = db.Column(db.Integer, db.ForeignKey('users.user_id', ondelete='SET NULL'))
    reviewed_at = db.Column(db.DateTime)
    reject_reason = db.Column(db.String(200))
    created_at = db.Column(db.DateTime, default=datetime.now, nullable=False)

    reviewer = db.relationship('User', foreign_keys=[reviewed_by], lazy='joined')

    __table_args__ = (
        db.CheckConstraint("status IN ('pending', 'approved', 'rejected')", name='ck_admin_signup_requests_status'),
    )
