from datetime import datetime

from app import db


class Member(db.Model):
    """会员账户表"""
    __tablename__ = 'members'

    member_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    member_no = db.Column(db.String(50), unique=True, nullable=False)
    member_name = db.Column(db.String(80), nullable=False)
    phone = db.Column(db.String(30), unique=True)
    level = db.Column(db.String(20), default='normal')
    points = db.Column(db.Integer, default=0)
    status = db.Column(db.String(20), default='active')
    registered_at = db.Column(db.DateTime, default=datetime.now)
    updated_at = db.Column(db.DateTime, onupdate=datetime.now)

