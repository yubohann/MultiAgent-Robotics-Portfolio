from datetime import datetime

from app import db


class Employee(db.Model):
    """员工档案表"""
    __tablename__ = 'employees'

    employee_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    employee_no = db.Column(db.String(50), unique=True, nullable=False)
    employee_name = db.Column(db.String(80), nullable=False)
    position = db.Column(db.String(50), nullable=False)
    phone = db.Column(db.String(30), unique=True)
    work_schedule = db.Column(db.String(120))
    status = db.Column(db.String(20), default='active')
    hired_at = db.Column(db.DateTime, default=datetime.now)
    updated_at = db.Column(db.DateTime, onupdate=datetime.now)

