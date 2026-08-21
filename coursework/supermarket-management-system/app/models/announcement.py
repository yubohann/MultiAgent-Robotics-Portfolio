from datetime import datetime

from app import db


class Announcement(db.Model):
    """系统公告"""
    __tablename__ = 'announcements'

    announcement_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    title = db.Column(db.String(120), nullable=False)
    content = db.Column(db.String(1000), nullable=False)
    level = db.Column(db.String(20), nullable=False, default='normal')
    target_role = db.Column(db.String(20), nullable=False, default='all')
    is_published = db.Column(db.Integer, nullable=False, default=1)
    created_by = db.Column(db.Integer, db.ForeignKey('users.user_id', ondelete='SET NULL'))
    created_at = db.Column(db.DateTime, default=datetime.now, nullable=False)
    updated_at = db.Column(db.DateTime)

    creator = db.relationship('User', foreign_keys=[created_by], lazy='joined')

    __table_args__ = (
        db.CheckConstraint("level IN ('normal', 'important')", name='ck_announcements_level'),
        db.CheckConstraint("target_role IN ('all', 'admin', 'cashier')", name='ck_announcements_target_role'),
        db.CheckConstraint('is_published IN (0, 1)', name='ck_announcements_published'),
    )
