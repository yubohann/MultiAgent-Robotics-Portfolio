from datetime import datetime

from app import db


class AnnouncementRead(db.Model):
    """公告已读记录"""
    __tablename__ = 'announcement_reads'

    read_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    announcement_id = db.Column(db.Integer, db.ForeignKey('announcements.announcement_id', ondelete='CASCADE'), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('users.user_id', ondelete='CASCADE'), nullable=False)
    read_at = db.Column(db.DateTime, default=datetime.now, nullable=False)

    announcement = db.relationship('Announcement', foreign_keys=[announcement_id], lazy='joined')
    user = db.relationship('User', foreign_keys=[user_id], lazy='joined')

    __table_args__ = (
        db.UniqueConstraint('announcement_id', 'user_id', name='uq_announcement_reads_announcement_user'),
    )
