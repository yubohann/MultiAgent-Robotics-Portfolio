from datetime import datetime

from app import db
from app.models import Announcement, AnnouncementRead


def _format_announcement(announcement, is_read):
    return {
        'announcement_id': announcement.announcement_id,
        'title': announcement.title,
        'content': announcement.content,
        'level': announcement.level,
        'target_role': announcement.target_role,
        'is_read': is_read,
        'created_at': announcement.created_at.strftime('%Y-%m-%d %H:%M:%S') if announcement.created_at else '',
    }


def get_announcements_for_user(user_id, role, limit=10):
    """获取用户可见公告列表。"""
    limit = max(1, min(int(limit or 10), 50))

    announcements = (
        Announcement.query
        .filter(Announcement.is_published == 1)
        .filter(Announcement.target_role.in_(['all', role]))
        .order_by(Announcement.created_at.desc())
        .limit(limit)
        .all()
    )

    announcement_ids = [item.announcement_id for item in announcements]
    read_ids = set()

    if announcement_ids:
        reads = (
            AnnouncementRead.query
            .filter(AnnouncementRead.user_id == user_id)
            .filter(AnnouncementRead.announcement_id.in_(announcement_ids))
            .all()
        )
        read_ids = {item.announcement_id for item in reads}

    return [_format_announcement(item, item.announcement_id in read_ids) for item in announcements]


def get_unread_announcement_count(user_id, role):
    """获取用户未读公告数。"""
    count = (
        db.session.query(db.func.count(Announcement.announcement_id))
        .outerjoin(
            AnnouncementRead,
            db.and_(
                AnnouncementRead.announcement_id == Announcement.announcement_id,
                AnnouncementRead.user_id == user_id,
            ),
        )
        .filter(Announcement.is_published == 1)
        .filter(Announcement.target_role.in_(['all', role]))
        .filter(AnnouncementRead.read_id.is_(None))
        .scalar()
    )
    return int(count or 0)


def mark_announcement_read(user_id, role, announcement_id):
    """标记单条公告已读。"""
    announcement = Announcement.query.filter_by(announcement_id=announcement_id, is_published=1).first()
    if not announcement:
        return False, '公告不存在'

    if announcement.target_role not in ('all', role):
        return False, '无权限读取该公告'

    existing = AnnouncementRead.query.filter_by(user_id=user_id, announcement_id=announcement_id).first()
    if existing:
        return True, '公告已读'

    try:
        db.session.add(AnnouncementRead(user_id=user_id, announcement_id=announcement_id))
        db.session.commit()
        return True, '标记成功'
    except Exception as e:
        db.session.rollback()
        return False, f'标记失败：{str(e)}'


def mark_all_announcements_read(user_id, role):
    """将当前用户可见公告全部标记为已读。"""
    visible_rows = (
        db.session.query(Announcement.announcement_id)
        .filter(Announcement.is_published == 1)
        .filter(Announcement.target_role.in_(['all', role]))
        .all()
    )
    visible_ids = [row[0] for row in visible_rows]

    if not visible_ids:
        return True, '暂无可读公告', 0

    read_rows = (
        db.session.query(AnnouncementRead.announcement_id)
        .filter(AnnouncementRead.user_id == user_id)
        .filter(AnnouncementRead.announcement_id.in_(visible_ids))
        .all()
    )
    read_ids = {row[0] for row in read_rows}

    new_records = [
        AnnouncementRead(user_id=user_id, announcement_id=announcement_id)
        for announcement_id in visible_ids
        if announcement_id not in read_ids
    ]

    if not new_records:
        return True, '已全部标记为已读', 0

    try:
        db.session.add_all(new_records)
        db.session.commit()
        return True, '已全部标记为已读', len(new_records)
    except Exception as e:
        db.session.rollback()
        return False, f'操作失败：{str(e)}', 0


def get_admin_announcements(limit=100):
    """管理员查看公告列表。"""
    return (
        Announcement.query
        .order_by(Announcement.created_at.desc())
        .limit(max(1, min(int(limit or 100), 300)))
        .all()
    )


def create_announcement(title, content, created_by, level='normal', target_role='all', is_published=1):
    """创建公告。"""
    title = (title or '').strip()
    content = (content or '').strip()
    level = (level or 'normal').strip().lower()
    target_role = (target_role or 'all').strip().lower()
    publish_value = 1 if str(is_published) in ('1', 'true', 'True') else 0

    if not title or not content:
        return False, '标题和内容不能为空'

    if len(title) > 120:
        return False, '标题长度不能超过120字符'

    if len(content) > 1000:
        return False, '内容长度不能超过1000字符'

    if level not in ('normal', 'important'):
        return False, '公告级别无效'

    if target_role not in ('all', 'admin', 'cashier'):
        return False, '目标角色无效'

    try:
        announcement = Announcement(
            title=title,
            content=content,
            level=level,
            target_role=target_role,
            is_published=publish_value,
            created_by=created_by,
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        db.session.add(announcement)
        db.session.commit()
        return True, '公告创建成功'
    except Exception as e:
        db.session.rollback()
        return False, f'公告创建失败：{str(e)}'


def set_announcement_publish_status(announcement_id, is_published):
    """设置公告发布状态。"""
    announcement = db.session.get(Announcement, announcement_id)
    if not announcement:
        return False, '公告不存在'

    try:
        announcement.is_published = 1 if str(is_published) in ('1', 'true', 'True') else 0
        announcement.updated_at = datetime.now()
        db.session.commit()
        return True, '公告状态更新成功'
    except Exception as e:
        db.session.rollback()
        return False, f'状态更新失败：{str(e)}'
