const sidebar = document.getElementById('sidebar');
const sidebarToggleBtn = document.getElementById('toggle-sidebar');
const headerToggleBtn = document.getElementById('header-toggle');
const overlay = document.getElementById('overlay');
const logoText = document.getElementById('logo-text');
const navTexts = document.querySelectorAll('.nav-text');
const navArrows = document.querySelectorAll('.nav-arrow');
const announcementBtn = document.getElementById('announcementBtn');
const announcementPanel = document.getElementById('announcementPanel');
const announcementList = document.getElementById('announcementList');
const announcementBadge = document.getElementById('announcementBadge');
const markAllReadBtn = document.getElementById('markAllReadBtn');
const themeToggleBtn = document.getElementById('themeToggleBtn');
const themeIconMoon = document.getElementById('themeIconMoon');
const themeIconSun = document.getElementById('themeIconSun');
let isCollapsed = localStorage.getItem('sidebarCollapsed') === 'true';
let isMobileOpen = false;

const applyTheme = (theme) => {
    const isDark = theme === 'dark';
    document.documentElement.classList.toggle('dark', isDark);

    if (themeIconMoon && themeIconSun) {
        themeIconMoon.classList.toggle('hidden', isDark);
        themeIconSun.classList.toggle('hidden', !isDark);
    }
    if (themeToggleBtn) {
        const title = isDark ? '切换到亮色主题' : '切换到暗色主题';
        themeToggleBtn.setAttribute('title', title);
        themeToggleBtn.setAttribute('aria-label', title);
    }
};

const resolveInitialTheme = () => {
    const storedTheme = localStorage.getItem('theme');
    if (storedTheme === 'dark' || storedTheme === 'light') {
        return storedTheme;
    }
    return document.documentElement.classList.contains('dark') ? 'dark' : 'light';
};

applyTheme(resolveInitialTheme());

if (themeToggleBtn) {
    themeToggleBtn.addEventListener('click', () => {
        const isDark = document.documentElement.classList.contains('dark');
        const nextTheme = isDark ? 'light' : 'dark';
        localStorage.setItem('theme', nextTheme);
        applyTheme(nextTheme);
    });
}

const toggleSidebar = () => {
    isCollapsed = !isCollapsed;
    localStorage.setItem('sidebarCollapsed', isCollapsed);

    if (isCollapsed) {
        sidebar.classList.remove('w-64');
        sidebar.classList.add('w-16');
        sidebarToggleBtn.classList.add('opacity-0', 'pointer-events-none');
        logoText.classList.add('opacity-0', 'translate-x-[-10px]');
        setTimeout(() => logoText.classList.add('hidden'), 200);
        navTexts.forEach((text) => {
            text.classList.add('opacity-0', 'translate-x-[-10px]');
            setTimeout(() => text.classList.add('hidden'), 200);
        });
        navArrows.forEach((arrow) => {
            arrow.classList.add('opacity-0');
            setTimeout(() => arrow.classList.add('hidden'), 200);
        });
        document.querySelectorAll('#sidebar nav a').forEach((link) => {
            link.classList.add('justify-center');
        });
    } else {
        sidebar.classList.remove('w-16');
        sidebar.classList.add('w-64');
        sidebarToggleBtn.classList.remove('opacity-0', 'pointer-events-none');
        logoText.classList.remove('hidden', 'opacity-0', 'translate-x-[-10px]');
        navTexts.forEach((text) => text.classList.remove('hidden', 'opacity-0', 'translate-x-[-10px]'));
        navArrows.forEach((arrow) => arrow.classList.remove('hidden', 'opacity-0'));
        document.querySelectorAll('#sidebar nav a').forEach((link) => {
            link.classList.remove('justify-center');
        });
    }
};

sidebarToggleBtn.addEventListener('click', toggleSidebar);
headerToggleBtn.addEventListener('click', toggleSidebar);

const toggleMobileMenu = () => {
    isMobileOpen = !isMobileOpen;
    if (isMobileOpen) {
        sidebar.classList.remove('-translate-x-full');
        overlay.classList.remove('hidden');
    } else {
        sidebar.classList.add('-translate-x-full');
        overlay.classList.add('hidden');
    }
};

overlay.addEventListener('click', () => {
    isMobileOpen = false;
    sidebar.classList.add('-translate-x-full');
    overlay.classList.add('hidden');
});

const checkMobile = () => {
    if (window.innerWidth < 1024) {
        sidebar.classList.add('-translate-x-full');
        if (isCollapsed) {
            toggleSidebar();
        }
    } else {
        sidebar.classList.remove('-translate-x-full');
        overlay.classList.add('hidden');
        isMobileOpen = false;
    }
};

if (isCollapsed && window.innerWidth >= 1024) {
    sidebar.classList.remove('w-64');
    sidebar.classList.add('w-16');
    sidebarToggleBtn.classList.add('opacity-0', 'pointer-events-none');
    logoText.classList.add('hidden', 'opacity-0', 'translate-x-[-10px]');
    navTexts.forEach((text) => {
        text.classList.add('hidden', 'opacity-0', 'translate-x-[-10px]');
    });
    navArrows.forEach((arrow) => {
        arrow.classList.add('hidden', 'opacity-0');
    });
    document.querySelectorAll('#sidebar nav a').forEach((link) => {
        link.classList.add('justify-center');
    });
}

window.addEventListener('resize', checkMobile);
checkMobile();

const escapeHtml = (value) => {
    return String(value || '')
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
};

const fetchUnreadCount = async () => {
    if (!announcementBadge) {
        return;
    }

    try {
        const response = await fetch('/api/announcements/unread-count');
        const result = await response.json();
        if (!result.success) {
            return;
        }

        const count = Number(result.unread_count || 0);
        if (count > 0) {
            announcementBadge.classList.remove('hidden');
            announcementBadge.textContent = count > 99 ? '99+' : String(count);
        } else {
            announcementBadge.classList.add('hidden');
            announcementBadge.textContent = '0';
        }
    } catch (error) {
        console.error('获取公告未读数失败:', error);
    }
};

const markAnnouncementAsRead = async (announcementId) => {
    try {
        const response = await fetch(`/api/announcements/${announcementId}/read`, {
            method: 'POST'
        });
        const result = await response.json();
        if (result.success) {
            await fetchAnnouncements();
            await fetchUnreadCount();
        }
    } catch (error) {
        console.error('标记公告已读失败:', error);
    }
};

const renderAnnouncements = (announcements) => {
    if (!announcementList) {
        return;
    }

    if (!announcements || !announcements.length) {
        announcementList.innerHTML = '<p class="px-4 py-6 text-sm text-slate-500 text-center">暂无公告消息</p>';
        return;
    }

    const html = announcements.map((item) => {
        const unreadDot = item.is_read
            ? ''
            : '<span class="w-2 h-2 bg-primary rounded-full mt-1 mr-2 flex-shrink-0"></span>';
        const levelBadge = item.level === 'important'
            ? '<span class="px-1.5 py-0.5 text-[10px] rounded bg-rose-100 text-rose-700">重要</span>'
            : '';
        const rowClass = item.is_read ? 'bg-white' : 'bg-blue-50/40';

        return `
            <button
                type="button"
                data-announcement-id="${item.announcement_id}"
                class="w-full text-left px-4 py-3 border-b border-slate-100 hover:bg-slate-50 transition-colors ${rowClass}"
            >
                <div class="flex items-start justify-between gap-2">
                    <div class="flex items-start min-w-0">
                        ${unreadDot}
                        <div class="min-w-0">
                            <p class="text-sm font-medium text-slate-800 truncate">${escapeHtml(item.title)}</p>
                            <p class="text-xs text-slate-500 mt-1 line-clamp-2">${escapeHtml(item.content)}</p>
                        </div>
                    </div>
                    ${levelBadge}
                </div>
                <p class="text-[11px] text-slate-400 mt-2">${escapeHtml(item.created_at)}</p>
            </button>
        `;
    }).join('');

    announcementList.innerHTML = html;

    announcementList.querySelectorAll('[data-announcement-id]').forEach((element) => {
        element.addEventListener('click', async () => {
            const announcementId = Number(element.getAttribute('data-announcement-id'));
            if (announcementId) {
                await markAnnouncementAsRead(announcementId);
            }
        });
    });
};

const fetchAnnouncements = async () => {
    if (!announcementList) {
        return;
    }

    try {
        const response = await fetch('/api/announcements?limit=8');
        const result = await response.json();
        if (!result.success) {
            announcementList.innerHTML = '<p class="px-4 py-6 text-sm text-red-500 text-center">加载公告失败</p>';
            return;
        }
        renderAnnouncements(result.data || []);
    } catch (error) {
        console.error('获取公告列表失败:', error);
        announcementList.innerHTML = '<p class="px-4 py-6 text-sm text-red-500 text-center">加载公告失败</p>';
    }
};

const markAllAnnouncementsAsRead = async () => {
    try {
        const response = await fetch('/api/announcements/read-all', {
            method: 'POST'
        });
        const result = await response.json();
        if (result.success) {
            await fetchAnnouncements();
            await fetchUnreadCount();
        }
    } catch (error) {
        console.error('全部标记已读失败:', error);
    }
};

if (announcementBtn && announcementPanel && announcementList && announcementBadge && markAllReadBtn) {
    announcementBtn.addEventListener('click', async (event) => {
        event.stopPropagation();
        const isHidden = announcementPanel.classList.contains('hidden');
        if (isHidden) {
            announcementPanel.classList.remove('hidden');
            await fetchAnnouncements();
            await fetchUnreadCount();
        } else {
            announcementPanel.classList.add('hidden');
        }
    });

    markAllReadBtn.addEventListener('click', async (event) => {
        event.stopPropagation();
        await markAllAnnouncementsAsRead();
    });

    document.addEventListener('click', (event) => {
        if (!event.target.closest('#announcementCenter')) {
            announcementPanel.classList.add('hidden');
        }
    });

    fetchUnreadCount();
    setInterval(fetchUnreadCount, 60000);
}
