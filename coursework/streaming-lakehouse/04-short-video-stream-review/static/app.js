// Frontend controller for the short-video review demo.
//
// 这个文件负责把 FastAPI 的 JSON API 渲染成一个可交互的网站：
// 模型选择、上传视频、运行样本流、查看状态统计、查看视频列表和事件流。
// 最重要的设计点是：轮询刷新时复用已有 <video> 节点，避免播放器被销毁后无法正常播放。

const state = {
  // 当前筛选状态只影响前端展示，不会改变后端数据。
  filter: "all",
  // processing 来自 /api/health，用于控制顶部 running/idle badge。
  processing: false,
  // 模型列表和当前模型来自 /api/models 或 /api/health。
  models: [],
  activeModel: null,
};

// 后端状态值到中文展示文本的映射。
const labels = {
  processing: "处理中",
  published: "已发布",
  review: "待复核",
  rejected: "已拒绝",
};

function $(selector) {
  // 小型选择器工具，避免在教学代码中重复 document.querySelector。
  return document.querySelector(selector);
}

function setStatus(message) {
  // 顶部上传区域的状态文字，用来反馈上传、清空、切换模型等操作。
  $("#statusText").textContent = message;
}

function escapeHtml(value) {
  // 所有进入 innerHTML 的用户/模型文本都先转义，避免标题或标签注入 HTML。
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

async function requestJson(url, options = {}) {
  // 统一封装 fetch，失败时尽量读取后端 detail，便于界面显示可理解的错误。
  const response = await fetch(url, options);
  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`;
    try {
      const payload = await response.json();
      message = payload.detail || payload.error || message;
    } catch {
      // Keep the default HTTP message.
    }
    throw new Error(message);
  }
  return response.json();
}

function renderStats(stats) {
  // 渲染四个统计卡片：全部、发布、复核、拒绝。
  $("#totalCount").textContent = stats.total ?? 0;
  $("#publishedCount").textContent = stats.published ?? 0;
  $("#reviewCount").textContent = stats.review ?? 0;
  $("#rejectedCount").textContent = stats.rejected ?? 0;
}

function renderModelSelector(payload) {
  // 用后端注册表渲染模型下拉框，保证前端候选和后端实际可选模型一致。
  state.models = payload.candidates || [];
  state.activeModel = payload.active || null;
  const select = $("#modelSelect");
  select.innerHTML = state.models
    .map((model) => {
      const selected = model.id === state.activeModel?.id ? " selected" : "";
      return `<option value="${escapeHtml(model.id)}"${selected}>${escapeHtml(model.name)}</option>`;
    })
    .join("");
  renderModelDetails(state.activeModel);
  renderModelGallery(state.models, state.activeModel);
}

function renderModelDetails(model) {
  // 展示模型定位、硬件建议、下载状态和 pull 命令，帮助同学判断本机是否能运行。
  const target = $("#modelDetails");
  if (!model) {
    target.textContent = "未加载模型配置";
    return;
  }
  const downloadState = model.downloaded ? "已下载到本机" : "未下载到本机";
  target.innerHTML = `
    <strong>${escapeHtml(model.family)}</strong>
    <span>${escapeHtml(model.recommended_for)}</span>
    <small>${escapeHtml(model.hardware)} · ${escapeHtml(model.estimated_memory_gb || "")}</small>
    <small>${escapeHtml(downloadState)} · ${escapeHtml(model.pull_command || "无需下载")} · ${escapeHtml(model.notes)}</small>
  `;
}

function renderModelGallery(models, activeModel) {
  // 直接展示所有候选模型，截图时能看到主模型、两个 4B 基线、备用模型和规则基线的完整工作量。
  const target = $("#modelGallery");
  if (!target) {
    return;
  }
  const priority = [
    "ministral-3-8b-ollama",
    "qwen3-vl-4b-ollama",
    "gemma3-4b-ollama",
    "local-baseline",
    "ministral-3-3b-ollama",
  ];
  const rank = (model) => {
    const index = priority.indexOf(model.id);
    return index === -1 ? priority.length + models.indexOf(model) : index;
  };
  const ordered = [...models].sort((a, b) => rank(a) - rank(b));
  target.innerHTML = ordered
    .map((model) => {
      const active = model.id === activeModel?.id ? " active" : "";
      const downloaded = model.downloaded ? "已下载" : "未下载";
      const modeLabel = model.mode === "local_baseline" ? "规则基线" : "视觉模型";
      return `
        <article class="model-chip${active}">
          <div>
            <strong>${escapeHtml(model.name)}</strong>
            <small>${escapeHtml(model.id)}</small>
          </div>
          <span>${escapeHtml(modeLabel)}</span>
          <span>${escapeHtml(model.memory_tier)} · ${escapeHtml(downloaded)}</span>
          <span>${escapeHtml(model.ollama_model || "no-weight")}</span>
        </article>
      `;
    })
    .join("");
}

async function loadModels() {
  // 页面初始化或切换失败后重新同步模型状态。
  const payload = await requestJson("/api/models");
  renderModelSelector(payload);
}

async function selectModel(event) {
  // 切换模型时后端会拒绝正在处理任务的请求，避免同一批视频混用模型。
  const modelId = event.target.value;
  setStatus("切换模型中");
  try {
    const payload = await requestJson("/api/models/select", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({model_id: modelId}),
    });
    state.activeModel = payload.active;
    renderModelDetails(payload.active);
    setStatus(`已切换为 ${payload.active.name}`);
  } catch (error) {
    setStatus(error.message);
    await loadModels();
  }
}

function reasonText(video) {
  // 卡片只展示前两条原因，完整原因仍保留在 /api/videos JSON 中。
  return (video.reasons || [])
    .map((reason) => reason.message)
    .slice(0, 2)
    .join(" ");
}

function metricText(video) {
  // 把关键指标压缩成一行，便于同学在截图中看到模型后端和基础视频信号。
  const metrics = video.metrics || {};
  const brightness = metrics.brightness?.avg ?? 0;
  const motion = metrics.motion?.avg ?? 0;
  const duration = metrics.duration_sec ?? 0;
  const model = metrics.model?.selected_name || "未记录模型";
  const backend = metrics.model?.backend || "unknown";
  return `${duration.toFixed(1)}s · 亮度 ${brightness} · 运动 ${motion} · ${model} / ${backend}`;
}

function isUnderstanding(video) {
  // processing 或 backend=pending 都表示后台理解还没完成。
  return video.status === "processing" || video.metrics?.model?.backend === "pending";
}

function renderCaption(video) {
  // 摘要未生成时只在文字区域显示骨架屏，不遮挡真实视频播放器。
  if (video.caption) {
    return `<p class="caption">${escapeHtml(video.caption)}</p>`;
  }
  if (!isUnderstanding(video)) {
    return `<p class="caption muted-empty">暂无摘要</p>`;
  }
  return `
    <p class="caption loading-copy" aria-label="正在理解视频内容">
      <span class="skeleton-line wide"></span>
      <span class="skeleton-line"></span>
    </p>
  `;
}

function renderTags(video) {
  // 标签未生成时只显示 pill 形骨架，保持卡片高度稳定。
  if ((video.tags || []).length) {
    return (video.tags || []).map((tag) => `<span>${escapeHtml(tag)}</span>`).join("");
  }
  if (!isUnderstanding(video)) {
    return `<span class="muted-pill">暂无标签</span>`;
  }
  return `
    <span class="skeleton-pill"></span>
    <span class="skeleton-pill short"></span>
    <span class="skeleton-pill"></span>
  `;
}

function renderVideoBody(video) {
  // 构造卡片正文。注意这里不包含 <video> 节点，避免刷新正文时重置播放器。
  const tags = renderTags(video);
  const statusLabel = labels[video.status] || video.status;
  const pending = isUnderstanding(video);
  return `
    <div class="video-title-row">
      <h2>${escapeHtml(video.title)}</h2>
      <b>${escapeHtml(statusLabel)}</b>
    </div>
    ${renderCaption(video)}
    <div class="tags ${pending ? "loading-tags" : ""}">${tags}</div>
    <div class="meta">
      <span>${pending ? "审核中" : `风险 ${escapeHtml(video.risk_score)}`}</span>
      <span>${escapeHtml(metricText(video))}</span>
    </div>
    <p class="reason">${escapeHtml(reasonText(video))}</p>
  `;
}

function createVideoCard(video) {
  // 创建一次卡片和播放器；后续轮询只调用 updateVideoCard 更新状态和文本。
  const card = document.createElement("article");
  card.dataset.videoId = video.id;
  card.innerHTML = `
    <video controls preload="metadata"></video>
    <div class="video-body"></div>
  `;
  updateVideoCard(card, video);
  return card;
}

function updateVideoCard(card, video) {
  // 更新卡片状态。这里必须保持 player 元素本身不被替换，否则播放进度会不断归零。
  card.className = `video-card ${video.status}`;
  const player = card.querySelector("video");
  if (player.dataset.mediaFile !== video.media_file) {
    // 只有媒体文件真的变化时才设置 src，避免轮询造成浏览器重新加载视频。
    player.dataset.mediaFile = video.media_file;
    player.src = `/media/${encodeURIComponent(video.media_file)}`;
  }

  if (video.thumbnail_file) {
    if (player.dataset.posterFile !== video.thumbnail_file) {
      // poster 同样只在封面变化时更新，避免多余的网络请求。
      player.dataset.posterFile = video.thumbnail_file;
      player.setAttribute("poster", `/media/${encodeURIComponent(video.thumbnail_file)}`);
    }
  } else if (player.dataset.posterFile) {
    delete player.dataset.posterFile;
    player.removeAttribute("poster");
  }

  const body = card.querySelector(".video-body");
  const bodyHtml = renderVideoBody(video);
  // renderKey 记录会影响正文的字段。只有这些字段变化时才重写 innerHTML，
  // 既减少 DOM 操作，也避免浏览器频繁重排影响视频播放。
  if (body.dataset.renderKey !== JSON.stringify({
    status: video.status,
    risk_score: video.risk_score,
    caption: video.caption,
    tags: video.tags || [],
    reasons: video.reasons || [],
    metricsModel: video.metrics?.model || {},
  })) {
    body.innerHTML = bodyHtml;
    body.dataset.renderKey = JSON.stringify({
      status: video.status,
      risk_score: video.risk_score,
      caption: video.caption,
      tags: video.tags || [],
      reasons: video.reasons || [],
      metricsModel: video.metrics?.model || {},
    });
  }
}

function renderVideos(videos) {
  // 根据当前筛选条件渲染视频列表，并复用已有卡片来保持播放器稳定。
  const grid = $("#videoGrid");
  const filtered =
    state.filter === "all" ? videos : videos.filter((video) => video.status === state.filter);

  if (filtered.length === 0) {
    grid.innerHTML = `<div class="empty">暂无视频</div>`;
    return;
  }

  grid.querySelectorAll(".empty").forEach((node) => node.remove());
  const visibleIds = new Set(filtered.map((video) => video.id));
  const cardsById = new Map();
  grid.querySelectorAll(".video-card").forEach((card) => {
    // 不在当前筛选范围的卡片直接移除；仍可通过切回筛选重新创建。
    cardsById.set(card.dataset.videoId, card);
    if (!visibleIds.has(card.dataset.videoId)) {
      card.remove();
    }
  });

  filtered.forEach((video, index) => {
    let card = cardsById.get(video.id);
    if (!card) {
      card = createVideoCard(video);
    } else {
      updateVideoCard(card, video);
    }
    if (grid.children[index] !== card) {
      // 保持后端排序，同时移动已有节点而不是重建节点。
      grid.insertBefore(card, grid.children[index] || null);
    }
  });
}

function renderEvents(events) {
  // 渲染最近事件流，让同学能观察 ingest/queued/worker/understanding 等阶段。
  const list = $("#eventList");
  if (!events.length) {
    list.innerHTML = `<li class="empty-event">暂无事件</li>`;
    return;
  }
  list.innerHTML = events
    .slice(0, 40)
    .map(
      (event) => `
        <li>
          <time>${new Date(event.created_at).toLocaleTimeString()}</time>
          <strong>${escapeHtml(event.stage)}</strong>
          <span>${escapeHtml(event.message)}</span>
        </li>
      `,
    )
    .join("");
}

async function refresh() {
  // 周期性并发拉取健康状态、视频列表和事件列表。
  // 三个请求互不依赖，用 Promise.all 可以减少页面等待时间。
  const [health, videoPayload, eventPayload] = await Promise.all([
    requestJson("/api/health"),
    requestJson("/api/videos"),
    requestJson("/api/events"),
  ]);
  state.processing = health.processing;
  state.activeModel = health.active_model || state.activeModel;
  $("#processingBadge").textContent = health.processing ? "running" : "idle";
  $("#processingBadge").classList.toggle("running", health.processing);
  renderStats(videoPayload.stats || {});
  renderVideos(videoPayload.videos || []);
  renderEvents(eventPayload.events || []);
}

async function runDemo() {
  // 触发内置样本流。后端只负责入队，worker 会异步完成理解和审核。
  setStatus("样本流处理中");
  try {
    await requestJson("/api/demo", { method: "POST" });
  } catch (error) {
    setStatus(error.message);
  } finally {
    await refresh();
  }
}

async function resetDemo() {
  // 清空状态前后都刷新页面，让 UI 与 SQLite 状态保持一致。
  setStatus("清空中");
  try {
    await requestJson("/api/reset", { method: "POST" });
    setStatus("已清空");
  } catch (error) {
    setStatus(error.message);
  } finally {
    await refresh();
  }
}

async function uploadVideo(event) {
  // 上传自选短视频。成功响应时视频已经进入 processing，并已写入本地队列。
  event.preventDefault();
  const input = $("#videoInput");
  if (!input.files.length) {
    setStatus("请选择视频文件");
    return;
  }
  const form = new FormData();
  // multipart/form-data 同时携带文件和标题；标题建议包含学号用于原创性检查。
  form.append("video", input.files[0]);
  form.append("title", $("#titleInput").value || input.files[0].name);
  setStatus("上传处理中");
  try {
    await requestJson("/api/upload", { method: "POST", body: form });
    $("#uploadForm").reset();
  } catch (error) {
    setStatus(error.message);
  } finally {
    await refresh();
  }
}

document.querySelectorAll(".tab").forEach((button) => {
  // 状态筛选只改变前端 filter，然后重新渲染当前后端数据。
  button.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((item) => item.classList.remove("active"));
    button.classList.add("active");
    state.filter = button.dataset.filter;
    refresh();
  });
});

// 绑定页面交互事件。
$("#runDemo").addEventListener("click", runDemo);
$("#resetDemo").addEventListener("click", resetDemo);
$("#uploadForm").addEventListener("submit", uploadVideo);
$("#modelSelect").addEventListener("change", selectModel);

// 页面启动时加载模型、渲染一次数据，然后定时轮询后台状态。
loadModels();
refresh();
setInterval(refresh, 1800);
