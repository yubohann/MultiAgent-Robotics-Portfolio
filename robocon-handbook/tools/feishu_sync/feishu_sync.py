#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""《ROBOCON 从入门到精通》飞书知识库同步工具。

目标结构（根节点由 .env 的 FEISHU_ROOT_NODE 指定）：
    根节点首页        内容 = 00_总目录.md
    ├─ 卷首           内容 = 00_卷首.md
    ├─ 第一篇 …       其下为 01..05 章节点
    ├─ …（共八篇）
    └─ 第八篇 …       其下为 35..37 章节点

增量逻辑：state.json 记录每个文件的内容哈希与对应节点 token。
文件没改动就跳过；改动过就原位重写（清空文档块后重新插入）；
节点被删除就重建，或按标题接入已有节点。

命令：
    python feishu_sync.py check    联网验证凭据、权限与目标节点
    python feishu_sync.py plan     本地预演，列出将新建/更新/跳过的文档
    python feishu_sync.py sync     执行同步（可反复运行）
    python feishu_sync.py status   查看本地同步状态
"""
import argparse
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

TOOL_DIR = Path(__file__).resolve().parent
BOOK_DIR = TOOL_DIR.parents[1] / "book"
STATE_PATH = TOOL_DIR / "state.json"
ENV_PATH = TOOL_DIR / ".env"

DEFAULT_BASE_URL = "https://open.feishu.cn"
RETRY_CODES = {99991400}
SAMPLE_PREFIX = "样"

ERROR_HINTS = {
    99991663: "App ID 或 App Secret 不正确，请核对 .env。",
    99991664: "App ID 或 App Secret 不正确，请核对 .env。",
    99991672: "应用权限不足：请在开放平台开通 wiki:wiki、docx:document（含文档块转换），并发布新版本。",
    1061002: "应用没有该文档的权限：请确认应用已加入知识库，权限为可编辑。",
    131006: "知识库节点不存在或应用无权访问：请核对根节点链接，并确认应用已加入该知识库。",
    131002: "知识库节点不存在或应用无权访问：请核对根节点链接，并确认应用已加入该知识库。",
}


class FeishuError(Exception):
    def __init__(self, code, msg, status=None):
        super().__init__(f"飞书接口错误 code={code} msg={msg}")
        self.code = code
        self.msg = msg
        self.status = status


def error_hint(err):
    return ERROR_HINTS.get(err.code, "")


def load_env():
    env = {}
    if ENV_PATH.exists():
        for raw in ENV_PATH.read_text(encoding="utf-8-sig").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip().strip('"').strip("'")
    for k in ("FEISHU_APP_ID", "FEISHU_APP_SECRET", "FEISHU_ROOT_NODE",
              "FEISHU_TENANT_DOMAIN", "FEISHU_BASE_URL"):
        if os.environ.get(k):
            env[k] = os.environ[k]
    return env


def require_env(env):
    missing = [k for k in ("FEISHU_APP_ID", "FEISHU_APP_SECRET", "FEISHU_ROOT_NODE") if not env.get(k)]
    if missing:
        raise SystemExit(
            "缺少配置：" + ", ".join(missing)
            + "\n请复制 .env.example 为 .env 并填写（步骤见 README.md）。"
        )


def parse_root_value(raw):
    raw = (raw or "").strip().strip('"').strip("'")
    m = re.match(r"https?://([^/]+)/wiki/([A-Za-z0-9]+)", raw)
    if m:
        return m.group(2), m.group(1)
    return raw, None


def load_state():
    if STATE_PATH.exists():
        try:
            state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        except Exception as exc:
            raise SystemExit(f"state.json 读取失败：{exc}")
        state.setdefault("files", {})
        return state
    return {"files": {}}


def save_state(state):
    state["last_sync"] = datetime.now().isoformat(timespec="seconds")
    STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def split_h1(text):
    lines = text.splitlines()
    for i, line in enumerate(lines):
        s = line.strip()
        if not s:
            continue
        if s.startswith("# "):
            title = s[2:].strip()
            body = "\n".join(lines[i + 1:]).lstrip("\n")
            return title, body
        break
    return None, text


def parse_toc(toc_text):
    sections = []
    current = None
    for line in toc_text.splitlines():
        s = line.strip()
        m = re.match(r"^## (第[^ ]+篇 .+)$", s)
        if m:
            current = {"title": m.group(1), "chapters": []}
            sections.append(current)
            continue
        m = re.match(r"^### 第 (\d+) 章 (.+)$", s)
        if m and current is not None:
            current["chapters"].append((int(m.group(1)), f"第 {m.group(1)} 章 {m.group(2)}"))
    return sections


def discover_files():
    md = sorted(BOOK_DIR.glob("*.md"))
    samples = [p for p in md if p.name.startswith(SAMPLE_PREFIX)]
    content = [p for p in md if not p.name.startswith(SAMPLE_PREFIX)]
    return content, samples


def build_targets(content_files):
    toc_path = BOOK_DIR / "00_总目录.md"
    preface_path = BOOK_DIR / "00_卷首.md"
    if not toc_path.exists():
        raise SystemExit("缺少 book/00_总目录.md")
    if not preface_path.exists():
        raise SystemExit("缺少 book/00_卷首.md")

    sections = parse_toc(toc_path.read_text(encoding="utf-8"))
    if not sections:
        raise SystemExit("00_总目录.md 解析失败：没有找到“## 第X篇 …”标题。")

    targets = [
        {"key": "root", "kind": "root", "file": toc_path, "title": "总目录", "parent": None},
        {"key": "00_卷首.md", "kind": "preface", "file": preface_path, "title": "卷首", "parent": "root"},
    ]
    used = {toc_path.name, preface_path.name}
    for sec in sections:
        skey = "section::" + sec["title"]
        targets.append({"key": skey, "kind": "section", "file": None,
                        "title": sec["title"], "parent": "root"})
        for num, ctitle in sec["chapters"]:
            matches = sorted(BOOK_DIR.glob(f"{num:02d}_*.md"))
            if len(matches) != 1:
                raise SystemExit(f"第 {num} 章对应文件不唯一或缺失：{[m.name for m in matches]}")
            path = matches[0]
            used.add(path.name)
            targets.append({"key": path.name, "kind": "chapter", "file": path,
                            "title": ctitle, "parent": skey})

    extras = sorted(p.name for p in content_files if p.name not in used)
    if extras:
        raise SystemExit("以下文件未出现在总目录中：" + ", ".join(extras))

    for t in targets:
        if t["kind"] == "section":
            t["hash"] = None
            t["body"] = None
            t["wiki_title"] = t["title"]
            continue
        text = t["file"].read_text(encoding="utf-8")
        h1, body = split_h1(text)
        t["hash"] = sha256_text(text)
        t["body"] = body
        if t["kind"] == "root":
            t["wiki_title"] = "总目录"
        else:
            t["wiki_title"] = h1 or t["title"]
    return targets


def filter_targets(targets, only):
    if not only:
        return targets
    pats = [s.strip() for s in only.split(",") if s.strip()]
    kept = []
    for t in targets:
        if t["kind"] == "section":
            kept.append(t)
            continue
        hit = False
        for p in pats:
            if p == "root" and t["kind"] == "root":
                hit = True
                break
            if re.fullmatch(r"\d{1,2}", p):
                if t["kind"] == "chapter" and int(t["key"][:2]) == int(p):
                    hit = True
                    break
                continue
            hay = " ".join([t["key"], t.get("title") or "",
                            t["file"].name if t.get("file") else ""])
            if p in hay:
                hit = True
                break
        if hit:
            kept.append(t)
    return kept

class Feishu:
    def __init__(self, app_id, app_secret, base_url=DEFAULT_BASE_URL, sleep=0.25):
        self.app_id = app_id
        self.app_secret = app_secret
        self.base = (base_url or DEFAULT_BASE_URL).rstrip("/")
        self.sleep = max(0.0, float(sleep))
        self.session = requests.Session()
        self.token = None
        self._expire = 0.0

    def auth(self):
        self._token()

    def _token(self):
        if self.token and time.time() < self._expire:
            return self.token
        data = self._request(
            "POST", "/open-apis/auth/v3/tenant_access_token/internal",
            json_body={"app_id": self.app_id, "app_secret": self.app_secret}, auth=False)
        self.token = data.get("tenant_access_token")
        if not self.token:
            raise FeishuError(-1, "鉴权响应中没有 tenant_access_token")
        self._expire = time.time() + int(data.get("expire") or 7200) - 300
        return self.token

    def _request(self, method, path, params=None, json_body=None, auth=True, max_tries=4):
        url = self.base + path
        headers = {}
        if auth:
            headers["Authorization"] = "Bearer " + self._token()
        last = ""
        for attempt in range(1, max_tries + 1):
            status = None
            try:
                resp = self.session.request(method, url, params=params, json=json_body,
                                            headers=headers, timeout=40)
                status = resp.status_code
                try:
                    data = resp.json() if resp.content else {}
                except Exception:
                    data = {}
            except Exception as exc:
                last = f"网络异常：{exc}"
                if attempt < max_tries:
                    time.sleep(1.5 * attempt)
                    continue
                raise FeishuError(-1, last)
            code = data.get("code", -1)
            if code == 0:
                time.sleep(self.sleep)
                return data.get("data") or {}
            if code in RETRY_CODES or (status and status >= 500):
                last = f"code={code} msg={data.get('msg', '')}"
                if attempt < max_tries:
                    time.sleep(2.0 * attempt)
                    continue
                raise FeishuError(code, f"频率受限或服务端错误，重试 {max_tries} 次仍失败：{last}", status)
            raise FeishuError(code, data.get("msg", ""), status)
        raise FeishuError(-1, last)

    def get_node(self, token, obj_type="wiki"):
        data = self._request("GET", "/open-apis/wiki/v2/spaces/get_node",
                             params={"token": token, "obj_type": obj_type})
        return data["node"]

    def list_nodes(self, space_id, parent_token=None):
        items, page_token = [], None
        while True:
            params = {"page_size": 50}
            if parent_token:
                params["parent_node_token"] = parent_token
            if page_token:
                params["page_token"] = page_token
            data = self._request("GET", f"/open-apis/wiki/v2/spaces/{space_id}/nodes", params=params)
            items.extend(data.get("items") or [])
            if not data.get("has_more"):
                break
            page_token = data.get("page_token")
        return items

    def create_node(self, space_id, parent_token, title):
        data = self._request("POST", f"/open-apis/wiki/v2/spaces/{space_id}/nodes",
                             json_body={"obj_type": "docx", "node_type": "origin",
                                        "parent_node_token": parent_token, "title": title})
        return data["node"]

    def update_title(self, space_id, node_token, title):
        return self._request(
            "POST", f"/open-apis/wiki/v2/spaces/{space_id}/nodes/{node_token}/update_title",
            json_body={"title": title})

    def convert_md(self, md_text):
        data = self._request("POST", "/open-apis/docx/v1/documents/blocks/convert",
                             json_body={"content_type": "markdown", "content": md_text})
        blocks = data.get("blocks") or []
        first = list(data.get("first_level_block_ids") or [])
        if blocks and not first:
            first = [b.get("block_id") for b in blocks if not b.get("parent_id")]
        return blocks, first

    def list_block_children(self, doc_id, block_id):
        items, page_token = [], None
        while True:
            params = {"page_size": 500}
            if page_token:
                params["page_token"] = page_token
            data = self._request(
                "GET", f"/open-apis/docx/v1/documents/{doc_id}/blocks/{block_id}/children",
                params=params)
            items.extend(data.get("items") or [])
            if not data.get("has_more"):
                break
            page_token = data.get("page_token")
        return items

    def insert_descendant(self, doc_id, block_id, children_ids, descendants, index=0):
        return self._request(
            "POST", f"/open-apis/docx/v1/documents/{doc_id}/blocks/{block_id}/descendant",
            json_body={"children_id": children_ids, "index": index, "descendants": descendants})

    def delete_block_children_range(self, doc_id, block_id, start, end):
        return self._request(
            "DELETE",
            f"/open-apis/docx/v1/documents/{doc_id}/blocks/{block_id}/children/batch_delete",
            json_body={"start_index": start, "end_index": end})

    def clear_doc(self, doc_id):
        for _ in range(20):
            children = self.list_block_children(doc_id, doc_id)
            if not children:
                return
            self.delete_block_children_range(doc_id, doc_id, 0, len(children))
        raise FeishuError(-1, f"文档 {doc_id} 清空失败：块数过多或接口异常")

    def write_markdown(self, doc_id, md_text):
        self.clear_doc(doc_id)
        blocks, first = self.convert_md(md_text)
        if not blocks:
            return
        if len(blocks) > 1900:
            raise FeishuError(-1, f"单文档块数 {len(blocks)} 超过接口上限，请拆分该章节文件")
        self.insert_descendant(doc_id, doc_id, first, blocks)

def plan_marker(t, state, force=False):
    st = state["files"].get(t["key"])
    if t["kind"] == "section":
        return "OK" if (st and st.get("node_token")) else "NEW"
    if force:
        return "CHG" if st else "NEW"
    if st and st.get("hash") == t["hash"]:
        return "OK"
    return "CHG" if st else "NEW"


def cmd_plan(args, env, targets, state):
    root_token, domain = parse_root_value(env.get("FEISHU_ROOT_NODE"))
    domain = env.get("FEISHU_TENANT_DOMAIN") or domain
    targets = filter_targets(targets, args.only)
    print("=" * 72)
    print("同步预演（本地，不联网）")
    print(f"根节点：{root_token or '（未配置：复制 .env.example 为 .env 并填写凭证）'}")
    if root_token and domain:
        print(f"链接：https://{domain}/wiki/{root_token}")
    if args.force:
        print("模式：--force（忽略哈希，全部重写）")
    print("=" * 72)
    counts = {"NEW": 0, "CHG": 0, "OK": 0}
    for t in targets:
        m = plan_marker(t, state, args.force)
        counts[m] += 1
        indent = "    " if t["kind"] == "chapter" else ""
        label = t["wiki_title"] if t["kind"] != "root" else "首页 · 总目录（写入根节点）"
        suffix = f"（{t['file'].name}）" if t.get("file") and t["kind"] != "root" else ""
        print(f"{indent}[{m}] {label}{suffix}")
    _, samples = discover_files()
    print("-" * 72)
    print(f"合计 {len(targets)} 个目标：新建 {counts['NEW']}，更新 {counts['CHG']}，跳过 {counts['OK']}")
    if samples:
        print(f"样张文件不参与同步：{', '.join(p.name for p in samples)}")
    if args.only:
        print(f"已按 --only {args.only} 过滤（篇级节点始终处理）")
    if counts["NEW"] or counts["CHG"]:
        print("执行同步：python feishu_sync.py sync")
    return 0


def cmd_status(args, env, targets, state):
    root_token = state.get("root_node") or parse_root_value(env.get("FEISHU_ROOT_NODE"))[0]
    files = state.get("files") or {}
    print(f"state.json：{STATE_PATH}")
    print(f"上次同步：{state.get('last_sync', '从未')}")
    print(f"根节点：{root_token or '未配置'}")
    print(f"知识空间：{state.get('space_id', '未知')}")
    if not files:
        print("尚无同步记录。先看 python feishu_sync.py plan，再执行 sync。")
        return 0
    print("-" * 72)
    for t in targets:
        st = files.get(t["key"])
        if not st:
            print(f"[未同步] {t.get('wiki_title') or t['key']}")
            continue
        tok = st.get("node_token") or ""
        short_hash = (st.get("hash") or "")[:8]
        if t["kind"] == "root":
            print(f"[首页]   总目录  node={tok}  hash={short_hash}")
        elif t["kind"] == "section":
            print(f"[篇]     {st.get('title') or t['key']}  node={tok}")
        else:
            print(f"[章]     {st.get('title') or t['key']}  node={tok}  hash={short_hash}")
    return 0

def cmd_check(args, env, targets, state):
    require_env(env)
    root_token, domain = parse_root_value(env["FEISHU_ROOT_NODE"])
    domain = env.get("FEISHU_TENANT_DOMAIN") or domain
    feishu = Feishu(env["FEISHU_APP_ID"], env["FEISHU_APP_SECRET"],
                    env.get("FEISHU_BASE_URL") or DEFAULT_BASE_URL, sleep=args.sleep)
    try:
        feishu.auth()
    except FeishuError as err:
        print(f"[错误] 应用鉴权失败：{err}")
        hint = error_hint(err)
        if hint:
            print(f"       {hint}")
        return 1
    print("[OK] 应用鉴权成功")

    try:
        root = feishu.get_node(root_token)
    except FeishuError as err:
        print(f"[错误] 无法访问根节点 {root_token}：{err}")
        hint = error_hint(err)
        if hint:
            print(f"       {hint}")
        return 1
    space_id = root.get("space_id")
    print(f"[OK] 根节点可访问：{root.get('title')}（space_id={space_id}）")
    if domain:
        print(f"     链接：https://{domain}/wiki/{root_token}")

    kids = feishu.list_nodes(space_id, root_token)
    print(f"[OK] 根节点现有 {len(kids)} 个子节点")

    try:
        feishu.convert_md("权限自检")
        print("[OK] Markdown 转文档块接口可用")
    except FeishuError as err:
        print(f"[错误] Markdown 转文档块接口不可用：{err}")
        hint = error_hint(err)
        if hint:
            print(f"       {hint}")
        return 1

    files = state.get("files") or {}
    dead = []
    for key, st in files.items():
        tok = st.get("node_token")
        if not tok or key == "root":
            continue
        try:
            feishu.get_node(tok)
        except FeishuError:
            dead.append(key)
    if dead:
        print(f"[警告] {len(dead)} 个已记录节点无法访问，同步时会重建或重接：")
        for key in dead[:10]:
            print(f"       - {key}")
    else:
        print("[OK] 已记录的节点全部可访问")

    targets = filter_targets(targets, args.only)
    counts = {"NEW": 0, "CHG": 0, "OK": 0}
    for t in targets:
        counts[plan_marker(t, state, args.force)] += 1
    print("-" * 72)
    print(f"当前待办：新建 {counts['NEW']}，更新 {counts['CHG']}，跳过 {counts['OK']}")
    return 0


def process_root(feishu, env, t, state, force):
    st = state["files"].get("root")
    if st and st.get("hash") == t["hash"] and not force:
        return "OK", "首页 · 总目录（无改动）"
    root_token = parse_root_value(env["FEISHU_ROOT_NODE"])[0]
    root_info = feishu.get_node(root_token)
    doc_id = root_info["obj_token"]
    feishu.write_markdown(doc_id, t["body"])
    state["files"]["root"] = {
        "role": "root", "node_token": root_info["node_token"], "obj_token": doc_id,
        "title": "总目录", "hash": t["hash"],
    }
    return ("CHG" if st else "NEW"), "首页 · 总目录（写入根节点）"


def children_of(feishu, space_id, child_cache, parent_token):
    if parent_token not in child_cache:
        by_title = {}
        for node in feishu.list_nodes(space_id, parent_token):
            title = (node.get("title") or "").strip()
            if title and title not in by_title:
                by_title[title] = node
        child_cache[parent_token] = by_title
    return child_cache[parent_token]


def process_section(feishu, space_id, root_token, t, state, child_cache):
    st = state["files"].get(t["key"])
    if st and st.get("node_token"):
        return "OK", f"{t['wiki_title']}（已存在）"
    cache = children_of(feishu, space_id, child_cache, root_token)
    node = cache.get(t["wiki_title"])
    verb = "CHG" if node else "NEW"
    if not node:
        node = feishu.create_node(space_id, root_token, t["wiki_title"])
        cache[t["wiki_title"]] = node
    state["files"][t["key"]] = {
        "role": "section", "node_token": node["node_token"], "obj_token": node.get("obj_token"),
        "title": t["wiki_title"], "parent": "root",
    }
    action = "接入已有节点" if verb == "CHG" else "新建节点"
    return verb, f"{t['wiki_title']}（{action}）"


def process_content(feishu, space_id, root_token, t, state, child_cache, force):
    st = state["files"].get(t["key"])
    changed = force or not st or st.get("hash") != t["hash"]
    if st and st.get("node_token") and not changed:
        return "OK", f"{t['wiki_title']}（无改动）"

    parent_token = root_token if t["parent"] == "root" else state["files"][t["parent"]]["node_token"]
    node = None
    obj_token = None
    st_valid = False
    if st and st.get("node_token"):
        try:
            node = feishu.get_node(st["node_token"])
            obj_token = node["obj_token"]
            st_valid = True
        except FeishuError:
            node = None

    verb = "CHG" if st_valid else "NEW"
    if node is None:
        cache = children_of(feishu, space_id, child_cache, parent_token)
        node = cache.get(t["wiki_title"])
        if node:
            obj_token = node.get("obj_token")
            verb = "CHG"
        else:
            node = feishu.create_node(space_id, parent_token, t["wiki_title"])
            obj_token = node.get("obj_token")
            cache[t["wiki_title"]] = node

    feishu.write_markdown(obj_token, t["body"])
    if st and st.get("title") and st["title"] != t["wiki_title"]:
        feishu.update_title(space_id, node["node_token"], t["wiki_title"])
    state["files"][t["key"]] = {
        "role": t["kind"], "node_token": node["node_token"], "obj_token": obj_token,
        "title": t["wiki_title"], "hash": t["hash"], "parent": t["parent"],
    }
    action = "更新" if verb == "CHG" else "新建"
    return verb, f"{t['wiki_title']}（{action}）"

def cmd_sync(args, env, targets, state):
    require_env(env)
    root_token, domain = parse_root_value(env["FEISHU_ROOT_NODE"])
    domain = env.get("FEISHU_TENANT_DOMAIN") or domain
    feishu = Feishu(env["FEISHU_APP_ID"], env["FEISHU_APP_SECRET"],
                    env.get("FEISHU_BASE_URL") or DEFAULT_BASE_URL, sleep=args.sleep)
    try:
        feishu.auth()
        root = feishu.get_node(root_token)
    except FeishuError as err:
        print(f"[错误] 连接飞书失败：{err}")
        hint = error_hint(err)
        if hint:
            print(f"       {hint}")
        return 1

    space_id = root.get("space_id")
    state["space_id"] = space_id
    state["root_node"] = root_token
    state["files"].setdefault("root", {})

    targets = filter_targets(targets, args.only)
    child_cache = {}
    stats = {"NEW": 0, "CHG": 0, "OK": 0, "ERR": 0}
    errors = []

    print("=" * 72)
    print(f"根节点：{root.get('title')}（space_id={space_id}）")
    if domain:
        print(f"链接：https://{domain}/wiki/{root_token}")
    print("=" * 72)

    for t in targets:
        try:
            if t["kind"] == "root":
                verb, label = process_root(feishu, env, t, state, args.force)
            elif t["kind"] == "section":
                verb, label = process_section(feishu, space_id, root_token, t, state, child_cache)
            else:
                verb, label = process_content(feishu, space_id, root_token, t, state, child_cache, args.force)
        except FeishuError as err:
            verb, label = "ERR", f"{t.get('wiki_title') or t['key']}（{err}）"
            errors.append((t, err))
        stats[verb] += 1
        indent = "    " if t["kind"] == "chapter" else ""
        print(f"{indent}[{verb}] {label}")
        save_state(state)

    print("-" * 72)
    print(f"完成：新建 {stats['NEW']}，更新 {stats['CHG']}，跳过 {stats['OK']}，失败 {stats['ERR']}")
    if errors:
        print("失败明细：")
        for t, err in errors:
            hint = error_hint(err)
            print(f"  - {t.get('wiki_title') or t['key']}：{err}")
            if hint:
                print(f"    {hint}")
        print("处理后可只重跑失败项，例如：python feishu_sync.py sync --only 12,15")
        return 1
    if domain:
        print(f"在线手册：https://{domain}/wiki/{root_token}")
    return 0


def main():
    parser = argparse.ArgumentParser(
        prog="feishu_sync",
        description="把 book/ 下的书稿同步到飞书知识库（增量、可反复运行）")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--force", action="store_true", help="忽略哈希，全部重写")
    common.add_argument("--only", default=None,
                        help="只同步匹配的文件，如 01,12,root 或卷首（篇级节点始终处理）")
    common.add_argument("--sleep", type=float, default=0.25,
                        help="每次接口调用后的间隔秒数，默认 0.25")
    sub = parser.add_subparsers(dest="cmd")
    sub.add_parser("plan", parents=[common], help="本地预演（不联网）")
    sub.add_parser("sync", parents=[common], help="执行同步")
    sub.add_parser("check", parents=[common], help="联网验证凭据、权限与目标节点")
    sub.add_parser("status", help="查看本地同步状态")
    args = parser.parse_args()
    if not args.cmd:
        parser.print_help()
        return 0
    if not hasattr(args, "only"):
        args.only = None
    if not hasattr(args, "force"):
        args.force = False
    if not hasattr(args, "sleep"):
        args.sleep = 0.25

    env = load_env()
    state = load_state()
    content_files, _ = discover_files()
    targets = build_targets(content_files)

    if args.cmd == "plan":
        return cmd_plan(args, env, targets, state)
    if args.cmd == "status":
        return cmd_status(args, env, targets, state)
    if args.cmd == "check":
        return cmd_check(args, env, targets, state)
    if args.cmd == "sync":
        return cmd_sync(args, env, targets, state)
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
