"""通过 GitHub Git Data API 推送（绕过本机 git→github.com:443 不通）。"""

from __future__ import annotations

import base64
import json
import os
import subprocess
import urllib.error
import urllib.request
from pathlib import Path

TOKEN = os.environ["GH_TOKEN"].strip()
OWNER = "Frankie01234"
REPO = "write-protected-discrete-bottleneck"
ROOT = Path(__file__).resolve().parents[1]
API = "https://api.github.com"


def api(method: str, path: str, data=None):
    url = f"{API}{path}"
    body = None if data is None else json.dumps(data).encode("utf-8")
    req = urllib.request.Request(url, data=body, method=method)
    req.add_header("Authorization", f"Bearer {TOKEN}")
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("X-GitHub-Api-Version", "2022-11-28")
    if body is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            raw = resp.read()
            return json.loads(raw.decode()) if raw else {}
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {path} -> {e.code}: {detail}") from e


def main() -> int:
    files = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).split(b"\0")
    paths = [p.decode("utf-8") for p in files if p]
    print(f"files={len(paths)}")

    tree_items = []
    for i, rel in enumerate(paths):
        content = (ROOT / rel).read_bytes()
        try:
            text = content.decode("utf-8")
            blob = api(
                "POST",
                f"/repos/{OWNER}/{REPO}/git/blobs",
                {"content": text, "encoding": "utf-8"},
            )
        except UnicodeDecodeError:
            b64 = base64.b64encode(content).decode("ascii")
            blob = api(
                "POST",
                f"/repos/{OWNER}/{REPO}/git/blobs",
                {"content": b64, "encoding": "base64"},
            )
        tree_items.append(
            {"path": rel, "mode": "100644", "type": "blob", "sha": blob["sha"]}
        )
        if (i + 1) % 5 == 0 or i == len(paths) - 1:
            print(f"  blob {i + 1}/{len(paths)}: {rel}")

    tree = api("POST", f"/repos/{OWNER}/{REPO}/git/trees", {"tree": tree_items})
    print("tree", tree["sha"])

    parent = None
    try:
        ref = api("GET", f"/repos/{OWNER}/{REPO}/git/ref/heads/main")
        parent = ref["object"]["sha"]
        print("parent", parent)
    except Exception as e:
        print("no parent yet:", e)

    msg = (
        "实现写保护离散瓶颈子集复现并完成三组实验日志。\n\n"
        "按论文自实现 Grid World + Gumbel 负结果与三层修复"
        "（detach/Memory/DP-Means），在三组不同数据上验证结构趋势并写入完整日志。"
    )
    commit_body = {
        "message": msg,
        "tree": tree["sha"],
        "author": {
            "name": "Haochen Bian",
            "email": "145898312+Frankie01234@users.noreply.github.com",
        },
        "committer": {
            "name": "Haochen Bian",
            "email": "145898312+Frankie01234@users.noreply.github.com",
        },
    }
    if parent and parent != "0" * 40:
        # 空仓库可能没有有效 blob
        try:
            api("GET", f"/repos/{OWNER}/{REPO}/git/commits/{parent}")
            commit_body["parents"] = [parent]
        except Exception:
            print("ignore invalid parent")

    commit = api("POST", f"/repos/{OWNER}/{REPO}/git/commits", commit_body)
    print("commit", commit["sha"])

    if "parents" in commit_body:
        api(
            "PATCH",
            f"/repos/{OWNER}/{REPO}/git/refs/heads/main",
            {"sha": commit["sha"], "force": True},
        )
    else:
        try:
            api(
                "POST",
                f"/repos/{OWNER}/{REPO}/git/refs",
                {"ref": "refs/heads/main", "sha": commit["sha"]},
            )
        except RuntimeError as e:
            # 已有 ref 则更新
            if "422" in str(e) or "Reference already exists" in str(e):
                api(
                    "PATCH",
                    f"/repos/{OWNER}/{REPO}/git/refs/heads/main",
                    {"sha": commit["sha"], "force": True},
                )
            else:
                raise

    info = api("GET", f"/repos/{OWNER}/{REPO}")
    print("DONE", info.get("html_url"))
    print("default_branch", info.get("default_branch"), "size", info.get("size"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
