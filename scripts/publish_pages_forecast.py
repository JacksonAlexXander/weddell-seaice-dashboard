"""Run inference for a newer OSI SAF observation and update the published Pages batch."""

from __future__ import annotations

import argparse
import base64
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory


def github(cli, repo, method, route, data=None):
    command = [cli, "api", "--method", method, f"repos/{repo}{route}"]
    if data is not None:
        command.extend(("--input", "-"))
    result = subprocess.run(command, input=json.dumps(data) if data is not None else None,
                            text=True, check=True, stdout=subprocess.PIPE, timeout=120)
    return json.loads(result.stdout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="OWNER/REPOSITORY")
    parser.add_argument("--runtime", required=True, type=Path)
    parser.add_argument("--work", required=True, type=Path, help="Writable working directory for observations and forecast")
    parser.add_argument("--gh", default="gh", help="Authenticated GitHub CLI executable")
    parser.add_argument("--snapshot", type=Path, help="Use an independently verified recent OSI SAF batch")
    parser.add_argument("--frontend", type=Path, help="Include a verified production build's index and JS/CSS assets")
    args = parser.parse_args()

    def api(method, route, data=None):
        return github(args.gh, args.repo, method, route, data)

    ref = "/git/refs/heads/gh-pages"
    head = api("GET", ref)["object"]["sha"]
    current_file = api("GET", f"/contents/snapshots/recent.json?ref={head}")
    current = json.loads(base64.b64decode(current_file["content"]))
    with TemporaryDirectory(prefix="publish-", dir=args.work) as output:
        subprocess.run(export_command(args, current["latest_at"], output), check=True, timeout=480)
        directory = Path(output)
        next_file = directory / "recent.json"
        entries = []
        if next_file.is_file():
            new = json.loads(next_file.read_text())
            if datetime.fromisoformat(new["latest_at"]) <= datetime.fromisoformat(current["latest_at"]):
                raise ValueError("Refusing to overwrite the published batch with older observations")
            key = new["fields_file"]
            if not re.fullmatch(r"fields-\d{4}-\d{2}-\d{2}\.json", key):
                raise ValueError("Invalid forecast filename")
            entries.extend(entry(f"snapshots/{name}", directory / name) for name in (key, "recent.json"))
            keep = {key, current["fields_file"]}
            snapshot_dir = api("GET", f"/contents/snapshots?ref={head}")
            entries.extend({"path": item["path"], "mode": "100644", "type": "blob", "sha": None}
                           for item in snapshot_dir if re.fullmatch(r"fields-\d{4}-\d{2}-\d{2}\.json", item["name"])
                           and item["name"] not in keep)
        if args.frontend:
            entries.append(entry("index.html", args.frontend / "index.html"))
            entries.extend(entry(f"assets/{file.name}", file) for file in (args.frontend / "assets").iterdir()
                           if file.suffix in (".js", ".css") and not file.name.startswith("."))
        if not entries:
            return
        if api("GET", ref)["object"]["sha"] != head:
            raise SystemExit("Pages branch advanced during inference; retry with the current batch.")
        parent = api("GET", f"/git/commits/{head}")
        tree = api("POST", "/git/trees", {"base_tree": parent["tree"]["sha"], "tree": entries})
        message = f"Publish OSI SAF forecast from {new['latest_at']}" if next_file.is_file() else "Update forecast status display"
        revision = api("POST", "/git/commits", {"tree": tree["sha"], "parents": [head], "message": message})
        api("PATCH", ref, {"sha": revision["sha"], "force": False})
        print(f"Published {revision['sha']}", flush=True)


def export_command(args, after, output):
    exporter = Path(__file__).with_name("export_pages.py")
    command = [sys.executable, str(exporter), "--runtime", str(args.runtime), "--out", output, "--after", after]
    if args.snapshot:
        return command + ["--snapshot", str(args.snapshot)]
    return command + ["--refresh", "--recent-cache", str(args.work / "latest.npz")]

def entry(path, file):
    return {"path": path, "mode": "100644", "type": "blob", "content": file.read_text(encoding="utf-8")}


if __name__ == "__main__":
    main()
