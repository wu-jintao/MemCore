#!/usr/bin/env python3
"""Send only synthetic data to the local baseline; never read private API keys."""

import argparse
import json
import os
from urllib import error, request
from urllib.parse import urlsplit
import uuid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    args = parser.parse_args()
    parsed = urlsplit(args.base_url)
    if parsed.scheme != "http" or parsed.hostname not in ("127.0.0.1", "localhost", "::1"):
        parser.error("This synthetic demo is limited to a local loopback HTTP server")
    token = os.environ.get("MEMORY_API_TOKEN")
    if not token:
        parser.error("Set the dedicated MEMORY_API_TOKEN used by your local server")
    base = args.base_url.rstrip("/")

    def send(path, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = request.Request(base + path, data=body, headers={
            "Content-Type": "application/json", "Authorization": "Bearer " + token})
        try:
            with request.urlopen(req, timeout=30) as response:
                return json.loads(response.read().decode("utf-8"))
        except error.HTTPError as response:
            raise RuntimeError("HTTP " + str(response.code) + ": " + response.read().decode("utf-8"))

    run = "synthetic-demo/" + uuid.uuid4().hex
    payload = {"request_id": "demo-add-1", "user_id": run + "/user-1", "session_id": "session-1",
               "messages": [
                   {"role": "user", "content": "我约好周六去星河书店，同行的人是小林。",
                    "timestamp": 1735689600123},
                   {"role": "assistant", "content": "另一条虚构记录：云杉咖啡馆有蓝色桌布。"}]}
    result = send("/add", payload)
    print("Add 已提交：" + json.dumps(result, ensure_ascii=False))
    same = send("/add", payload)
    assert same == result
    print("相同请求重试：200，返回值相同。")
    found = send("/search", {"query": "星河书店同行", "user_id": payload["user_id"], "top_k": 2})
    assert found["data"] and all("id" in v and v["content"] for v in found["data"])
    print("立即 Search：\n" + json.dumps(found, ensure_ascii=False, indent=2))
    unrelated = send("/search", {"query": "星河书店同行", "user_id": run + "/user-2", "top_k": 2})
    assert unrelated == {"data": []}
    print("另一个完整 user_id 的检索结果：" + json.dumps(unrelated, ensure_ascii=False))


if __name__ == "__main__":
    main()
