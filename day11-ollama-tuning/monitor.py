"""Day11 实验三：Ollama 本地模型监控 + 并发压测小工具
功能：
  1. 单模型基准：冷启/热启耗时 + 显存占用(size_vram)
  2. 并发压测：同时发 N 个请求，测吞吐与稳定性
  3. 跑完自动卸载模型，不常驻显存
模型：本地 Ollama（默认 http://localhost:11434）
依赖：pip install requests
"""
import time
import json
import requests
from concurrent.futures import ThreadPoolExecutor

BASE = "http://localhost:11434"
PROMPT = "用一句话解释什么是向量数据库"


def generate(model: str, prompt: str, keep_alive: str = "30m", num_ctx: int = 4096) -> float:
    """发一次生成请求，返回耗时（秒）。stream:false 保证拿到完整结果。"""
    t0 = time.perf_counter()
    resp = requests.post(
        f"{BASE}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "keep_alive": keep_alive,
            "options": {"num_ctx": num_ctx},
        },
        timeout=120,
    )
    resp.raise_for_status()
    return time.perf_counter() - t0


def get_ps() -> list:
    """调 /api/ps，返回当前加载的模型列表（含 size_vram / context_length）。"""
    resp = requests.get(f"{BASE}/api/ps", timeout=10)
    resp.raise_for_status()
    return resp.json().get("models", [])


def find_model(ps_models: list, name: str) -> dict | None:
    for m in ps_models:
        if m["name"] == name:
            return m
    return None


def benchmark(model: str, num_ctx: int = 4096):
    print(f"\n=== 基准测试：{model} (num_ctx={num_ctx}) ===")
    dur = generate(model, PROMPT, keep_alive="30m", num_ctx=num_ctx)
    print(f"  首次(冷启)响应耗时：{dur:.2f}s")
    ps = get_ps()
    info = find_model(ps, model)
    if info:
        vram_gb = info.get("size_vram", 0) / (1024 ** 3)
        total_gb = info.get("size", 0) / (1024 ** 3)
        print(f"  显存占用(size_vram)：{vram_gb:.2f} GB | 模型总大小：{total_gb:.2f} GB | CONTEXT：{info.get('context_length')}")
    dur2 = generate(model, PROMPT, keep_alive="30m", num_ctx=num_ctx)
    print(f"  热启响应耗时：{dur2:.2f}s  (加速 {dur/dur2:.1f}×)")


def concurrency_test(model: str, n: int = 4, num_ctx: int = 4096):
    """并发发 n 个请求，记录总耗时与每个请求耗时。"""
    print(f"\n=== 并发压测：{model}  同时 {n} 个请求 ===")
    with ThreadPoolExecutor(max_workers=n) as ex:
        t0 = time.perf_counter()
        futures = [ex.submit(generate, model, PROMPT, "30m", num_ctx) for _ in range(n)]
        durs = [f.result() for f in futures]
        total = time.perf_counter() - t0
    ok = sum(1 for d in durs if d > 0)
    print(f"  成功 {ok}/{n} | 墙钟总耗时：{total:.2f}s | 平均单请求：{sum(durs)/len(durs):.2f}s | 吞吐：{n/total:.2f} req/s")


def cleanup(models: list):
    for m in models:
        requests.post(f"{BASE}/api/generate", json={"model": m, "prompt": "", "keep_alive": 0}, timeout=30)
        print(f"  已卸载 {m}")


if __name__ == "__main__":
    models = ["qwen2.5:1.5b", "qwen2.5:3b", "qwen2.5:7b"]
    for m in models:
        benchmark(m)
    # 并发演示：小模型适合并发，大模型受显存限制
    concurrency_test("qwen2.5:1.5b", n=4)
    concurrency_test("qwen2.5:7b", n=2)
    print("\n--- 清理常驻模型 ---")
    cleanup(models)
