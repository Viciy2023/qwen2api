#!/usr/bin/env python3
import os
import sys
import subprocess
import time
import signal
from pathlib import Path

# ==========================================
# qwen2API Enterprise Gateway - Python 跨平台点火脚本
# ==========================================

WORKSPACE_DIR = Path(__file__).parent.absolute()
BACKEND_DIR = WORKSPACE_DIR / "backend"
FRONTEND_DIR = WORKSPACE_DIR / "frontend"
LOGS_DIR = WORKSPACE_DIR / "logs"

def ensure_dirs():
    LOGS_DIR.mkdir(exist_ok=True)
    (WORKSPACE_DIR / "data").mkdir(exist_ok=True)

def check_and_install_dependencies():
    print("⚡ [系统预检] 正在扫描底层铁壁的 Python 环境...")
    python_exec = sys.executable
    
    # 安装后端依赖
    try:
        subprocess.check_call(
            [python_exec, "-m", "pip", "install", "-r", "requirements.txt", "--quiet"],
            cwd=BACKEND_DIR,
            stdout=None, # 将 pip 安装日志直接输出，暴露错误
            stderr=subprocess.STDOUT
        )
    except Exception as e:
        print(f"⚠ [预检警告] 后端依赖安装异常: {e}")
        
    print("⚡ [系统预检] 正在下载并配置浏览器内核 (Camoufox)...")
    try:
        subprocess.check_call(
            [python_exec, "-m", "camoufox", "fetch"],
            cwd=WORKSPACE_DIR,
            stdout=None, # 将输出打印到终端，暴露详细报错
            stderr=subprocess.STDOUT
        )
    except Exception as e:
        print(f"⚠ [预检警告] 浏览器内核配置异常: {e}")

    print("⚡ [系统预检] 正在扫描前端王座的 Node 环境...")
    is_windows = (os.name == "nt")
    npm_cmd = "npm install" if is_windows else ["npm", "install"]
    
    # 检查前端 node_modules 是否存在，如果不存在或为了安全起见，执行 npm install
    try:
        # 为了给用户一个清晰的进度，不吞噬这里的输出
        print("  -> 正在执行 npm install (可能需要一点时间，请耐心等待)...")
        subprocess.check_call(
            npm_cmd,
            cwd=FRONTEND_DIR,
            shell=is_windows,
            stdout=None, # 将 npm install 的日志也直接输出到终端，让你能看到为什么 npm 失败
            stderr=subprocess.STDOUT
        )
        print("✓ [预检通过] 前端依赖已就绪。")
    except subprocess.CalledProcessError as e:
        print(f"❌ [预检失败] 前端 npm install 失败，请检查是否安装了 Node.js: {e}")
        sys.exit(1)
    
def start_backend() -> subprocess.Popen:
    print("⚡ 正在唤醒底层铁壁 (Backend)...")
    log_file = open(LOGS_DIR / "backend.log", "w", encoding="utf-8")
    
    # 根据系统判断 python 执行文件
    python_exec = sys.executable
    
    # 注入 PYTHONPATH，让 backend 内的绝对导入生效
    env = os.environ.copy()
    env["PYTHONPATH"] = str(WORKSPACE_DIR)
    
    proc = subprocess.Popen(
        [python_exec, "backend/main.py"],
        cwd=WORKSPACE_DIR,
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT
    )
    print(f"✓ Backend 已点火 (PID: {proc.pid}) -> 日志: logs/backend.log")
    return proc

def start_frontend() -> subprocess.Popen:
    print("⚡ 正在唤醒前端王座 (Admin Dashboard)...")
    # 让前端的日志既写文件，也同时在终端输出（如果它很快报错的话），这里直接用 None 让它在终端打印，方便你排查
    log_file = open(LOGS_DIR / "frontend.log", "w", encoding="utf-8")
    
    # 跨平台调用 npm
    is_windows = (os.name == "nt")
    npm_cmd = "npm run dev" if is_windows else ["npm", "run", "dev"]
    
    proc = subprocess.Popen(
        npm_cmd,
        cwd=FRONTEND_DIR,
        shell=is_windows, # 在 Windows 上通过 shell 启动 npm
        # 将输出直接抛到你的终端，不吞噬，让你看清楚真正的死因
        stdout=None,
        stderr=None
    )
    print(f"✓ Frontend 已点火 (PID: {proc.pid}) -> 终端直出报错")
    return proc

def main():
    ensure_dirs()
    check_and_install_dependencies()
    
    backend_proc = start_backend()
    time.sleep(1) # 稍微错开启动时间
    frontend_proc = start_frontend()
    
    print("\n==========================================")
    print("帝国已上线。")
    print("▶ 前端中枢: http://localhost:5173")
    print("▶ 后端核心: http://localhost:8080")
    print("==========================================")
    print("按 Ctrl+C 掐断所有进程并关闭系统。")
    
    def signal_handler(sig, frame):
        print("\n\n⚠ 收到关闭指令，正在掐断进程...")
        backend_proc.terminate()
        frontend_proc.terminate()
        backend_proc.wait()
        frontend_proc.wait()
        print("✓ 所有进程已被摧毁，帝国下线。")
        sys.exit(0)
        
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # 保持主进程存活，同时监控子进程状态
    try:
        while True:
            if backend_proc.poll() is not None:
                print(f"❌ Backend 异常退出 (Exit Code: {backend_proc.returncode})")
                break
            if frontend_proc.poll() is not None:
                print(f"❌ Frontend 异常退出 (Exit Code: {frontend_proc.returncode})")
                break
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        # 如果是因为某个子进程挂了跳出循环，确保把另一个也杀掉
        if backend_proc.poll() is None: backend_proc.terminate()
        if frontend_proc.poll() is None: frontend_proc.terminate()

if __name__ == "__main__":
    main()
