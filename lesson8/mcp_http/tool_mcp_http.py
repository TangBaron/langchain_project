from fastmcp import FastMCP
from pathlib import Path
import os
import subprocess
import sys
import io

# 强制 stdout/stderr 使用 utf-8，避免 Windows 下 gbk 编码问题
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

mcp = FastMCP("agent-tools")

LOG_FILE = "mcp_server.log"


def log(msg: str):
    """输出日志到 stderr 和日志文件"""
    print(msg, file=sys.stderr, flush=True)
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
    except Exception:
        pass


@mcp.tool()
def read_file(file_path: str) -> str:
    """读取指定路径的文件内容。"""
    log(f"[工具调用] read_file(\"{file_path}\")")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        log(f"[工具调用] read_file(\"{file_path}\") - 成功读取 {len(content)} 字节")
        return f"文件内容:\n{content}"
    except Exception as e:
        log(f"[工具调用] read_file(\"{file_path}\") - 错误: {e}")
        return f"读取文件失败: {e}"


@mcp.tool()
def write_file(file_path: str, content: str) -> str:
    """向指定路径写入文件内容，自动创建目录。"""
    log(f"[工具调用] write_file(\"{file_path}\")")
    try:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        log(f"[工具调用] write_file(\"{file_path}\") - 成功写入 {len(content)} 字节")
        return f"文件写入成功: {file_path}"
    except Exception as e:
        log(f"[工具调用] write_file(\"{file_path}\") - 错误: {e}")
        return f"写入文件失败: {e}"


@mcp.tool()
def list_directory(directory_path: str) -> str:
    """列出指定目录下的所有文件和文件夹。"""
    log(f"[工具调用] list_directory(\"{directory_path}\")")
    try:
        entries = os.listdir(directory_path)
        log(f"[工具调用] list_directory(\"{directory_path}\") - 找到 {len(entries)} 个项目")
        content = "\n".join(f"- {e}" for e in entries)
        return f"目录内容:\n{content}"
    except Exception as e:
        log(f"[工具调用] list_directory(\"{directory_path}\") - 错误: {e}")
        return f"列出目录失败: {e}"


@mcp.tool()
def execute_command(command: str, working_directory: str = "") -> str:
    """执行系统命令，支持指定工作目录，实时显示输出。"""
    cwd = working_directory if working_directory else os.getcwd()
    log(f"[工具调用] execute_command(\"{command}\") - 工作目录: {cwd}")

    try:
        env = os.environ.copy()
        env["CI"] = "true"

        process = subprocess.Popen(
            command,
            shell=True,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            errors='replace',
            stdin=subprocess.DEVNULL,
            env=env,
        )

        output_lines = []
        for line in process.stdout:
            line = line.rstrip()
            output_lines.append(line)

        try:
            process.wait(timeout=100)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            log(f"[工具调用] execute_command(\"{command}\") - 超时已终止")
            return f"命令执行超时（已终止）: {command}\n\n" + "\n".join(output_lines)

        if process.returncode == 0:
            log(f"[工具调用] execute_command(\"{command}\") - 执行成功")
            output = "\n".join(output_lines) if output_lines else "(无输出)"
            return f"命令执行成功: {command}\n\n{output}"
        else:
            log(f"[工具调用] execute_command(\"{command}\") - 执行失败，退出码: {process.returncode}")
            error_output = "\n".join(output_lines) if output_lines else "(无错误输出)"
            return f"命令执行失败，退出码: {process.returncode}\n错误: {error_output}"
    except Exception as e:
        log(f"[工具调用] execute_command(\"{command}\") - 异常: {e}")
        return f"命令执行失败: {e}"


if __name__ == "__main__":
    # 使用 HTTP/SSE 传输模式运行，默认监听 localhost:8000
    mcp.run(transport="streamable-http")
