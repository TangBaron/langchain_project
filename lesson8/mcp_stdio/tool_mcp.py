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


@mcp.tool()
def read_file(file_path: str) -> str:
    """
    读取指定路径的文件内容。
    :param file_path: 读取文件的路径
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        return f"文件内容:\n{content}"
    except Exception as e:
        return f"读取文件失败: {e}"


@mcp.tool()
def write_file(file_path: str, content: str) -> str:
    """
    向指定路径写入文件内容，自动创建目录。
    :param file_path: 要写入文件的路径
    :param content: 要写入文件的内容
    """
    try:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"文件写入成功: {file_path}"
    except Exception as e:
        return f"写入文件失败: {e}"


@mcp.tool()
def list_directory(directory_path: str) -> str:
    """
    列出指定目录下的所有文件和文件夹。
    :param directory_path: 要列出目录的路径
    """
    try:
        entries = os.listdir(directory_path)
        content = "\n".join(f"- {e}" for e in entries)
        return f"目录内容:\n{content}"
    except Exception as e:
        return f"列出目录失败: {e}"


@mcp.tool()
def execute_command(command: str, working_directory: str = "") -> str:
    """
    执行指定系统命令，支持指定工作目录，实时显示输出。
    :param command: 要执行的命令
    :param working_directory: 工作目录，默认当前目录
    """
    cwd = working_directory if working_directory else os.getcwd()

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
            return f"命令执行超时（已终止）: {command}\n\n" + "\n".join(output_lines)

        if process.returncode == 0:
            output = "\n".join(output_lines) if output_lines else "(无输出)"
            return f"命令执行成功: {command}\n\n{output}"
        else:
            error_output = "\n".join(output_lines) if output_lines else "(无错误输出)"
            return f"命令执行失败，退出码: {process.returncode}\n错误: {error_output}"
    except Exception as e:
        return f"命令执行失败: {e}"


if __name__ == "__main__":
    mcp.run()
