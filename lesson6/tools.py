from langchain.tools import tool

@tool
def read_file(file_path: str) -> str:
    """
    读取指定路径的文件内容。
    :param file_path: 读取文件的路径
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        print(f"  [工具调用] read_file(\"{file_path}\") - 成功读取 {len(content)} 字节")
        return f"文件内容:\n{content}"
    except Exception as e:
        print(f"  [工具调用] read_file(\"{file_path}\") - 错误: {e}")
        return f"读取文件失败: {e}"

from pathlib import Path

@tool
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
        print(f"  [工具调用] write_file(\"{file_path}\") - 成功写入 {len(content)} 字节")
        return f"文件写入成功: {file_path}"
    except Exception as e:
        print(f"  [工具调用] write_file(\"{file_path}\") - 错误: {e}")
        return f"写入文件失败: {e}"

import os

@tool
def list_directory(directory_path: str) -> str:
    """
    列出指定目录下的所有文件和文件夹。
    :param directory_path: 指定目录路径
    """
    try:
        entries = os.listdir(directory_path)
        print(f"  [工具调用] list_directory(\"{directory_path}\") - 找到 {len(entries)} 个项目")
        content = "\n".join(f"- {e}" for e in entries)
        return f"目录内容:\n{content}"
    except Exception as e:
        print(f"  [工具调用] list_directory(\"{directory_path}\") - 错误: {e}")
        return f"列出目录失败: {e}"

import subprocess
import sys
@tool
def execute_command(command: str, working_directory: str = "") -> str:
    """
    执行系统命令，支持指定工作目录，实时显示输出。
    :param command: 要执行的命令
    :param working_directory: 工作目录（推荐指定）
    """
    cwd = working_directory if working_directory else os.getcwd()
    print(f"  [工具调用] execute_command(\"{command}\")" +
          (f" - 工作目录: {cwd}" if working_directory else ""))

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
            print(f"    {line}")
            sys.stdout.flush()

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
            print(f"  [工具调用] execute_command(\"{command}\") - 执行成功")
            output = "\n".join(output_lines) if output_lines else "(无输出)"
            return f"命令执行成功: {command}\n\n{output}"
        else:
            print(f"  [工具调用] execute_command(\"{command}\") - 执行失败，退出码: {process.returncode}")
            error_output = "\n".join(output_lines) if output_lines else "(无错误输出)"
            return f"命令执行失败，退出码: {process.returncode}\n错误: {error_output}"
    except Exception as e:
        return f"命令执行失败: {e}"

all_tools = [read_file, write_file, execute_command, list_directory]

if __name__ == '__main__':
    pass
    # execute_command("npx -y create-vite@latest retodo-app --template react-ts -- --no-interactive")