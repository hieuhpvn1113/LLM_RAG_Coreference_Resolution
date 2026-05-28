import os
import shutil
import subprocess
import sys
from mcp.server.fastmcp import FastMCP

# Khoi tao MCP Server
mcp = FastMCP("LLM_RAG")

# Lay duong dan tu file JSON cau hinh (env)
ALLOWED_DIRECTORY = os.environ.get("ALLOWED_DIRECTORY", os.getcwd())


def get_safe_path(relative_path: str) -> str:
    """Ham bo tro de dam bao duong dan luon nam trong vung an toan."""
    clean_path = os.path.normpath(relative_path).lstrip("/")
    if clean_path.startswith(".."):
        raise ValueError("Khong the truy cap khu vuc ngoai vung cau hinh ALLOWED_DIRECTORY.")
    return os.path.join(ALLOWED_DIRECTORY, clean_path)


@mcp.tool()
def run_terminal_command(command: str, relative_path: str = "."):
    """
    Chay mot lenh CMD/Terminal tai thu muc du an hoac thu muc con cu the.
    Vi du: command="pip install requests", command="python test.py"
    """
    try:
        execution_dir = get_safe_path(relative_path)

        if not os.path.exists(execution_dir):
            return f"Loi: Thu muc thuc thi khong ton tai: {relative_path}"

        result = subprocess.run(
            command,
            cwd=execution_dir,
            shell=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
        )

        output = []
        if result.stdout:
            output.append(f"--- [STDOUT] ---\n{result.stdout}")
        if result.stderr:
            output.append(f"--- [STDERR] ---\n{result.stderr}")

        if not output:
            return f"Lenh da thuc thi thanh cong (Ma thoat: {result.returncode}), khong co du lieu xuat ra."

        return "\n".join(output)

    except subprocess.TimeoutExpired:
        return f"Loi: Lenh bi dung do chay qua thoi gian quy dinh (Timeout 60s): {command}"
    except Exception as e:
        return f"Loi he thong khi thuc thi lenh: {str(e)}"


@mcp.tool()
def list_files(relative_path: str = "."):
    """Liet ke danh sach file va folder trong thu muc du an."""
    try:
        search_path = get_safe_path(relative_path)
        if not os.path.exists(search_path):
            return f"Loi: Duong dan khong ton tai: {relative_path}"
        return os.listdir(search_path)
    except Exception as e:
        return f"Loi: {str(e)}"


@mcp.tool()
def read_file(file_name: str):
    """Doc noi dung cua mot file cu the."""
    try:
        file_path = get_safe_path(file_name)
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Loi khong the doc file: {str(e)}"


@mcp.tool()
def write_file(file_name: str, content: str):
    """Ghi noi dung vao file. Tu dong tao file moi neu chua co, hoac ghi de neu da ton tai."""
    try:
        file_path = get_safe_path(file_name)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Da luu/tao file thanh cong: {file_name}"
    except Exception as e:
        return f"Loi khi ghi file: {str(e)}"


@mcp.tool()
def make_directory(dir_name: str):
    """Tao mot hoac nhieu thu muc moi (tu dong tao thu muc long nhau neu can)."""
    try:
        dir_path = get_safe_path(dir_name)
        os.makedirs(dir_path, exist_ok=True)
        return f"Da tao thu muc thanh cong: {dir_name}"
    except Exception as e:
        return f"Loi khi tao thu muc: {str(e)}"


@mcp.tool()
def delete_path(target_path: str):
    """Xoa mot file hoac mot thu muc (bao gom tat ca file/thu muc con ben trong)."""
    try:
        full_path = get_safe_path(target_path)
        if not os.path.exists(full_path):
            return f"Loi: Duong dan khong ton tai de xoa: {target_path}"

        if os.path.isdir(full_path):
            shutil.rmtree(full_path)
            return f"Da xoa thu muc va toan bo noi dung ben trong: {target_path}"
        os.remove(full_path)
        return f"Da xoa file thanh cong: {target_path}"
    except Exception as e:
        return f"Loi khi xoa: {str(e)}"


@mcp.tool()
def move_path(source_path: str, destination_path: str):
    """Di chuyen hoac doi ten file/thu muc tu vi tri cu sang vi tri moi."""
    try:
        src = get_safe_path(source_path)
        dst = get_safe_path(destination_path)

        if not os.path.exists(src):
            return f"Loi: Vi tri nguon khong ton tai: {source_path}"

        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.move(src, dst)
        return f"Da di chuyen/doi ten tu '{source_path}' sang '{destination_path}' thanh cong."
    except Exception as e:
        return f"Loi khi di chuyen: {str(e)}"


if __name__ == "__main__":
    mcp.run()
