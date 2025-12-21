import os
import sys
import importlib.util

SRC_DIR = os.path.join(os.path.dirname(__file__), "src")

def list_files():
    """Hiển thị các file .py trong thư mục src"""
    try:
        files = os.listdir(SRC_DIR)
    except FileNotFoundError:
        print(f"Không tìm thấy thư mục {SRC_DIR}")
        return

    py_files = [f for f in files if f.endswith(".py")]
    if not py_files:
        print("Không có file Python nào trong thư mục src.")
    else:
        print("Các file Python trong thư mục src:")
        for f in py_files:
            print(f" - {f}")

def execute_file(filename):
    """Thực thi file python có tên filename trong thư mục src"""
    filepath = os.path.join(SRC_DIR, filename)
    if not os.path.isfile(filepath):
        print(f"File {filename} không tồn tại trong thư mục src.")
        return

    # import module từ đường dẫn file
    spec = importlib.util.spec_from_file_location("module_to_run", filepath)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        print(f"Đã có lỗi khi thực thi {filename}:")
        print(e)

def print_usage():
    print("Cách dùng:")
    print("  python main.py            -> liệt kê các file trong src")
    print("  python main.py <tên file> -> thực thi file <tên file> (trong src)")

def main():
    if len(sys.argv) == 1:
        # chỉ gọi main.py không kèm arg -> list
        list_files()
    elif len(sys.argv) == 2:
        filename = sys.argv[1]
        if not filename.endswith(".py"):
            filename += ".py"
        execute_file(filename)
    else:
        print_usage()

if __name__ == "__main__":
    main()
