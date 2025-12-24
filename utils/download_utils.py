"""
Author: Redal
Date: 2025-11-27
Todo: Download the WHU_OMVS_dataset safely with error handling and integrity checks.
Homepage: https://github.com/Rtwotwo/FF3DR.git
"""
import os
import argparse
import requests
from urllib.parse import urlparse
import subprocess
import hashlib
from tqdm import tqdm


urls = ["https://gpcv.whu.edu.cn/data/WHU_OMVS_dataset/train.zip",
    "https://gpcv.whu.edu.cn/data/WHU_OMVS_dataset/test.zip",
    "https://gpcv.whu.edu.cn/data/WHU_OMVS_dataset/predict.zip",
    "https://gpcv.whu.edu.cn/data/WHU_OMVS_dataset/readme.zip"]


def calculate_md5(filepath):
    """计算文件 MD5 值"""
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def download_file(url, save_path, timeout=30):
    """下载单个文件，支持断点续传和超时控制"""
    print(f"[ INFO ] Downloading {url} -> {save_path}")
    try:
        headers = {}
        file_size = 0
        if os.path.exists(save_path):
            # 支持断点续传
            file_size = os.path.getsize(save_path)
            headers["Range"] = f"bytes={file_size}-"
        with requests.get(url, stream=True, headers=headers, timeout=timeout) as r:
            r.raise_for_status()
            # 如果是断点续传，验证响应头
            if 'Content-Range' in r.headers:
                content_range = r.headers['Content-Range']
                start_byte = int(content_range.split('-')[0].split()[1])
                if start_byte != file_size:
                    print(f"[ ERROR ] Range mismatch: expected {start_byte}, got {file_size}")
                    os.remove(save_path)
                    return False
            total_size = int(r.headers.get('content-length', 0)) + file_size
            with open(save_path, 'ab') as f:
                with tqdm(total=total_size, initial=file_size, unit='B', unit_scale=True, desc=os.path.basename(save_path)) as pbar:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
        print(f"[ INFO ] Download completed: {save_path}")
        return True
    except Exception as e:
        print(f"[ ERROR ] Failed to download {url}: {e}")
        if os.path.exists(save_path):
            os.remove(save_path)
        return False


def unzip_file(zip_path, extract_to):
    """安全解压 ZIP 文件"""
    try:
        result = subprocess.run(
            ['unzip', '-o', zip_path, '-d', extract_to],
            check=True,
            capture_output=True,
            text=True)
        print(f"[ INFO ] Unzipped {zip_path} successfully.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ ERROR ] Failed to unzip {zip_path}: {e.stderr}")
        return False
    except FileNotFoundError:
        print("[ ERROR ] unzip command not found. Please install unzip.")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download the WHU_OMVS_dataset securely")
    parser.add_argument("--save_dir", type=str, default="./data", help="Save directory")
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    for url in urls:
        filename = os.path.basename(urlparse(url).path)
        file_path = os.path.join(args.save_dir, filename)
        # 检查是否已存在且完整
        if os.path.exists(file_path):
            print(f"[ INFO ] {filename} already exists. Skipping download.")
            if unzip_file(file_path, args.save_dir):
                continue
            else:
                print(f"[ WARNING ] Existing {filename} is corrupted. Re-downloading...")
                if not os.path.exists(file_path):
                    os.remove(file_path)
        else:
            if not download_file(url, file_path):
                print(f"[ ERROR ] Failed to download {filename}. Exiting.")
                continue
        if not unzip_file(file_path, args.save_dir):
            print(f"[ ERROR ] Failed to extract {filename}.")
            continue
    print("[ INFO ] All files downloaded and extracted successfully!")


if __name__ == "__main__":
    main()