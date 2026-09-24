import os
import sys
import zipfile
import time

def create_v8_zip():
    zip_name = "tcg-ai-v8.zip"
    source_dir = os.path.abspath(os.path.dirname(__file__))
    # 输出到 D:\dowload 根目录下，绝不放在工程目录内
    target_dir = os.path.dirname(source_dir)
    zip_path = os.path.join(target_dir, zip_name)
    
    exclude_dirs = {".git", "__pycache__", ".venv", "node_modules", "scratch", ".gemini", ".vs"}
    exclude_files = {
        zip_name, "tcg-ai-v6.zip", "tcg-ai-v8.zip",
        "run_stress_test.bat", "stress_test_balancer.py",
        "pack_v8.py", "patch_cards.py", ".env"
    }
    
    print(f"==================================================")
    print(f" 开始打包 TCG-AI v8 独立全能整合包: {zip_name}")
    print(f" 源目录: {source_dir}")
    print(f" 目标输出路径: {zip_path}")
    print(f" 规则: 无权重文件(*.pth), 无压力测试, 无私钥配置, 保留嵌入式Python标准库")
    print(f"==================================================")
    
    start_time = time.time()
    count = 0
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(source_dir):
            # 过滤排除目录
            dirs[:] = [d for d in dirs if d not in exclude_dirs and not d.startswith('.')]
            
            for file in files:
                # 排除规则
                if file in exclude_files:
                    continue
                if file.endswith('.log') or file.endswith('.tmp'):
                    continue
                # 排除所有模型权重文件
                if file.endswith('.pth') or file.endswith('.pt'):
                    continue
                # 排除压力测试相关脚本及日志
                if "stress_test" in file:
                    continue
                # 排除打包类临时脚本
                if file.startswith("pack_") or file.startswith("repack_") or file.startswith("patch_"):
                    continue
                # 必须保留便携 Python 运行时必需的核心标准库 python310.zip，排除其他外部 zip
                if file.endswith('.zip') and file != "python310.zip":
                    continue
                
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, source_dir)
                
                try:
                    zipf.write(file_path, arcname)
                    count += 1
                    if count % 2000 == 0:
                        print(f"  [打包进度] 已归档 {count} 个文件...")
                except Exception as e:
                    print(f"  [警告] 添加文件失败 {file_path}: {e}")
                    
    elapsed = time.time() - start_time
    size_mb = os.path.getsize(zip_path) / (1024 * 1024)
    print(f"==================================================")
    print(f" 打包完成！成功生成: {zip_name}")
    print(f" 归档文件数: {count} 个")
    print(f" 压缩包体积: {size_mb:.2f} MB")
    print(f" 打包总耗时: {elapsed:.2f} 秒")
    print(f" 最终绝对路径: {zip_path}")
    print(f"==================================================")

if __name__ == "__main__":
    create_v8_zip()
