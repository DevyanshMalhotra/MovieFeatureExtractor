import platform
import sys
import psutil
import subprocess
import pkg_resources

# Function to get all system configurations
def get_system_info():
    system_info = {
        "OS": f"{platform.system()} {platform.release()}",
        "OS Version": platform.version(),
        "Machine": platform.machine(),
        "Processor": platform.processor(),
        "Python Version": sys.version,
        "CPU Cores": f"{psutil.cpu_count(logical=False)} (Physical), {psutil.cpu_count(logical=True)} (Logical)",
        "CPU Frequency": f"{psutil.cpu_freq()} MHz",
        "CPU Usage": f"{psutil.cpu_percent(interval=1)}%",
        "Memory": f"Total: {psutil.virtual_memory().total / (1024 ** 3):.2f} GB, Available: {psutil.virtual_memory().available / (1024 ** 3):.2f} GB, Usage: {psutil.virtual_memory().percent}%",
        "Disk Space": f"Total: {psutil.disk_usage('/').total / (1024 ** 3):.2f} GB, Used: {psutil.disk_usage('/').used / (1024 ** 3):.2f} GB, Usage: {psutil.disk_usage('/').percent}%",
        "Installed Packages": "\n".join([f"{pkg.key}=={pkg.version}" for pkg in pkg_resources.working_set])
    }
    
    # GPU Info (if available)
    try:
        result = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total,memory.free,memory.used", "--format=csv,noheader,nounits"])
        result = result.decode('utf-8').strip().split("\n")
        system_info["GPU"] = "\n".join(result)
    except Exception:
        system_info["GPU"] = "No GPU found"

    return system_info

# Print System Info
sys_info = get_system_info()
for key, value in sys_info.items():
    print(f"{key}: {value}\n")
