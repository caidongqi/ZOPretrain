import psutil
import time
import csv
from datetime import datetime

# ================= 配置区域 =================
DURATION = 3600   # 持续监控时间 (秒)
INTERVAL = 5      # 采样间隔 (秒)
TOP_N = 10        # 记录前 N 个进程
FILENAME = f"mac_metrics_detailed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
# ===========================================

def get_network_usage(last_net):
    current_net = psutil.net_io_counters()
    sent = current_net.bytes_sent - last_net.bytes_sent
    recv = current_net.bytes_recv - last_net.bytes_recv
    return sent, recv, current_net

def get_disk_usage(last_disk):
    current_disk = psutil.disk_io_counters()
    read = current_disk.read_bytes - last_disk.read_bytes
    write = current_disk.write_bytes - last_disk.write_bytes
    return read, write, current_disk

def get_top_processes(n=10):
    """
    获取 CPU 和 内存 占用最高的 N 个进程。
    注意：macOS 不支持通过 psutil 获取单个进程的网络/磁盘 IO。
    """
    # 获取所有进程的快照
    procs = []
    for p in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
        try:
            # cpu_percent 需要调用一次以初始化，这里直接取值可能第一次不准，但在循环中会趋于准确
            # 也可以在这里显式调用 p.cpu_percent(interval=None)
            p_info = p.info
            # 简单的过滤，忽略占用极低的闲置进程
            if p_info['cpu_percent'] is None: p_info['cpu_percent'] = 0
            if p_info['memory_percent'] is None: p_info['memory_percent'] = 0
            procs.append(p_info)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass

    # 按 CPU 排序 (能耗参考)
    top_cpu_list = sorted(procs, key=lambda x: x['cpu_percent'], reverse=True)[:n]
    # 格式化为字符串: "Chrome(20.5%) | Python(10.2%)..."
    top_cpu_str = " | ".join([f"{p['name']}({p['cpu_percent']}%)" for p in top_cpu_list])

    # 按 内存 排序
    top_mem_list = sorted(procs, key=lambda x: x['memory_percent'], reverse=True)[:n]
    top_mem_str = " | ".join([f"{p['name']}({p['memory_percent']:.1f}%)" for p in top_mem_list])

    return top_cpu_str, top_mem_str

print(f"🚀 开始深度监控...")
print(f"📂 数据将保存至: {FILENAME}")
print(f"⏱️  预计时长: {DURATION/60} 分钟 (每 {INTERVAL} 秒采样一次)")

# 定义 CSV 头部
header = [
    "Timestamp", 
    "Total_CPU(%)", 
    "Total_Mem(%)", 
    "Net_Sent(Bytes)", 
    "Net_Recv(Bytes)", 
    "Disk_Read(Bytes)", 
    "Disk_Write(Bytes)", 
    "Battery(%)",
    f"Top_{TOP_N}_CPU_Processes_High_Energy", # 新增列
    f"Top_{TOP_N}_Mem_Processes"              # 新增列
]

with open(FILENAME, mode='w', newline='', encoding='utf-8-sig') as file:
    writer = csv.writer(file)
    writer.writerow(header)

    # 初始化计数器
    last_net = psutil.net_io_counters()
    last_disk = psutil.disk_io_counters()
    
    # 首次调用 cpu_percent 给每个进程预热，避免第一次数据为 0
    psutil.cpu_percent(interval=None)
    for p in psutil.process_iter():
        try:
            p.cpu_percent(interval=None)
        except:
            pass
            
    start_time = time.time()
    
    try:
        while (time.time() - start_time) < DURATION:
            loop_start = time.time()
            current_time = datetime.now().strftime('%H:%M:%S')
            
            # 1. 获取系统级指标
            cpu_total = psutil.cpu_percent(interval=None)
            mem_total = psutil.virtual_memory().percent
            net_sent, net_recv, last_net = get_network_usage(last_net)
            disk_read, disk_write, last_disk = get_disk_usage(last_disk)
            
            battery = psutil.sensors_battery()
            bat_percent = battery.percent if battery else "N/A"

            # 2. 获取进程级指标 (Top N)
            # 注意：这里需要一点时间遍历进程，这包含在 interval 等待时间内
            top_cpu_str, top_mem_str = get_top_processes(TOP_N)

            # 3. 写入 CSV
            writer.writerow([
                current_time, cpu_total, mem_total, 
                net_sent, net_recv, disk_read, disk_write, bat_percent,
                top_cpu_str, top_mem_str
            ])
            
            # 打印简报，确认脚本在运行
            print(f"[{current_time}] CPU: {cpu_total}% | Net: {net_recv/1024:.1f}KB/s | 写入完成")

            # 智能休眠：扣除代码执行消耗的时间，保证采样间隔相对准确
            elapsed = time.time() - loop_start
            sleep_time = max(0, INTERVAL - elapsed)
            time.sleep(sleep_time)
            
    except KeyboardInterrupt:
        print("\n🛑 监控手动停止。")

print(f"\n✅ 完成！请查看文件: {FILENAME}")