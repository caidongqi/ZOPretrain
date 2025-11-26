import psutil
import time
import csv
from datetime import datetime

# 配置
DURATION = 3600  # 持续时间：1小时 (秒)
INTERVAL = 5     # 采样间隔：5秒
FILENAME = f"mac_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

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

print(f"开始监控... 数据将保存至 {FILENAME}")
print(f"预计时长: {DURATION/60} 分钟")

# 初始化头部
header = ["Timestamp", "CPU_Usage(%)", "Memory_Usage(%)", "Net_Sent(Bytes)", "Net_Recv(Bytes)", "Disk_Read(Bytes)", "Disk_Write(Bytes)", "Battery_Percent"]

with open(FILENAME, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(header)

    # 获取初始计数器
    last_net = psutil.net_io_counters()
    last_disk = psutil.disk_io_counters()
    
    start_time = time.time()
    
    try:
        while (time.time() - start_time) < DURATION:
            current_time = datetime.now().strftime('%H:%M:%S')
            
            # 1. CPU & Memory
            cpu = psutil.cpu_percent(interval=None)
            mem = psutil.virtual_memory().percent
            
            # 2. Network (Delta)
            net_sent, net_recv, last_net = get_network_usage(last_net)
            
            # 3. Disk (Delta)
            disk_read, disk_write, last_disk = get_disk_usage(last_disk)

            # 4. Battery (Energy proxy)
            battery = psutil.sensors_battery()
            bat_percent = battery.percent if battery else "N/A"

            # 写入行
            writer.writerow([current_time, cpu, mem, net_sent, net_recv, disk_read, disk_write, bat_percent])
            
            time.sleep(INTERVAL)
            
    except KeyboardInterrupt:
        print("\n监控手动停止。")

print(f"\n完成！数据已保存至 {FILENAME}")