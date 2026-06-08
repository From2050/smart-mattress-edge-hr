import serial
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button
from collections import deque
import datetime
import threading
import numpy as np

# 設置串口參數（請根據實際情況替換）
# ser = serial.Serial('/dev/tty.usbserial-0001', 115200, timeout=1)
# ser = serial.Serial('/dev/tty.wchusbserial57340014541', 921600, timeout=1)
ser = serial.Serial('/dev/tty.usbmodem101', 921600, timeout=1)


# 創建存儲數據的隊列 (33個通道 - 32個ADC + 1個PPG)
queue_size = 100
adc_data = [deque([0]*queue_size, maxlen=queue_size) for _ in range(32)]
ppg_data = deque([0]*queue_size, maxlen=queue_size)

# 創建一個列表來存儲錄製的數據
recorded_data = []
is_recording = False
exit_flag = False

# 初始化圖表
fig = plt.figure(figsize=(18, 12))  # 加大圖表尺寸以便更好地顯示所有子圖
plt.subplots_adjust(bottom=0.15, hspace=0.4, wspace=0.2)  # 減小子圖間距，使排列更緊密

# 創建子圖布局
# 第一行：ADC1-16 (每行16個子圖)
adc1_axes = []
for i in range(16):
    ax = plt.subplot(3, 16, i + 1)  # 3行16列布局，第一行
    ax.set_title(f'ADC{i+1}', fontsize=8)
    ax.tick_params(axis='both', labelsize=6)
    ax.tick_params(axis='y', labelleft=False)  # 隱藏Y軸坐標數字
    ax.set_yticks([])  # 移除Y軸刻度
    if i > 0:  # 只在第一個子圖顯示Y軸標籤
        ax.set_ylabel('')
    adc1_axes.append(ax)

# 第二行：ADC17-32 (每行16個子圖)
adc2_axes = []
for i in range(16):
    ax = plt.subplot(3, 16, i + 17)  # 3行16列布局，第二行
    ax.set_title(f'ADC{i+17}', fontsize=8)
    ax.tick_params(axis='both', labelsize=6)
    ax.tick_params(axis='y', labelleft=False)  # 隱藏Y軸坐標數字
    ax.set_yticks([])  # 移除Y軸刻度
    if i > 0:  # 只在第一個子圖顯示Y軸標籤
        ax.set_ylabel('')
    adc2_axes.append(ax)

# 第三行：PPG (1 個圖，使用整行)
ppg_axis = plt.subplot(3, 1, 3)  # 最後一行完整顯示
ppg_axis.set_title('PPG Signal')
ppg_axis.set_xlabel('Sample')
ppg_axis.set_ylabel('Value')

# 為每個子圖創建線條，使用不同顏色區分
colors = plt.cm.rainbow(np.linspace(0, 1, 32))  # 使用彩虹色階為每個ADC通道創建不同顏色
lines_adc1 = [ax.plot([], [], color=colors[i])[0] for i, ax in enumerate(adc1_axes)]
lines_adc2 = [ax.plot([], [], color=colors[i+16])[0] for i, ax in enumerate(adc2_axes)]
line_ppg, = ppg_axis.plot([], [], 'r-', linewidth=1.5)

def update(frame):
    x_data = list(range(queue_size))
    
    # 更新 ADC1-16 線
    for i, line in enumerate(lines_adc1):
        line.set_data(x_data, list(adc_data[i]))
        adc1_axes[i].relim()
        adc1_axes[i].autoscale_view()
    
    # 更新 ADC17-32 線
    for i, line in enumerate(lines_adc2):
        line.set_data(x_data, list(adc_data[i+16]))
        adc2_axes[i].relim()
        adc2_axes[i].autoscale_view()
    
    # 更新 PPG 線
    line_ppg.set_data(x_data, list(ppg_data))
    ppg_axis.relim()
    ppg_axis.autoscale_view()
    
    return lines_adc1 + lines_adc2 + [line_ppg]

def read_serial():
    global is_recording, exit_flag

    while not exit_flag:
        try:
            # 從串口讀取數據
            stream = ser.readline().decode('latin-1').strip()
            if stream:
                try:
                    # 解析數據並添加到隊列中
                    values = stream.split(',')
                    
                    # 確保數據至少有34個值 (32 ADC + PPG + cnt + HR)
                    if len(values) >= 34:
                        # 更新所有 ADC 通道數據
                        for i in range(32):
                            adc_data[i].append(float(values[i]))
                        
                        # 更新 PPG 數據 (在 ADC32 之後)
                        ppg_value = float(values[32])
                        ppg_data.append(ppg_value)

                        # 如果正在錄製，將數據添加到recorded_data
                        if is_recording:
                            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                            recorded_data.append((timestamp, values))

                except ValueError as e:
                    print(f"Error parsing data: {e}")
                    pass
        except serial.SerialException:
            print("Serial port closed.")
            break
        except Exception as e:
            print(f"Error: {e}")

def start_stop_recording(event):
    global is_recording
    is_recording = not is_recording
    if is_recording:
        print("Recording started...")
        start_stop_button.label.set_text("Stop")
    else:
        print("Recording stopped.")
        start_stop_button.label.set_text("Record")

def save_data(event):
    if recorded_data:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"ppg_data_{timestamp}.txt"
        with open(filename, 'w') as f:
            for item in recorded_data:
                time_stamp = item[0]
                values = ','.join(str(val) for val in item[1])
                f.write(f"{time_stamp},{values}\n")
        print(f"Data saved to {filename}")
    else:
        print("No data to save.")

def clear_data(event):
    global recorded_data
    recorded_data = []
    print("Recorded data cleared.")

def exit_program(event):
    global exit_flag
    exit_flag = True
    plt.close()
    print("Program exited.")

# 創建按鈕
ax_start_stop = plt.axes([0.1, 0.05, 0.15, 0.075])
ax_save = plt.axes([0.3, 0.05, 0.15, 0.075])
ax_clear = plt.axes([0.5, 0.05, 0.15, 0.075])
ax_exit = plt.axes([0.7, 0.05, 0.15, 0.075])

start_stop_button = Button(ax_start_stop, 'Record')
save_button = Button(ax_save, 'Save')
clear_button = Button(ax_clear, 'Clear')
exit_button = Button(ax_exit, 'Exit')

start_stop_button.on_clicked(start_stop_recording)
save_button.on_clicked(save_data)
clear_button.on_clicked(clear_data)
exit_button.on_clicked(exit_program)

# 設置主標題
plt.suptitle('32-Channel ADC + PPG Monitoring', fontsize=16)

# 啟動串口讀取線程
serial_thread = threading.Thread(target=read_serial, daemon=True)
serial_thread.start()

# 創建動畫
ani = animation.FuncAnimation(fig, update, interval=100, blit=True)

plt.tight_layout(rect=[0, 0.12, 1, 0.97])  # 調整版面配置，避開按鈕區域
plt.show()

# 關閉串口
ser.close()