/*
 * SPDX-FileCopyrightText: 2010-2022 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: CC0-1.0
 */

#include <stdio.h>
#include <string.h>
#include "sdkconfig.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "driver/gpio.h"
#include "driver/uart.h"
#include "esp_system.h"
#include "esp_mac.h"
#include "spi_flash_mmap.h" // 替換已弃用的头文件

#include "driver/i2c_master.h" // 新的 I2C master 標頭檔
/* #include "esp_adc/adc_oneshot.h"
#include "esp_adc/adc_cali.h" // 如果需要校準，則需要
#include "esp_adc/adc_cali_scheme.h" // 如果需要校準，則需要 */

// 新增 limits.h 以使用 INT16_MIN
#include <limits.h>
// 新增 rom/ets_sys.h 以使用 ets_delay_us
#include "rom/ets_sys.h"

// 導入 MAX30105 相關庫
#include "MAX30105.h"
#include "heartRate.h"
#include "Wire.h"

// --- 新增任務標籤 ---
static const char *TAG_MAIN = "APP_MAIN";
static const char *TAG_ADS = "ADS1115_TASK";
static const char *TAG_MAX = "MAX30105_TASK";
static const char *TAG_OUT = "OUTPUT_TASK";
// --- 結束新增任務標籤 ---

// --- ADS1115 定義 ---
#define ADS1115_I2C_MASTER_SCL_IO    GPIO_NUM_38      // ESP32-S3 I2C SCL for ADS1115
#define ADS1115_I2C_MASTER_SDA_IO    GPIO_NUM_37      // ESP32-S3 I2C SDA for ADS1115
#define ADS1115_I2C_PORT_NUM         I2C_NUM_1        // 使用 I2C 控制器 1
#define ADS1115_I2C_MASTER_FREQ_HZ   400000           // I2C 時脈頻率 (400kHz)
#define ADS1115_ADDR                 0x48             // ADS1115 I2C 位址
#define ADS1115_ADDR_2               0x49             // ADS1115-2 I2C 位址
#define ADS1115_REG_CONVERSION       0x00
#define ADS1115_REG_CONFIG           0x01

// --- 新的 I2C Master API 控制代碼 (全域) ---
static i2c_master_bus_handle_t i2c_bus_handle_ads; // I2C1 的 Bus Handle
static i2c_master_dev_handle_t ads1115_dev_handle_1 = NULL; // For 0x48
static i2c_master_dev_handle_t ads1115_dev_handle_2 = NULL; // For 0x49
// --- 結束 ADS1115 定義 ---

// --- MAX30105 全域物件 ---
MAX30105 particleSensor;
// --- 結束 MAX30105 全域物件 ---

// lastBeat and beatsPerMinute are declared extern in heartRate.h and defined in heartRate.cpp

#define FINGER_ON 7000  // 紅外線最小量（判斷手指有沒有上）

// --- ESP32-S3 引腳定義 ---
#define I2C_SDA_PIN 8   // ESP32-S3 I2C SDA 引脚 for MAX30105 (I2C0)
#define I2C_SCL_PIN 7   // ESP32-S3 I2C SCL 引脚 for MAX30105 (I2C0)

// ESP32-S3 MUX 控制引腳
#define MUX_S0_PIN GPIO_NUM_35  // MUX 選擇線 S0
#define MUX_S1_PIN GPIO_NUM_36  // MUX 選擇線 S1
#define NUM_ADC_CHANNELS 8     // 單個 ADS1115 或 MUX 後的 ADC 通道數
#define NUM_MUX_SETTINGS 4     // MUX 設定數 (4-to-1)
#define TOTAL_PRESSURE_CHANNELS (NUM_ADC_CHANNELS * NUM_MUX_SETTINGS) // 總壓力通道數 (32)

// --- ESP32-S3 ADC 配置 (不再使用內部 ADC 進行壓力測量) ---
// ADC1 通道對應的 GPIO 引腳:
// ADC1_CH0 = GPIO1, ADC1_CH1 = GPIO2, ADC1_CH2 = GPIO3, ADC1_CH3 = GPIO4,
// ADC1_CH4 = GPIO5, ADC1_CH5 = GPIO6, ADC1_CH6 = GPIO7, ADC1_CH7 = GPIO8

// --- 佇列和資料結構定義 ---
#define QUEUE_LENGTH 5

typedef struct {
    int16_t values[TOTAL_PRESSURE_CHANNELS];
    int64_t timestamp;
} PressureData_t;

typedef struct {
    long ir_value;
    int64_t timestamp;
} IRData_t;

static QueueHandle_t pressure_data_queue;
static QueueHandle_t ir_data_queue;
// --- 結束佇列和資料結構定義 ---


// --- ADS1115 I2C 初始化函數 (供 ADS1115 任務使用) ---
static esp_err_t ads1115_i2c_master_init_internal(void) {
    ESP_LOGI(TAG_ADS, "Initializing I2C master for ADS1115 using new API...");
    i2c_master_bus_config_t i2c_mst_config = {}; 
    i2c_mst_config.clk_source = I2C_CLK_SRC_DEFAULT;
    i2c_mst_config.i2c_port = ADS1115_I2C_PORT_NUM;
    i2c_mst_config.scl_io_num = ADS1115_I2C_MASTER_SCL_IO;
    i2c_mst_config.sda_io_num = ADS1115_I2C_MASTER_SDA_IO;
    i2c_mst_config.glitch_ignore_cnt = 7;
    i2c_mst_config.intr_priority = 1; 
    i2c_mst_config.trans_queue_depth = 0; 
    i2c_mst_config.flags.enable_internal_pullup = true; 
    
    esp_err_t err = i2c_new_master_bus(&i2c_mst_config, &i2c_bus_handle_ads);
    if (err != ESP_OK) {
        ESP_LOGE(TAG_ADS, "Failed to create I2C master bus for ADS1115: %s", esp_err_to_name(err));
        return err;
    }
    ESP_LOGI(TAG_ADS, "I2C master bus for ADS1115 initialized.");

    i2c_device_config_t dev_cfg_1 = {}; 
    dev_cfg_1.dev_addr_length = I2C_ADDR_BIT_LEN_7;
    dev_cfg_1.device_address = ADS1115_ADDR;
    dev_cfg_1.scl_speed_hz = ADS1115_I2C_MASTER_FREQ_HZ;

    err = i2c_master_bus_add_device(i2c_bus_handle_ads, &dev_cfg_1, &ads1115_dev_handle_1);
    if (err != ESP_OK) {
        ESP_LOGE(TAG_ADS, "Failed to add ADS1115 device 1 (0x%02X): %s", ADS1115_ADDR, esp_err_to_name(err));
        i2c_del_master_bus(i2c_bus_handle_ads); 
        return err;
    }
    ESP_LOGI(TAG_ADS, "ADS1115 device 1 (0x%02X) added to bus.", ADS1115_ADDR);

    i2c_device_config_t dev_cfg_2 = {}; 
    dev_cfg_2.dev_addr_length = I2C_ADDR_BIT_LEN_7;
    dev_cfg_2.device_address = ADS1115_ADDR_2;
    dev_cfg_2.scl_speed_hz = ADS1115_I2C_MASTER_FREQ_HZ;

    err = i2c_master_bus_add_device(i2c_bus_handle_ads, &dev_cfg_2, &ads1115_dev_handle_2);
    if (err != ESP_OK) {
        ESP_LOGE(TAG_ADS, "Failed to add ADS1115 device 2 (0x%02X): %s", ADS1115_ADDR_2, esp_err_to_name(err));
        if (ads1115_dev_handle_1) i2c_master_bus_rm_device(ads1115_dev_handle_1);
        i2c_del_master_bus(i2c_bus_handle_ads);
        return err;
    }
    ESP_LOGI(TAG_ADS, "ADS1115 device 2 (0x%02X) added to bus.", ADS1115_ADDR_2);
    
    ESP_LOGI(TAG_ADS, "I2C master and ADS1115 devices initialized successfully.");
    return ESP_OK;
}
// --- 結束 ADS1115 I2C 初始化函數 ---

// --- ADS1115 讀取函數 (供 ADS1115 任務使用) ---
static int16_t read_ads1115_value_internal(i2c_master_dev_handle_t dev_handle, uint8_t ads_internal_channel_idx, const char* log_tag_dev_id) {
    if (dev_handle == NULL) {
        ESP_LOGE(TAG_ADS, "ADS1115 read: null device handle for %s, channel %d", log_tag_dev_id, ads_internal_channel_idx);
        return INT16_MIN;
    }

    uint8_t config_payload[3];
    config_payload[0] = ADS1115_REG_CONFIG;
    uint8_t mux_config_bits = 0b100 + ads_internal_channel_idx;
    config_payload[1] = (0x80 | (mux_config_bits << 4) | (0b001 << 1) | 0x01); 
    config_payload[2] = 0xE3; // DR = 860 SPS, Comparator disabled

    esp_err_t err = i2c_master_transmit(dev_handle, config_payload, sizeof(config_payload), 250);
    if (err != ESP_OK) {
        ESP_LOGE(TAG_ADS, "ADS1115 (%s) config write failed for channel %d: %s", log_tag_dev_id, ads_internal_channel_idx, esp_err_to_name(err));
        return INT16_MIN;
    }

    int64_t conversion_start_time = esp_timer_get_time();
    int64_t polling_timeout_us = 3000; 
    bool conversion_done = false;
    bool i2c_error_during_polling = false;
    uint8_t last_read_config_msb = 0xFF; 

    while (esp_timer_get_time() - conversion_start_time < polling_timeout_us) {
        uint8_t point_to_config_reg = ADS1115_REG_CONFIG;
        err = i2c_master_transmit(dev_handle, &point_to_config_reg, 1, 20); 
        if (err != ESP_OK) {
            ESP_LOGE(TAG_ADS, "ADS1115 (%s Ch%d) poll: set pointer to config reg failed: %s", log_tag_dev_id, ads_internal_channel_idx, esp_err_to_name(err));
            i2c_error_during_polling = true;
            break; 
        }

        uint8_t config_read_buffer[2];
        err = i2c_master_receive(dev_handle, config_read_buffer, 2, 20); 
        if (err != ESP_OK) {
            ESP_LOGE(TAG_ADS, "ADS1115 (%s Ch%d) poll: read config reg failed: %s", log_tag_dev_id, ads_internal_channel_idx, esp_err_to_name(err));
            i2c_error_during_polling = true;
            break; 
        }
        
        last_read_config_msb = config_read_buffer[0]; 
        // If OS bit is 1, it means the device is not busy (conversion is complete)
        if (last_read_config_msb & 0x80) { 
            conversion_done = true;
            break;
        }
        ets_delay_us(100); 
    }

    if (!conversion_done) {
        if (i2c_error_during_polling) {
            ESP_LOGE(TAG_ADS, "ADS1115 (%s Ch%d) conversion check failed due to I2C error during polling.", log_tag_dev_id, ads_internal_channel_idx);
        } else { 
            ESP_LOGE(TAG_ADS, "ADS1115 (%s Ch%d) conversion timeout after %lld us. OS bit still set. Last config MSB read: 0x%02X (Config written MSB was: 0x%02X)", 
                     log_tag_dev_id, ads_internal_channel_idx, polling_timeout_us, last_read_config_msb, config_payload[1]);
        }
        return INT16_MIN;
    }

    uint8_t read_buffer[2];
    uint8_t target_reg = ADS1115_REG_CONVERSION;
    err = i2c_master_transmit(dev_handle, &target_reg, 1, 100); 
    if (err != ESP_OK) {
        ESP_LOGE(TAG_ADS, "ADS1115 (%s) set read pointer to conversion reg failed for channel %d: %s", log_tag_dev_id, ads_internal_channel_idx, esp_err_to_name(err));
        return INT16_MIN;
    }

    err = i2c_master_receive(dev_handle, read_buffer, 2, 100); 
    if (err != ESP_OK) {
        ESP_LOGE(TAG_ADS, "ADS1115 (%s) read conversion failed for channel %d: %s", log_tag_dev_id, ads_internal_channel_idx, esp_err_to_name(err));
        return INT16_MIN;
    }
    return (int16_t)((read_buffer[0] << 8) | read_buffer[1]);
}
// --- 結束 ADS1115 讀取函數 ---

// --- GPIO 初始化函數 (供 ADS1115 任務使用) ---
static void init_mux_gpio_internal() {
    ESP_LOGI(TAG_ADS, "Initializing MUX GPIOs...");
    gpio_config_t io_conf = {};
    io_conf.intr_type = GPIO_INTR_DISABLE;
    io_conf.mode = GPIO_MODE_OUTPUT;
    io_conf.pin_bit_mask = (1ULL << MUX_S0_PIN) | (1ULL << MUX_S1_PIN);
    io_conf.pull_down_en = GPIO_PULLDOWN_DISABLE;
    io_conf.pull_up_en = GPIO_PULLUP_DISABLE;
    esp_err_t err = gpio_config(&io_conf);
    if (err != ESP_OK) {
        ESP_LOGE(TAG_ADS, "Failed to configure MUX GPIOs: %s", esp_err_to_name(err));
    } else {
        ESP_LOGI(TAG_ADS, "MUX GPIOs initialized successfully.");
        gpio_set_level(MUX_S0_PIN, 0);
        gpio_set_level(MUX_S1_PIN, 0);
    }
}
// --- 結束 GPIO 初始化函數 ---

// --- MUX 控制函數 (供 ADS1115 任務使用) ---
static void set_mux_channel_internal(uint8_t mux_setting) {
    gpio_set_level(MUX_S0_PIN, mux_setting & 0x01);
    gpio_set_level(MUX_S1_PIN, (mux_setting >> 1) & 0x01);
    // ESP_LOGD(TAG_ADS, "MUX 設置: S0=%d, S1=%d (mux_setting=%d)", mux_setting & 0x01, (mux_setting >> 1) & 0x01, mux_setting);
    ets_delay_us(500); 
}
// --- 結束 MUX 控制函數 ---

// --- ADS1115 讀取任務 ---
void ads1115_read_task(void *pvParameters) {
    ESP_LOGI(TAG_ADS, "ADS1115 Read Task Started.");
    init_mux_gpio_internal();
    esp_err_t ads_init_status = ads1115_i2c_master_init_internal();
    if (ads_init_status != ESP_OK) {
        ESP_LOGE(TAG_ADS, "ADS1115 I2C initialization failed. Task will not read pressure data.");
        // 在此處可以選擇刪除任務或進入一個安全的空閒狀態
        vTaskDelete(NULL);
        return;
    }

    PressureData_t p_data;
  // [實驗修改] 固定 MUX 為 0，之後不再切換
    set_mux_channel_internal(0); 
    while (1) {
        // ESP_LOGI(TAG_ADS, "Loop: Starting pressure scan...");
        p_data.timestamp = esp_timer_get_time();
        bool overall_read_success = true;
        // 1. 初始化所有數據為 0 (這樣沒讀到的通道就會自動輸出 0)
        for (int i = 0; i < TOTAL_PRESSURE_CHANNELS; i++) {
            p_data.values[i] = 0;
        }
        if (ads_init_status == ESP_OK) {
            // for (uint8_t mux_idx = 0; mux_idx < NUM_MUX_SETTINGS; mux_idx++) {
            for (uint8_t mux_idx = 0; mux_idx < 1; mux_idx++) {    
                // set_mux_channel_internal(mux_idx);
                
                // 虛擬讀取以穩定 MUX
                // if (ads1115_dev_handle_1) {
                //     volatile int16_t dummy_val_ads1_ain0 __attribute__((unused)) = read_ads1115_value_internal(ads1115_dev_handle_1, 0, "ADS1(0x48)-dummy");
                //     // ESP_LOGD(TAG_ADS, "Mux %d: Dummy read ADS1 AIN0: %d", mux_idx, dummy_val_ads1_ain0);
                // }
                // if (ads1115_dev_handle_2) {
                //     volatile int16_t dummy_val_ads2_ain0 __attribute__((unused)) = read_ads1115_value_internal(ads1115_dev_handle_2, 0, "ADS2(0x49)-dummy");
                //     // ESP_LOGD(TAG_ADS, "Mux %d: Dummy read ADS2 AIN0: %d", mux_idx, dummy_val_ads2_ain0);
                // }

                // for (int adc_idx = 0; adc_idx < NUM_ADC_CHANNELS; adc_idx++) {
                for (int adc_idx = 0; adc_idx < NUM_ADC_CHANNELS; adc_idx++) {    
                    int pressure_idx = mux_idx * NUM_ADC_CHANNELS + adc_idx;
                    i2c_master_dev_handle_t current_dev_handle = NULL;
                    const char* current_dev_log_tag = "UNKNOWN_ADS";
                    uint8_t ads_chip_channel_idx;

                    if (adc_idx < 4) {
                        current_dev_handle = ads1115_dev_handle_1;
                        current_dev_log_tag = "ADS1(0x48)";
                        ads_chip_channel_idx = adc_idx;
                    } else {
                        current_dev_handle = ads1115_dev_handle_2;
                        current_dev_log_tag = "ADS2(0x49)";
                        ads_chip_channel_idx = adc_idx - 4;
                    }

                    if (current_dev_handle != NULL) {
                        int16_t ads_value = read_ads1115_value_internal(current_dev_handle, ads_chip_channel_idx, current_dev_log_tag);
                        if (ads_value == INT16_MIN) {
                            p_data.values[pressure_idx] = -2; // 表示讀取錯誤
                            overall_read_success = false;
                        } else {
                            p_data.values[pressure_idx] = ads_value;
                        }
                    } else {
                        p_data.values[pressure_idx] = -4; // 表示設備控制代碼無效
                        overall_read_success = false;
                    }
                }
            }
        } else { // ads_init_status != ESP_OK
            for (int i = 0; i < TOTAL_PRESSURE_CHANNELS; i++) {
                p_data.values[i] = -3; // 表示 ADS 初始化失敗
            }
            overall_read_success = false;
        }
        
        if (overall_read_success) {
            ESP_LOGD(TAG_ADS, "Loop: Successfully read 32 pressure channels.");
        } else {
            ESP_LOGW(TAG_ADS, "Loop: Error reading some pressure channels.");
        }

        // ESP_LOGI(TAG_ADS, "Loop: Pressure scan complete. Attempting to send to queue...");
        if (xQueueSend(pressure_data_queue, &p_data, pdMS_TO_TICKS(10)) != pdPASS) {
            ESP_LOGW(TAG_ADS, "Loop: Failed to send pressure data to queue.");
        } else {
            // ESP_LOGI(TAG_ADS, "Loop: Successfully sent pressure data to queue.");
        }
        // ESP_LOGI(TAG_ADS, "Loop: Delaying task...");
        vTaskDelay(pdMS_TO_TICKS(1)); // 目標約 20ms 掃描一次 - 將延遲縮短以提高數據速率
        // ESP_LOGI(TAG_ADS, "Loop: Woke up from delay.");
    }
}
// --- 結束 ADS1115 讀取任務 ---

// --- MAX30105 讀取任務 ---
void max30105_read_task(void *pvParameters) {
    ESP_LOGI(TAG_MAX, "MAX30105 Read Task Started.");

    // 初始化 I2C for MAX30105 (I2C0)
    Wire.begin(I2C_SDA_PIN, I2C_SCL_PIN);
    ESP_LOGI(TAG_MAX, "I2C for MAX30105 initialized on I2C0.");

    vTaskDelay(pdMS_TO_TICKS(100)); // 確保 I2C 穩定

    if (!particleSensor.begin(Wire, I2C_SPEED_FAST)) {
        ESP_LOGE(TAG_MAX, "MAX30105 sensor not found! Check connections and power.");
        ESP_LOGE(TAG_MAX, "Terminating MAX30105 task.");
        vTaskDelete(NULL);
        return;
    }
    ESP_LOGI(TAG_MAX, "MAX30105 initialized successfully.");

    uint8_t ledBrightness = 0x7F;
    uint8_t sampleAverage = 1;
    uint8_t ledMode = 2;
    int sampleRate = 800;
    int pulseWidth = 215;
    int adcRange = 16384;

    particleSensor.setup(ledBrightness, sampleAverage, ledMode, sampleRate, pulseWidth, adcRange);
    particleSensor.enableDIETEMPRDY();
    particleSensor.setPulseAmplitudeRed(0x0A);
    particleSensor.setPulseAmplitudeGreen(0);
    ESP_LOGI(TAG_MAX, "MAX30105 configured. Starting data acquisition.");

    IRData_t ir_data;

    while (1) {
        // ESP_LOGI(TAG_MAX, "Loop: Attempting to get IR value...");
        ir_data.ir_value = particleSensor.getIR();
        ir_data.timestamp = esp_timer_get_time(); // Timestamp after getting value
        // ESP_LOGI(TAG_MAX, "Loop: Got IR value: %ld. Timestamp: %lld", ir_data.ir_value, ir_data.timestamp);

        // ESP_LOGI(TAG_MAX, "Loop: Attempting to send IR data to queue...");
        if (xQueueSend(ir_data_queue, &ir_data, pdMS_TO_TICKS(10)) != pdPASS) {
            ESP_LOGW(TAG_MAX, "Loop: Failed to send IR data to queue.");
        } else {
            // ESP_LOGI(TAG_MAX, "Loop: Successfully sent IR data to queue.");
        }
        // ESP_LOGI(TAG_MAX, "Loop: Delaying task...");
        vTaskDelay(pdMS_TO_TICKS(50)); // 嘗試匹配壓力感測器的速率
        // ESP_LOGI(TAG_MAX, "Loop: Woke up from delay.");
    }
}
// --- 結束 MAX30105 讀取任務 ---

// --- 數據輸出任務 ---
void output_task(void *pvParameters) {
    ESP_LOGI(TAG_OUT, "Output Task Started.");
    PressureData_t p_data;
    IRData_t ir_d;
    char output_buffer[512]; // 增加到 512 以確保足夠空間

    // 初始化數據結構以避免使用未初始化的值
    for(int i=0; i<TOTAL_PRESSURE_CHANNELS; ++i) p_data.values[i] = -99; // 預設錯誤值
    p_data.timestamp = 0;
    ir_d.ir_value = -99; // 預設錯誤值
    ir_d.timestamp = 0;

    bool p_data_fresh = false;

    while (1) {
        // ESP_LOGI(TAG_OUT, "Loop: Waiting for data...");
        bool received_pressure_this_cycle = false;
        bool received_ir_this_cycle = false;

        if (xQueueReceive(pressure_data_queue, &p_data, pdMS_TO_TICKS(5)) == pdPASS) {
            ESP_LOGI(TAG_OUT, "Loop: Received pressure data. Timestamp: %lld", p_data.timestamp);
            p_data_fresh = true;
            received_pressure_this_cycle = true;
        }
        if (xQueueReceive(ir_data_queue, &ir_d, pdMS_TO_TICKS(5)) == pdPASS) {
            ESP_LOGI(TAG_OUT, "Loop: Received IR data. Value: %ld, Timestamp: %lld", ir_d.ir_value, ir_d.timestamp);
            received_ir_this_cycle = true;
        }

        if (!received_pressure_this_cycle && !received_ir_this_cycle) {
            ESP_LOGD(TAG_OUT, "Loop: No new data received from queues in this cycle's non-blocking check.");
        }

        if (p_data_fresh) {
            // ESP_LOGI(TAG_OUT, "Loop: p_data_fresh is true. Preparing output string...");
            int offset = 0;
            for (int i = 0; i < TOTAL_PRESSURE_CHANNELS; i++) {
                offset += snprintf(output_buffer + offset, sizeof(output_buffer) - offset, "%d,", p_data.values[i]);
                if (offset >= sizeof(output_buffer)) {
                    ESP_LOGE(TAG_OUT, "Output buffer overflow when printing pressure readings!");
                    break; 
                }
            }
            if (offset < sizeof(output_buffer)) {
                 // 使用壓力數據的時間戳作為主要時間戳
                snprintf(output_buffer + offset, sizeof(output_buffer) - offset, "%ld,%lld\n", ir_d.ir_value, p_data.timestamp);
            } else {
                 ESP_LOGW(TAG_OUT, "Output buffer full before IR/Timestamp.");
            }
            printf("%s", output_buffer);
            // ESP_LOGI(TAG_OUT, "Loop: Output printed.");
            p_data_fresh = false; // 重置標誌，等待下一次新的壓力數據
        } else {
            ESP_LOGD(TAG_OUT, "Loop: p_data_fresh is false. No output this cycle.");
        }
        
        // ESP_LOGI(TAG_OUT, "Loop: Delaying task...");
        vTaskDelay(pdMS_TO_TICKS(10)); // 保持一個較小的延遲，讓其他任務運行
        // ESP_LOGI(TAG_OUT, "Loop: Woke up from delay.");
    }
}
// --- 結束數據輸出任務 ---


extern "C" void app_main(void) {
    printf("系統初始化開始，晶片：ESP32-S3\n");
    ESP_LOGI(TAG_MAIN, "Current portTICK_PERIOD_MS: %d", (int)portTICK_PERIOD_MS);

    uart_set_baudrate(UART_NUM_0, 921600);
    ESP_LOGI(TAG_MAIN, "UART0 baudrate set to 921600");

    // 創建佇列
    pressure_data_queue = xQueueCreate(QUEUE_LENGTH, sizeof(PressureData_t));
    ir_data_queue = xQueueCreate(QUEUE_LENGTH, sizeof(IRData_t));

    if (pressure_data_queue == NULL || ir_data_queue == NULL) {
        ESP_LOGE(TAG_MAIN, "Failed to create queues. Halting.");
        while(1);
    }
    ESP_LOGI(TAG_MAIN, "Queues created successfully.");

    // 創建任務
    // 堆疊大小參考：原 max30105_task 使用 8192。I2C 和感測器操作可能需要較多堆疊。
    // 輸出任務主要做字串格式化，4096 應該足夠。
    xTaskCreate(ads1115_read_task, "ads1115_read_task", 8192, NULL, 6, NULL);
    xTaskCreate(max30105_read_task, "max30105_read_task", 8192, NULL, 5, NULL);
    xTaskCreate(output_task, "output_task", 4096, NULL, 4, NULL);

    ESP_LOGI(TAG_MAIN, "All tasks created. Main thread finishing.");
    // app_main 可以結束，FreeRTOS 排程器將接管任務執行
}
