#include "Wire.h"
#include "esp_log.h"
#include <string.h>
#include "freertos/FreeRTOS.h" // Added for pdMS_TO_TICKS
#include "freertos/task.h"   // Added for pdMS_TO_TICKS (often included with FreeRTOS.h but good to be explicit)

static const char* TAG = "Wire";

// 默認 I2C 引腳
#define I2C_DEFAULT_SDA_PIN 21 // ESP32 default I2C0 pins
#define I2C_DEFAULT_SCL_PIN 22

// 全局實例化 - 確保這個全局變量被正確定義
TwoWire Wire;

TwoWire::TwoWire() : i2c_bus_handle(nullptr), i2c_dev_handle(nullptr), current_address(0), 
                     rxIndex(0), rxLength(0), txLength(0), 
                     current_sda_pin(-1), current_scl_pin(-1), current_clk_speed(0) {
    // Constructor
}

TwoWire::~TwoWire() {
    end();
}

void TwoWire::begin(int sda, int scl, uint32_t frequency) {
    if (sda < 0) sda = I2C_DEFAULT_SDA_PIN;
    if (scl < 0) scl = I2C_DEFAULT_SCL_PIN;

    ESP_LOGI(TAG, "Initializing I2C0 (SDA:%d, SCL:%d, Freq:%lu Hz) with new API", sda, scl, frequency);

    if (sda == scl) {
        ESP_LOGE(TAG, "SDA and SCL pins cannot be the same");
        return;
    }

    current_sda_pin = sda;
    current_scl_pin = scl;
    current_clk_speed = frequency;

    if (i2c_bus_handle) {
        ESP_LOGW(TAG, "I2C bus already initialized. Deleting existing bus handle.");
        // Potentially add device removal from bus if devices were added
        if (i2c_dev_handle) {
             i2c_master_bus_rm_device(i2c_dev_handle);
             i2c_dev_handle = nullptr;
        }
        i2c_del_master_bus(i2c_bus_handle);
        i2c_bus_handle = nullptr;
    }

    i2c_master_bus_config_t i2c_mst_config = {
        .i2c_port = I2C_NUM_0, // Use I2C_NUM_0 for MAX30105
        .sda_io_num = (gpio_num_t)sda,
        .scl_io_num = (gpio_num_t)scl,
        .clk_source = I2C_CLK_SRC_DEFAULT,
        .glitch_ignore_cnt = 7,
        .intr_priority = 0, // Initialize intr_priority to a default value (e.g., 0 for no interrupt or a specific priority)
        .trans_queue_depth = 0, // Initialize trans_queue_depth (0 for default)
        .flags = {
            .enable_internal_pullup = true
        }
    };
    // .pclk_hz = 0, // Auto-detect

    esp_err_t err = i2c_new_master_bus(&i2c_mst_config, &i2c_bus_handle);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Failed to create I2C master bus: %s", esp_err_to_name(err));
        i2c_bus_handle = nullptr;
        return;
    }

    ESP_LOGI(TAG, "I2C master bus (I2C0) initialized successfully.");
    // Device handle will be created/updated in beginTransmission or requestFrom
}

void TwoWire::end() {
    ESP_LOGI(TAG, "Ending I2C communication.");
    if (i2c_dev_handle) {
        esp_err_t err = i2c_master_bus_rm_device(i2c_dev_handle);
        if (err != ESP_OK) {
            ESP_LOGW(TAG, "Failed to remove I2C device: %s", esp_err_to_name(err));
        }
        i2c_dev_handle = nullptr;
    }
    if (i2c_bus_handle) {
        esp_err_t err = i2c_del_master_bus(i2c_bus_handle);
        if (err != ESP_OK) {
            ESP_LOGW(TAG, "Failed to delete I2C master bus: %s", esp_err_to_name(err));
        }
        i2c_bus_handle = nullptr;
    }
    current_sda_pin = -1;
    current_scl_pin = -1;
    current_clk_speed = 0;
}

void TwoWire::setClock(uint32_t frequency) {
    ESP_LOGI(TAG, "SetClock called with frequency %lu Hz", frequency);
    if (!i2c_bus_handle) {
        ESP_LOGE(TAG, "I2C bus not initialized. Call begin() first.");
        return;
    }
    if (current_sda_pin < 0 || current_scl_pin < 0) {
         ESP_LOGE(TAG, "SDA/SCL pins not set. Call begin() first.");
        return;
    }

    // With the new API, bus frequency is part of the bus config.
    // To change it, we might need to delete and re-create the bus.
    // Or, if a device is already added, this might not be easily changeable without re-adding the device.
    // For simplicity, we'll re-initialize if the frequency changes.
    // This is a simplification; a more robust implementation might handle device re-attachment.
    if (frequency != current_clk_speed) {
        ESP_LOGI(TAG, "Re-initializing I2C bus for new frequency: %lu Hz", frequency);
        end(); // Clean up existing bus and device handles
        begin(current_sda_pin, current_scl_pin, frequency); // Re-initialize with new frequency
    } else {
        ESP_LOGI(TAG, "Frequency %lu Hz is already set.", frequency);
    }
}

// Helper to ensure device handle is ready for a specific address
static esp_err_t ensure_device_handle(TwoWire* wire, uint8_t address, uint32_t clk_speed, 
                                      i2c_master_bus_handle_t bus_handle, 
                                      i2c_master_dev_handle_t* dev_handle,
                                      uint8_t* current_i2c_address) {
    if (*dev_handle && *current_i2c_address == address) {
        return ESP_OK; // Already configured for this address
    }
    if (*dev_handle) { // Device handle exists but for a different address
        i2c_master_bus_rm_device(*dev_handle);
        *dev_handle = nullptr;
    }

    i2c_device_config_t dev_cfg = {
        .dev_addr_length = I2C_ADDR_BIT_LEN_7,
        .device_address = address,
        .scl_speed_hz = clk_speed,
        .scl_wait_us = 0,
        .flags = {0} // Initialize flags
    };

    *current_i2c_address = address; // Update current address before attempting to add
    esp_err_t err = i2c_master_bus_add_device(bus_handle, &dev_cfg, dev_handle);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Failed to add I2C device 0x%02X: %s", address, esp_err_to_name(err));
        *dev_handle = nullptr; // Ensure handle is null on failure
    }
    return err;
}

bool TwoWire::beginTransmission(uint8_t address) {
    if (!i2c_bus_handle) {
        ESP_LOGE(TAG, "I2C bus not initialized in beginTransmission. Call begin() first.");
        return false;
    }
    slaveAddress = address;
    txLength = 0;

    // Ensure the device handle is ready for this transaction
    // The actual device addition to the bus happens here if not already done for this address
    esp_err_t err = ensure_device_handle(this, slaveAddress, current_clk_speed, i2c_bus_handle, &i2c_dev_handle, &current_address);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Device setup failed for 0x%02X in beginTransmission", slaveAddress);
        return false;
    }
    return true;
}

size_t TwoWire::write(uint8_t data) {
    if (txLength >= sizeof(txBuffer)) {
        ESP_LOGE(TAG, "TX buffer overflow");
        return 0;
    }
    txBuffer[txLength++] = data;
    return 1;
}

size_t TwoWire::write(const uint8_t* data, size_t len) {
    if (txLength + len > sizeof(txBuffer)) {
        ESP_LOGE(TAG, "TX buffer would overflow, limiting length");
        len = sizeof(txBuffer) - txLength;
    }
    if (len > 0) {
        memcpy(txBuffer + txLength, data, len);
        txLength += len;
    }
    return len;
}

uint8_t TwoWire::endTransmission(bool stopBit) {
    // stopBit is implicitly handled by the new API; transactions are generally self-contained.
    // The new API's i2c_master_transmit sends a START, address, data, and STOP.
    if (!i2c_dev_handle) {
        ESP_LOGE(TAG, "I2C device handle not available in endTransmission for address 0x%02X.", slaveAddress);
        return 4; // Generic error
    }

    if (txLength == 0) { // Some libraries might call endTransmission without writing data (e.g. to check for device presence)
        // Perform a quick check by trying to transmit 0 bytes. 
        // This is not a perfect presence check but can reveal basic communication issues.
        uint8_t dummy_byte = 0; // Initialize dummy_byte
        esp_err_t err = i2c_master_transmit(i2c_dev_handle, &dummy_byte, 0, pdMS_TO_TICKS(I2C_TIMEOUT_MS));
        txLength = 0; // Reset buffer length
        if (err == ESP_OK) return 0; // Success (device ACKed its address)
        if (err == ESP_ERR_TIMEOUT) { ESP_LOGW(TAG, "I2C timeout on 0-byte write (device check) for 0x%02X", slaveAddress); return 2;}
        ESP_LOGW(TAG, "I2C error %s on 0-byte write (device check) for 0x%02X", esp_err_to_name(err), slaveAddress); return 4;
    }

    esp_err_t err = i2c_master_transmit(i2c_dev_handle, txBuffer, txLength, pdMS_TO_TICKS(I2C_TIMEOUT_MS));
    txLength = 0; // Reset buffer length

    if (err == ESP_OK) {
        return 0; // Success
    } else if (err == ESP_ERR_TIMEOUT) {
        ESP_LOGE(TAG, "I2C transmission timeout for 0x%02X", slaveAddress);
        return 2; // Corresponds to old API's timeout error
    } else if (err == ESP_FAIL && slaveAddress != 0) { // ESP_FAIL can mean NACK on address phase
        ESP_LOGE(TAG, "I2C transmission NACK or error for 0x%02X: %s", slaveAddress, esp_err_to_name(err));
        return 3; // NACK on data (though new API might not distinguish as clearly as old)
    } else {
        ESP_LOGE(TAG, "I2C transmission error for 0x%02X: %s", slaveAddress, esp_err_to_name(err));
        return 4; // Other error
    }
}

uint8_t TwoWire::requestFrom(uint8_t address, uint8_t quantity, bool stopBit) {
    // stopBit is implicitly handled by the new API.
    if (!i2c_bus_handle) {
        ESP_LOGE(TAG, "I2C bus not initialized in requestFrom. Call begin() first.");
        return 0;
    }
    if (quantity == 0) {
        rxLength = 0;
        rxIndex = 0;
        return 0;
    }
    if (quantity > sizeof(rxBuffer)) {
        ESP_LOGW(TAG, "Requested quantity %u exceeds rxBuffer size %u, truncating.", quantity, (unsigned int)sizeof(rxBuffer));
        quantity = sizeof(rxBuffer);
    }

    slaveAddress = address;
    // Ensure the device handle is ready for this transaction
    esp_err_t err = ensure_device_handle(this, slaveAddress, current_clk_speed, i2c_bus_handle, &i2c_dev_handle, &current_address);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Device setup failed for 0x%02X in requestFrom", slaveAddress);
        rxLength = 0;
        rxIndex = 0;
        return 0;
    }
    
    if (!i2c_dev_handle) {
         ESP_LOGE(TAG, "I2C device handle not available in requestFrom for address 0x%02X.", slaveAddress);
        rxLength = 0;
        rxIndex = 0;
        return 0;
    }

    err = i2c_master_receive(i2c_dev_handle, rxBuffer, quantity, pdMS_TO_TICKS(I2C_TIMEOUT_MS));

    if (err == ESP_OK) {
        rxLength = quantity;
        rxIndex = 0;
        return quantity;
    } else {
        ESP_LOGE(TAG, "I2C receive error for 0x%02X: %s", slaveAddress, esp_err_to_name(err));
        rxLength = 0;
        rxIndex = 0;
        return 0;
    }
}

int TwoWire::available() {
    return rxLength - rxIndex;
}

int TwoWire::read() {
    if (rxIndex < rxLength) {
        return rxBuffer[rxIndex++];
    }
    return -1;
}