#ifndef WIRE_H
#define WIRE_H

#include <stdint.h>
#include <stddef.h>
#include "driver/i2c_master.h" // New API

// I2C 配置
#define I2C_NUM         I2C_NUM_0 // Keep for logical port, though new API handles it differently
#define I2C_SPEED_FAST  400000  // 400KHz
#define I2C_TIMEOUT_MS  1000    // 超時時間 (used as ticks for new API)

class TwoWire {
public:
    TwoWire();
    ~TwoWire();
    
    void begin(int sda = -1, int scl = -1, uint32_t frequency = I2C_SPEED_FAST); // Added frequency
    void end();
    
    bool beginTransmission(uint8_t address);
    uint8_t endTransmission(bool stopBit = true); // stopBit might be less relevant or handled differently
    
    void setClock(uint32_t frequency);
    
    size_t write(uint8_t data);
    size_t write(const uint8_t* data, size_t len);
    
    uint8_t requestFrom(uint8_t address, uint8_t quantity, bool stopBit = true); // stopBit might be less relevant
    int available();
    int read();
    
private:
    i2c_master_bus_handle_t i2c_bus_handle; // New API
    i2c_master_dev_handle_t i2c_dev_handle; // New API
    uint8_t current_address; // To track current device address for dev_handle management

    uint8_t slaveAddress; // Still used to store target address
    
    uint8_t rxBuffer[128];
    uint8_t txBuffer[128];
    uint8_t rxIndex;
    uint8_t rxLength;
    uint8_t txLength;
    
    int current_sda_pin;
    int current_scl_pin;
    uint32_t current_clk_speed; // Store current clock speed
};

extern TwoWire Wire;

#endif // WIRE_H