idf_component_register(SRCS "hello_world_main.cpp"
                         "MAX30105.cpp"
                         "heartRate.cpp"
                         "spo2_algorithm.cpp"
                         "Wire.cpp"
                    INCLUDE_DIRS "."
                    REQUIRES esp-tflite-micro)
