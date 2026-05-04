cmake_minimum_required(VERSION 3.16)
include($ENV{IDF_PATH}/tools/cmake/project.cmake)

# ESP32-C5 target support is stable from IDF v5.4. v5.5+ recommended.
if(IDF_VERSION_MAJOR LESS 5 OR
   (IDF_VERSION_MAJOR EQUAL 5 AND IDF_VERSION_MINOR LESS 4))
    message(FATAL_ERROR
        "c5_display requires ESP-IDF >= 5.4 for ESP32-C5 target support "
        "(got ${IDF_VERSION}).")
endif()

project(c5_display)
