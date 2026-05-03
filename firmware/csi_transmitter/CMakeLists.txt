cmake_minimum_required(VERSION 3.16)
include($ENV{IDF_PATH}/tools/cmake/project.cmake)

# esp_now_set_peer_rate_config (used to pin broadcasts to HT20 MCS0
# so they carry HT-LTF) was added in IDF v5.1; the receiver needs
# v5.3+ for working CSI callbacks, and we build both projects against
# the same IDF, so enforce v5.3 here too for consistency.
if(IDF_VERSION_MAJOR LESS 5 OR
   (IDF_VERSION_MAJOR EQUAL 5 AND IDF_VERSION_MINOR LESS 3))
    message(FATAL_ERROR
        "CSIDetector requires ESP-IDF >= 5.3 (got ${IDF_VERSION}).")
endif()

project(csi_transmitter)
