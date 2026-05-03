cmake_minimum_required(VERSION 3.16)
include($ENV{IDF_PATH}/tools/cmake/project.cmake)

# CSI rx callback in IDF v5.2.x silently never fires on some chip
# revisions; v5.3+ is required. Fail the configure step early instead
# of letting the firmware boot and then go silent at runtime.
if(IDF_VERSION_MAJOR LESS 5 OR
   (IDF_VERSION_MAJOR EQUAL 5 AND IDF_VERSION_MINOR LESS 3))
    message(FATAL_ERROR
        "CSIDetector requires ESP-IDF >= 5.3 (got ${IDF_VERSION}). "
        "Update with: cd $ENV{IDF_PATH} && git checkout release/v5.5 && "
        "git submodule update --init --recursive && ./install.sh esp32s3")
endif()

project(csi_receiver)
