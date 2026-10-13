idf_component_register(
    SRCS "csi_settings.c" "csi_settings_parse.c"
    INCLUDE_DIRS "include"
    REQUIRES esp_partition
)
