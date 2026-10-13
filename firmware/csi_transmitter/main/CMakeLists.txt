idf_component_register(
    SRCS "main.c"
    INCLUDE_DIRS "."
    REQUIRES csi_settings nvs_flash esp_wifi esp_event esp_netif
)
