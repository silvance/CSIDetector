// CSI transmitter: brings up a SoftAP on a fixed channel and emits a UDP
// broadcast packet at a steady rate. Each broadcast generates one CSI sample
// on the receiver, so this rate is the effective CSI sample rate.

#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "esp_event.h"
#include "esp_netif.h"
#include "esp_wifi.h"
#include "nvs_flash.h"
#include "lwip/sockets.h"

static const char *TAG = "csi_tx";

static void wifi_init_softap(void) {
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    esp_netif_create_default_wifi_ap();

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));

    wifi_config_t wifi_config = {
        .ap = {
            .ssid = CONFIG_CSI_TX_AP_SSID,
            .ssid_len = strlen(CONFIG_CSI_TX_AP_SSID),
            .channel = CONFIG_CSI_TX_AP_CHANNEL,
            .password = CONFIG_CSI_TX_AP_PASSWORD,
            .max_connection = 4,
            .authmode = WIFI_AUTH_WPA2_PSK,
        },
    };
    if (strlen(CONFIG_CSI_TX_AP_PASSWORD) == 0) {
        wifi_config.ap.authmode = WIFI_AUTH_OPEN;
    }

    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_AP));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_AP, &wifi_config));
    ESP_ERROR_CHECK(esp_wifi_start());

    // Lock the long-guard interval / disable rate adaptation drift so CSI
    // samples are comparable across time.
    ESP_ERROR_CHECK(esp_wifi_config_11b_rate(WIFI_IF_AP, true));
}

static void broadcast_task(void *arg) {
    const int rate_hz = CONFIG_CSI_TX_BEACON_RATE_HZ;
    const TickType_t period = pdMS_TO_TICKS(1000 / rate_hz);

    int sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (sock < 0) {
        ESP_LOGE(TAG, "socket: errno %d", errno);
        vTaskDelete(NULL);
        return;
    }
    int broadcast = 1;
    setsockopt(sock, SOL_SOCKET, SO_BROADCAST, &broadcast, sizeof(broadcast));

    struct sockaddr_in dst = {
        .sin_family = AF_INET,
        .sin_port = htons(5005),
        .sin_addr.s_addr = htonl(INADDR_BROADCAST),
    };

    uint32_t seq = 0;
    char payload[32];
    TickType_t next = xTaskGetTickCount();
    while (1) {
        int n = snprintf(payload, sizeof(payload), "csi-ping %lu", (unsigned long)seq++);
        sendto(sock, payload, n, 0, (struct sockaddr *)&dst, sizeof(dst));
        vTaskDelayUntil(&next, period);
    }
}

void app_main(void) {
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);

    wifi_init_softap();
    ESP_LOGI(TAG, "SoftAP up: ssid=%s channel=%d rate=%dHz",
             CONFIG_CSI_TX_AP_SSID, CONFIG_CSI_TX_AP_CHANNEL,
             CONFIG_CSI_TX_BEACON_RATE_HZ);

    xTaskCreate(broadcast_task, "broadcast", 4096, NULL, 5, NULL);
}
