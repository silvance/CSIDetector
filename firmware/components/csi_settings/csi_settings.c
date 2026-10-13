// Reads the csi_cfg partition written by the setup app.

#include "csi_settings.h"

#include <stdlib.h>
#include "esp_log.h"
#include "esp_partition.h"

static const char *TAG = "csi_settings";

void csi_settings_load(csi_settings_t *s) {
    const esp_partition_t *part = esp_partition_find_first(
        ESP_PARTITION_TYPE_DATA, (esp_partition_subtype_t)CSI_SETTINGS_SUBTYPE,
        CSI_SETTINGS_LABEL);
    if (!part) {
        ESP_LOGW(TAG, "no %s partition: using the built-in defaults", CSI_SETTINGS_LABEL);
        return;
    }
    size_t len = part->size < CSI_SETTINGS_MAX_SIZE ? part->size : CSI_SETTINGS_MAX_SIZE;
    uint8_t *buf = malloc(len);
    if (!buf) {
        ESP_LOGE(TAG, "out of memory reading the settings: using the built-in defaults");
        return;
    }
    esp_err_t err = esp_partition_read(part, 0, buf, len);
    int rc = err == ESP_OK ? csi_settings_parse(buf, len, s) : CSI_SETTINGS_ERR_EMPTY;
    free(buf);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "reading %s failed (%s): using the built-in defaults",
                 CSI_SETTINGS_LABEL, esp_err_to_name(err));
    } else if (rc == CSI_SETTINGS_OK) {
        ESP_LOGI(TAG, "settings from the setup app%s%s%s (they take precedence over "
                      "menuconfig; idf.py erase-flash clears them)",
                 s->label[0] ? " for \"" : "", s->label, s->label[0] ? "\"" : "");
    } else if (rc == CSI_SETTINGS_ERR_EMPTY) {
        ESP_LOGI(TAG, "no settings from the setup app: using the built-in defaults");
    } else {
        ESP_LOGE(TAG, "settings block %s: using the built-in defaults. "
                      "Flash this board again from the setup app.",
                 csi_settings_strerror(rc));
    }
}
