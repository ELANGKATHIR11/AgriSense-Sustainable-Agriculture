/*
  AgriSense ESP32 Firmware Stub
  - Minimal example to compile and connect to WiFi
  - Replace with real sensor code in agrisense_esp32.ino
*/

#include <WiFi.h>

const char* ssid = "YOUR_SSID";
const char* password = "YOUR_PASSWORD";

void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println("AgriSense ESP32 stub starting...");

  WiFi.begin(ssid, password);
  Serial.print("Connecting to WiFi");
  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 20) {
    delay(500);
    Serial.print('.');
    attempts++;
  }

  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\nWiFi connected");
    Serial.print("IP: ");
    Serial.println(WiFi.localIP());
  } else {
    Serial.println("\nWiFi connection failed (stub)");
  }
}

void loop() {
  // Placeholder: read sensors, publish to backend
  delay(10000);
}
