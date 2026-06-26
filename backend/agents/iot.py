from backend.agents.base_agent import BaseAgent

class IoTAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="IoTAgent",
            role="IoT Engineer",
            skills=["ESP32 firmware", "GPIO layout", "C++ Embedded programming"]
        )

class MQTTAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="MQTTAgent",
            role="MQTT Engineer",
            skills=["Broker setup", "Topic layout", "Message queue routing"]
        )

class EdgeAIAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="EdgeAIAgent",
            role="Edge AI Specialist",
            skills=["Raspberry Pi setup", "Model quantization", "Edge TPU compilation"]
        )
