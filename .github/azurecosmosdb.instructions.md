---
applyTo: '**'
---
# Azure Cosmos DB Instructions for AgriSense Project

## 1. Data Modeling Best Practices for AgriSense
- **Model for Access Patterns**:
  - **Sensor Readings**: Write-heavy, query by `deviceId` + `timestamp`.
  - **Crop Data**: Read-heavy, query by `cropType` or `region`.
- **Hierarchy & Embedding**:
  - Embed **Daily Aggregates** (min/max/avg) within the Device or Crop document where possible to reduce read operations.
  - Keep item size under 2MB.

## 2. Partition Key Strategy
AgriSense relies on efficient scaling for thousands of IoT sensors.
- **SensorData Container**:
  - **Partition Key**: `/deviceId` (Logical partition per device).
  - **Why**: Most queries filter by device ("Show me status of Device A").
- **Users Container**:
  - **Partition Key**: `/email` or `/userId`.
- **Alerts Container**:
  - **Partition Key**: `/severity` (if querying by severity) OR `/deviceId` (if querying alerts per device). Recommend `/deviceId`.

## 3. SDK Usage (Python Async)
Always use the `azure-cosmos` async client to prevent blocking the FastAPI event loop.

```python
from azure.cosmos.aio import CosmosClient
from azure.cosmos import PartitionKey

# Async Singleton Pattern
class CosmosDB:
    _instance = None
    
    @classmethod
    async def get_instance(cls):
        if not cls._instance:
            cls._instance = CosmosClient(url=URL, credential=KEY)
        return cls._instance

async def get_sensor_data(device_id: str):
    client = await CosmosDB.get_instance()
    database = client.get_database_client(DATABASE_NAME)
    container = database.get_container_client("SensorData")
    
    # Query with Partition Key
    query = "SELECT * FROM c WHERE c.deviceId = @deviceId"
    params = [{"name": "@deviceId", "value": device_id}]
    
    # Enable cross-partition query ONLY if necessary (avoid for single device lookups)
    items = container.query_items(
        query=query,
        parameters=params,
        partition_key=device_id
    )
    return [item async for item in items]
```

## 4. Emulator & Development
- **Local Dev**: Use Azure Cosmos DB Emulator.
- **Connection String**: `AccountEndpoint=https://localhost:8081/;AccountKey=C2y6yDjf5/R+ob0N8A7Cgv30VRDJIWEHLM+4QDU5DE2nQ9nDuVTqobD4b8mGGyPMbIZnqyMsEcaGQy67XIw/Jw==`
- **SSL**: Ensure root certificate is installed or verify SSL is disabled for local dev if safe.

## 5. Implementation for AgriSense
### AI/Chat/Contextual Applications
- Store user chat history in a `ChatHistory` container.
- Partition by `/userId`.
- Use "Vector Search" (if available in Cosmos MongoDB API vCore or NoSQL Vector store) for RAG context retrieval.

### IoT Scenarios
- Store raw telemetry (temp, humidity, soil moisture) in `SensorData`.
- Use Time-to-Live (TTL) on raw data (e.g., 90 days) to auto-delete old records, while keeping aggregated summaries in a separate container/item indefinitely.

## 6. Throughput & Cost
- Start with **Autoscale** throughput for unpredictable IoT workloads.
- Monitor `429` (Request Rate Too Large) errors in logs.
- Retry logic is built into the SDK, but ensure your application handles timeouts gracefully.
