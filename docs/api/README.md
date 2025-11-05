# API Documentation

## Base URL

```
http://localhost:8000  # Local development
https://api.graphrag-crypto.railway.app  # Production
```

## Authentication

> **Note**: Currently, the API is running without authentication in development mode. For production, implement JWT authentication.

## Rate Limiting

- 100 requests per minute per IP (local development)
- 1000 requests per minute per API key (production)

## Health Check

```http
GET /health
```

**Response:**

```json
{
  "status": "healthy",
  "timestamp": "2025-11-04T13:30:00Z",
  "services": {
    "neo4j": "connected",
    "redis": "connected",
    "timescaledb": "connected"
  }
}
```

## Data Ingestion Endpoints

### Start Real-time Feed

```http
POST /ingest/start
```

**Request Body:**

```json
{
  "symbols": ["BTC/USDT", "ETH/USDT"],
  "interval": 60
}
```

**Response:**

```json
{
  "status": "success",
  "message": "Started real-time feed for BTC/USDT, ETH/USDT with 60s interval"
}
```

### Stop Real-time Feed

```http
POST /ingest/stop
```

**Response:**

```json
{
  "status": "success",
  "message": "Stopped real-time feed"
}
```

### Ingest Historical Data

```http
POST /ingest/historical
```

**Query Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| symbol | string | Yes | - | Trading pair (e.g., BTC/USDT) |
| days | integer | No | 30 | Number of days of historical data to fetch |
| timeframe | string | No | 1h | Candle timeframe (1m, 5m, 15m, 1h, 1d) |
| exchange | string | No | binance | Exchange to fetch data from |

**Response:**

```json
{
  "status": "success",
  "message": "Successfully ingested 30 days of 1h data for BTC/USDT"
}
```

## Technical Analysis Endpoints

### Calculate Technical Indicator

```http
POST /analyze/technical
```

**Request Body:**

```json
{
  "symbol": "BTC/USDT",
  "indicator": "rsi",
  "params": {
    "timeframe": "1h",
    "period": 14
  }
}
```

**Response:**

```json
{
  "symbol": "BTC/USDT",
  "indicator": "rsi",
  "value": 65.42,
  "metadata": {
    "period": 14,
    "timeframe": "1h",
    "last_updated": "2025-11-04T13:30:00Z"
  }
}
```

## Error Responses

### 400 Bad Request

```json
{
  "detail": "Invalid symbol format. Expected format: BASE/QUOTE (e.g., BTC/USDT)"
}
```

### 500 Internal Server Error

```json
{
  "detail": "Failed to connect to the exchange. Please try again later."
}
```

### 2. Graph Queries

#### Run Cypher Query

```http
POST /v1/graph/query
```

**Request Body:**

```json
{
  "query": "MATCH (c:Cryptocurrency {symbol: $symbol}) RETURN c",
  "parameters": {
    "symbol": "BTC"
  }
}
```

**Response:**

```json
{
  "results": [
    {
      "c": {
        "symbol": "BTC",
        "name": "Bitcoin",
        "market_cap": 1.2e12
      }
    }
  ],
  "stats": {
    "nodes_created": 0,
    "relationships_created": 0,
    "query_time_ms": 24
  }
}
```

### 3. Agent Endpoints

#### Get Prediction

```http
POST /v1/agents/predict
```

**Request Body:**

```json
{
  "symbol": "BTC/USDT",
  "horizon_days": 7,
  "indicators": ["rsi", "macd", "bollinger_bands"]
}
```

**Response:**

```json
{
  "symbol": "BTC/USDT",
  "prediction": {
    "price_target": 69210.50,
    "confidence": 0.85,
    "horizon_days": 7,
    "indicators": {
      "rsi": 68.5,
      "macd": {"value": 124.3, "signal": 118.7, "histogram": 5.6},
      "bollinger_bands": {"upper": 69870.2, "middle": 67432.1, "lower": 64993.9}
    },
    "timestamp": "2023-10-15T14:30:00Z"
  }
}
```

### 4. System Health

#### Health Check

```http
GET /v1/health
```

**Response:**

```json
{
  "status": "healthy",
  "timestamp": "2023-10-15T14:30:00Z",
  "services": {
    "graphiti": {
      "status": "ok",
      "version": "4.4.0"
    },
    "timescaledb": {
      "status": "ok",
      "version": "14.0.0"
    },
    "redis": {
      "status": "ok",
      "version": "7.0.0"
    },
    "together_ai": {
      "status": "ok",
      "model": "togethercomputer/llama-3-70b"
    }
  }
}
```

## Error Handling

### Error Response Format

```json
{
  "error": {
    "code": "invalid_request",
    "message": "Invalid parameter: symbol",
    "details": {
      "parameter": "symbol",
      "expected": "string format 'BASE/QUOTE'"
    },
    "request_id": "req_1234567890"
  }
}
```

### Error Codes

| Code | Description |
|------|-------------|
| 400 | Bad Request - Invalid request parameters |
| 401 | Unauthorized - Invalid or missing authentication |
| 403 | Forbidden - Insufficient permissions |
| 404 | Not Found - Resource not found |
| 429 | Too Many Requests - Rate limit exceeded |
| 500 | Internal Server Error - Something went wrong |
| 503 | Service Unavailable - Service temporarily unavailable |

## WebSocket API

### Real-time Price Updates

```
wss://api.graphrag-crypto.railway.app/v1/ws/prices
```

**Subscribe to Updates:**

```json
{
  "action": "subscribe",
  "symbols": ["BTC/USDT", "ETH/USDT"],
  "interval": "1s"
}
```

**Update Message:**

```json
{
  "type": "price_update",
  "timestamp": "2023-10-15T14:30:00.000Z",
  "data": [
    {
      "symbol": "BTC/USDT",
      "price": 67432.10,
      "volume": 1.2345,
      "exchange": "binance"
    }
  ]
}
```

## Client Libraries

### Python

```python
from graphrag import GraphRAGClient

client = GraphRAGClient(api_key="your_api_key")

# Get price
price = client.get_price(symbol="BTC/USDT")

# Run prediction
prediction = client.predict(symbol="BTC/USDT", horizon_days=7)

# Subscribe to real-time updates
for update in client.subscribe_prices(["BTC/USDT"]):
    print(f"{update['symbol']}: {update['price']}")
```

## Rate Limits

| Endpoint | Rate Limit |
|----------|------------|
| /v1/market/* | 100/req/min |
| /v1/graph/* | 50/req/min |
| /v1/agents/* | 20/req/min |
| /v1/ws/* | 10 connections/ip |

## Versioning

API versioning follows the format `v1`, `v2`, etc. The current version is `v1`.

## Changelog

### v1.0.0 (2023-10-15)
- Initial release
- Market data endpoints
- Graph query interface
- Agent prediction API
- WebSocket support for real-time updates
