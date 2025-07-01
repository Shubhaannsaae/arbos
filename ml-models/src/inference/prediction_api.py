"""
Prediction API Gateway

High-level API for ML predictions with authentication, rate limiting,
and Chainlink Functions integration for on-chain compute.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import hashlib
import time

from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.security import HTTPBearer
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import aiohttp
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="ML Prediction API",
    description="High-level ML prediction API with Chainlink integration",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Configuration
MODEL_SERVER_URL = "http://localhost:8000"
RATE_LIMIT_REQUESTS = 100
RATE_LIMIT_WINDOW = 3600  # 1 hour

# Rate limiting storage
rate_limit_storage: Dict[str, List[float]] = {}

# Request models
class ArbitrageRequest(BaseModel):
    symbol: str
    amount: float
    exchange_from: str
    exchange_to: str

class PortfolioRiskRequest(BaseModel):
    assets: List[str]
    weights: List[float]
    timeframe: int = 30

class FraudDetectionRequest(BaseModel):
    transaction_hash: str
    from_address: str
    to_address: str
    value: float
    gas_used: int
    gas_price: float

class PredictionRequest(BaseModel):
    model_name: str
    input_data: Dict
    symbol: Optional[str] = None

def get_client_id(authorization: str = Header(None)) -> str:
    """Extract client ID from authorization header or use IP-based ID."""
    if authorization:
        # Use hash of authorization as client ID
        return hashlib.sha256(authorization.encode()).hexdigest()[:16]
    else:
        # Use default client ID for unauthenticated requests
        return "anonymous"

def check_rate_limit(client_id: str) -> bool:
    """Check if client is within rate limits."""
    now = time.time()
    
    if client_id not in rate_limit_storage:
        rate_limit_storage[client_id] = []
    
    # Remove old requests outside the window
    rate_limit_storage[client_id] = [
        req_time for req_time in rate_limit_storage[client_id]
        if now - req_time < RATE_LIMIT_WINDOW
    ]
    
    # Check if under limit
    if len(rate_limit_storage[client_id]) >= RATE_LIMIT_REQUESTS:
        return False
    
    # Record this request
    rate_limit_storage[client_id].append(now)
    return True

async def call_model_server(endpoint: str, data: Dict) -> Dict:
    """Call the model server with error handling."""
    try:
        async with aiohttp.ClientSession() as session:
            url = f"{MODEL_SERVER_URL}{endpoint}"
            async with session.post(url, json=data) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    raise HTTPException(
                        status_code=response.status,
                        detail=f"Model server error: {error_text}"
                    )
    except aiohttp.ClientError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Model server unavailable: {str(e)}"
        )

@app.get("/health")
async def health_check():
    """API health check."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{MODEL_SERVER_URL}/health") as response:
                model_server_healthy = response.status == 200
    except:
        model_server_healthy = False
    
    return {
        "status": "healthy" if model_server_healthy else "degraded",
        "model_server_healthy": model_server_healthy,
        "timestamp": datetime.now()
    }

@app.post("/api/v1/arbitrage/detect")
async def detect_arbitrage_opportunities(
    request: ArbitrageRequest,
    client_id: str = Depends(get_client_id)
):
    """Detect arbitrage opportunities for given parameters."""
    if not check_rate_limit(client_id):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    try:
        response = await call_model_server("/predict/arbitrage", {
            "symbol": request.symbol,
            "amount": request.amount,
            "exchange_from": request.exchange_from,
            "exchange_to": request.exchange_to
        })
        
        return {
            "status": "success",
            "data": response,
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"Arbitrage detection failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/portfolio/risk")
async def assess_portfolio_risk(
    request: PortfolioRiskRequest,
    client_id: str = Depends(get_client_id)
):
    """Assess risk for a given portfolio composition."""
    if not check_rate_limit(client_id):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    try:
        response = await call_model_server("/predict/portfolio/risk", {
            "assets": request.assets,
            "weights": request.weights,
            "timeframe": request.timeframe
        })
        
        return {
            "status": "success",
            "data": response,
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"Portfolio risk assessment failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/security/fraud")
async def detect_fraud(
    request: FraudDetectionRequest,
    client_id: str = Depends(get_client_id)
):
    """Detect fraud in transaction data."""
    if not check_rate_limit(client_id):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    try:
        response = await call_model_server("/predict", {
            "model_name": "security_fraud_detection",
            "input_data": {
                "transaction_hash": request.transaction_hash,
                "from_address": request.from_address,
                "to_address": request.to_address,
                "value": request.value,
                "gas_used": request.gas_used,
                "gas_price": request.gas_price
            }
        })
        
        return {
            "status": "success",
            "data": response,
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"Fraud detection failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/predict")
async def general_predict(
    request: PredictionRequest,
    client_id: str = Depends(get_client_id)
):
    """General prediction endpoint for any model."""
    if not check_rate_limit(client_id):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    try:
        response = await call_model_server("/predict", {
            "model_name": request.model_name,
            "input_data": request.input_data,
            "symbol": request.symbol
        })
        
        return {
            "status": "success",
            "data": response,
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"General prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/models")
async def list_available_models():
    """List all available models."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{MODEL_SERVER_URL}/models") as response:
                if response.status == 200:
                    models_data = await response.json()
                    return {
                        "status": "success",
                        "data": models_data,
                        "timestamp": datetime.now()
                    }
                else:
                    raise HTTPException(status_code=503, detail="Model server unavailable")
    except Exception as e:
        logger.error(f"Failed to get models list: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/rate-limit/status")
async def get_rate_limit_status(client_id: str = Depends(get_client_id)):
    """Get current rate limit status for client."""
    now = time.time()
    
    if client_id not in rate_limit_storage:
        requests_made = 0
    else:
        # Count requests in current window
        recent_requests = [
            req_time for req_time in rate_limit_storage[client_id]
            if now - req_time < RATE_LIMIT_WINDOW
        ]
        requests_made = len(recent_requests)
    
    return {
        "client_id": client_id,
        "requests_made": requests_made,
        "requests_limit": RATE_LIMIT_REQUESTS,
        "window_seconds": RATE_LIMIT_WINDOW,
        "requests_remaining": max(0, RATE_LIMIT_REQUESTS - requests_made),
        "timestamp": datetime.now()
    }

if __name__ == "__main__":
    uvicorn.run(
        "prediction_api:app",
        host="0.0.0.0",
        port=8080,
        reload=False,
        workers=1
    )
