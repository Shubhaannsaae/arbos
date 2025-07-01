"""
Production Model Server

FastAPI-based model server with Chainlink integration for real-time inference,
model lifecycle management, and cross-chain deployment capabilities.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
import sys
import os

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
import joblib

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.arbitrage.price_prediction import PricePredictionModel
from models.arbitrage.opportunity_detection import ArbitrageOpportunityDetector
from models.portfolio.risk_assessment import RiskAssessmentModel
from models.security.fraud_detection import FraudDetectionModel
from data.loaders.chainlink_loader import ChainlinkLoader

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="ML Models Inference Server",
    description="Production ML inference server with Chainlink integration",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model registry
models: Dict[str, Any] = {}
chainlink_loader: Optional[ChainlinkLoader] = None

# Request/Response models
class PredictionRequest(BaseModel):
    model_name: str = Field(..., description="Name of the model to use")
    input_data: Dict[str, Any] = Field(..., description="Input data for prediction")
    symbol: Optional[str] = Field(None, description="Asset symbol (if applicable)")

class PredictionResponse(BaseModel):
    model_name: str
    prediction: Any
    confidence: Optional[float] = None
    timestamp: datetime
    processing_time_ms: float

class HealthResponse(BaseModel):
    status: str
    models_loaded: int
    chainlink_connected: bool
    timestamp: datetime

@app.on_event("startup")
async def startup_event():
    """Initialize models and Chainlink connection on startup."""
    logger.info("Starting model server...")
    
    try:
        # Initialize Chainlink loader
        global chainlink_loader
        chainlink_loader = ChainlinkLoader({
            'ethereum': os.getenv('ETHEREUM_RPC_URL', 'https://eth.llamarpc.com'),
            'avalanche': os.getenv('AVALANCHE_RPC_URL', 'https://api.avax.network/ext/bc/C/rpc')
        })
        
        # Load models
        await load_all_models()
        
        logger.info(f"Model server started successfully with {len(models)} models")
        
    except Exception as e:
        logger.error(f"Failed to start model server: {e}")
        raise

async def load_all_models():
    """Load all trained models."""
    global models
    
    model_configs = {
        'arbitrage_price_prediction': {
            'class': PricePredictionModel,
            'path': 'models/arbitrage_price_prediction.joblib'
        },
        'arbitrage_opportunity_detection': {
            'class': ArbitrageOpportunityDetector,
            'path': 'models/arbitrage_opportunity_detection.joblib'
        },
        'portfolio_risk_assessment': {
            'class': RiskAssessmentModel,
            'path': 'models/portfolio_risk_assessment.joblib'
        },
        'security_fraud_detection': {
            'class': FraudDetectionModel,
            'path': 'models/security_fraud_detection.joblib'
        }
    }
    
    for model_name, config in model_configs.items():
        try:
            if os.path.exists(config['path']):
                model = config['class']()
                model.load_model(config['path'])
                models[model_name] = model
                logger.info(f"Loaded model: {model_name}")
            else:
                logger.warning(f"Model file not found: {config['path']}")
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        models_loaded=len(models),
        chainlink_connected=chainlink_loader is not None,
        timestamp=datetime.now()
    )

@app.get("/models")
async def list_models():
    """List all available models."""
    return {
        "models": list(models.keys()),
        "count": len(models),
        "timestamp": datetime.now()
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make prediction using specified model."""
    start_time = datetime.now()
    
    if request.model_name not in models:
        raise HTTPException(
            status_code=404, 
            detail=f"Model '{request.model_name}' not found"
        )
    
    try:
        model = models[request.model_name]
        
        # Get real-time data if symbol provided
        if request.symbol and chainlink_loader:
            price_data = await chainlink_loader.get_latest_price(request.symbol)
            if price_data:
                request.input_data['current_price'] = price_data.price
                request.input_data['price_timestamp'] = price_data.timestamp
        
        # Make prediction
        if hasattr(model, 'predict'):
            if request.model_name == 'arbitrage_price_prediction':
                import pandas as pd
                input_df = pd.DataFrame([request.input_data])
                prediction = model.predict(input_df)
            elif request.model_name == 'arbitrage_opportunity_detection':
                import pandas as pd
                input_df = pd.DataFrame([request.input_data])
                opportunities = model.detect_opportunities(input_df)
                prediction = [opp.__dict__ for opp in opportunities]
            elif request.model_name == 'portfolio_risk_assessment':
                import pandas as pd
                input_series = pd.Series(request.input_data)
                risk_assessment = model.assess_risk(input_series)
                prediction = risk_assessment.__dict__
            elif request.model_name == 'security_fraud_detection':
                import pandas as pd
                input_df = pd.DataFrame([request.input_data])
                fraud_result = model.detect_fraud(input_df)
                prediction = fraud_result[0].__dict__ if fraud_result else None
            else:
                prediction = model.predict(request.input_data)
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Model '{request.model_name}' does not support prediction"
            )
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return PredictionResponse(
            model_name=request.model_name,
            prediction=prediction,
            confidence=getattr(prediction, 'confidence', None) if hasattr(prediction, 'confidence') else None,
            timestamp=datetime.now(),
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Prediction failed for {request.model_name}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

@app.post("/predict/arbitrage")
async def predict_arbitrage(
    symbol: str,
    amount: float,
    exchange_from: str,
    exchange_to: str
):
    """Specialized arbitrage prediction endpoint."""
    if chainlink_loader is None:
        raise HTTPException(status_code=503, detail="Chainlink loader not available")
    
    try:
        # Get current price from Chainlink
        price_data = await chainlink_loader.get_latest_price(symbol)
        if not price_data:
            raise HTTPException(status_code=404, detail=f"Price data not found for {symbol}")
        
        # Prepare arbitrage input
        arbitrage_input = {
            'symbol': symbol,
            'amount': amount,
            'current_price': price_data.price,
            'exchange_from': exchange_from,
            'exchange_to': exchange_to,
            'timestamp': price_data.timestamp
        }
        
        # Get opportunity detection
        if 'arbitrage_opportunity_detection' in models:
            import pandas as pd
            input_df = pd.DataFrame([arbitrage_input])
            opportunities = models['arbitrage_opportunity_detection'].detect_opportunities(input_df)
            
            return {
                'symbol': symbol,
                'current_price': price_data.price,
                'opportunities': [opp.__dict__ for opp in opportunities],
                'timestamp': datetime.now()
            }
        else:
            raise HTTPException(status_code=503, detail="Arbitrage model not available")
            
    except Exception as e:
        logger.error(f"Arbitrage prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/portfolio/risk")
async def predict_portfolio_risk(
    assets: List[str],
    weights: List[float],
    timeframe: int = 30
):
    """Specialized portfolio risk assessment endpoint."""
    try:
        if len(assets) != len(weights):
            raise HTTPException(status_code=400, detail="Assets and weights length mismatch")
        
        if abs(sum(weights) - 1.0) > 0.01:
            raise HTTPException(status_code=400, detail="Weights must sum to 1.0")
        
        # Get current prices for all assets
        portfolio_data = {}
        if chainlink_loader:
            for asset in assets:
                price_data = await chainlink_loader.get_latest_price(asset)
                if price_data:
                    portfolio_data[f'{asset}_price'] = price_data.price
        
        # Add portfolio composition
        for i, (asset, weight) in enumerate(zip(assets, weights)):
            portfolio_data[f'weight_{asset}'] = weight
        
        portfolio_data['portfolio_size'] = len(assets)
        portfolio_data['timeframe'] = timeframe
        
        # Get risk assessment
        if 'portfolio_risk_assessment' in models:
            import pandas as pd
            input_series = pd.Series(portfolio_data)
            risk_assessment = models['portfolio_risk_assessment'].assess_risk(input_series)
            
            return {
                'portfolio': dict(zip(assets, weights)),
                'risk_assessment': risk_assessment.__dict__,
                'timestamp': datetime.now()
            }
        else:
            raise HTTPException(status_code=503, detail="Portfolio risk model not available")
            
    except Exception as e:
        logger.error(f"Portfolio risk prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reload_models")
async def reload_models(background_tasks: BackgroundTasks):
    """Reload all models in the background."""
    background_tasks.add_task(load_all_models)
    return {"message": "Model reload initiated", "timestamp": datetime.now()}

if __name__ == "__main__":
    uvicorn.run(
        "model_server:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=False,
        workers=1
    )
