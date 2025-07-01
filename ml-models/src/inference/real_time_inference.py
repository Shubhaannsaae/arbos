"""
Real-Time Inference Engine

Handles streaming data, real-time predictions, and Chainlink Functions
integration for on-chain compute and cross-chain inference.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
import json
from dataclasses import dataclass, asdict
import websockets
import aiohttp

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class StreamingPrediction:
    """Real-time prediction result."""
    model_name: str
    prediction: Any
    confidence: float
    input_data: Dict[str, Any]
    timestamp: datetime
    latency_ms: float

@dataclass
class StreamConfig:
    """Configuration for streaming inference."""
    model_name: str
    input_source: str  # 'chainlink', 'websocket', 'api'
    prediction_frequency: int  # seconds
    batch_size: int = 1
    enabled: bool = True

class RealTimeInferenceEngine:
    """
    Production real-time inference engine with Chainlink integration
    for streaming predictions and on-chain compute.
    """
    
    def __init__(self, model_server_url: str = "http://localhost:8000"):
        """Initialize real-time inference engine."""
        self.model_server_url = model_server_url
        self.active_streams: Dict[str, StreamConfig] = {}
        self.prediction_callbacks: Dict[str, List[Callable]] = {}
        self.running = False
        self.tasks: List[asyncio.Task] = []
        
        # Data queues for different sources
        self.chainlink_queue = asyncio.Queue(maxsize=1000)
        self.websocket_queue = asyncio.Queue(maxsize=1000)
        self.api_queue = asyncio.Queue(maxsize=1000)
        
        # Performance tracking
        self.prediction_count = 0
        self.error_count = 0
        self.total_latency = 0.0
        
    async def start(self):
        """Start the real-time inference engine."""
        logger.info("Starting real-time inference engine...")
        self.running = True
        
        # Start processing tasks
        self.tasks = [
            asyncio.create_task(self._process_chainlink_stream()),
            asyncio.create_task(self._process_websocket_stream()),
            asyncio.create_task(self._process_api_stream()),
            asyncio.create_task(self._monitor_performance())
        ]
        
        logger.info("Real-time inference engine started")
    
    async def stop(self):
        """Stop the real-time inference engine."""
        logger.info("Stopping real-time inference engine...")
        self.running = False
        
        # Cancel all tasks
        for task in self.tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self.tasks, return_exceptions=True)
        
        logger.info("Real-time inference engine stopped")
    
    def add_stream(self, stream_id: str, config: StreamConfig):
        """Add a new streaming prediction configuration."""
        self.active_streams[stream_id] = config
        self.prediction_callbacks[stream_id] = []
        logger.info(f"Added stream: {stream_id} with model: {config.model_name}")
    
    def remove_stream(self, stream_id: str):
        """Remove a streaming prediction configuration."""
        if stream_id in self.active_streams:
            del self.active_streams[stream_id]
            del self.prediction_callbacks[stream_id]
            logger.info(f"Removed stream: {stream_id}")
    
    def add_prediction_callback(self, stream_id: str, callback: Callable[[StreamingPrediction], None]):
        """Add callback for prediction results."""
        if stream_id in self.prediction_callbacks:
            self.prediction_callbacks[stream_id].append(callback)
    
    async def submit_data(self, source: str, data: Dict[str, Any]):
        """Submit data for real-time processing."""
        try:
            if source == 'chainlink':
                await self.chainlink_queue.put(data)
            elif source == 'websocket':
                await self.websocket_queue.put(data)
            elif source == 'api':
                await self.api_queue.put(data)
            else:
                logger.warning(f"Unknown data source: {source}")
        except asyncio.QueueFull:
            logger.warning(f"Queue full for source: {source}")
    
    async def _process_chainlink_stream(self):
        """Process Chainlink data stream."""
        while self.running:
            try:
                # Wait for data with timeout
                data = await asyncio.wait_for(
                    self.chainlink_queue.get(),
                    timeout=1.0
                )
                
                await self._process_data('chainlink', data)
                self.chainlink_queue.task_done()
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Chainlink stream processing error: {e}")
                self.error_count += 1
    
    async def _process_websocket_stream(self):
        """Process WebSocket data stream."""
        while self.running:
            try:
                data = await asyncio.wait_for(
                    self.websocket_queue.get(),
                    timeout=1.0
                )
                
                await self._process_data('websocket', data)
                self.websocket_queue.task_done()
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"WebSocket stream processing error: {e}")
                self.error_count += 1
    
    async def _process_api_stream(self):
        """Process API data stream."""
        while self.running:
            try:
                data = await asyncio.wait_for(
                    self.api_queue.get(),
                    timeout=1.0
                )
                
                await self._process_data('api', data)
                self.api_queue.task_done()
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"API stream processing error: {e}")
                self.error_count += 1
    
    async def _process_data(self, source: str, data: Dict[str, Any]):
        """Process incoming data and make predictions."""
        start_time = datetime.now()
        
        # Find matching streams
        matching_streams = [
            (stream_id, config) for stream_id, config in self.active_streams.items()
            if config.input_source == source and config.enabled
        ]
        
        for stream_id, config in matching_streams:
            try:
                # Make prediction
                prediction_result = await self._make_prediction(
                    config.model_name,
                    data
                )
                
                # Calculate latency
                latency = (datetime.now() - start_time).total_seconds() * 1000
                
                # Create streaming prediction
                streaming_pred = StreamingPrediction(
                    model_name=config.model_name,
                    prediction=prediction_result.get('prediction'),
                    confidence=prediction_result.get('confidence', 0.0),
                    input_data=data,
                    timestamp=datetime.now(),
                    latency_ms=latency
                )
                
                # Call callbacks
                for callback in self.prediction_callbacks[stream_id]:
                    try:
                        callback(streaming_pred)
                    except Exception as e:
                        logger.error(f"Callback error for stream {stream_id}: {e}")
                
                # Update metrics
                self.prediction_count += 1
                self.total_latency += latency
                
                logger.debug(f"Processed prediction for stream {stream_id}: {latency:.2f}ms")
                
            except Exception as e:
                logger.error(f"Prediction processing error for stream {stream_id}: {e}")
                self.error_count += 1
    
    async def _make_prediction(self, model_name: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Make prediction using model server."""
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.model_server_url}/predict"
                payload = {
                    "model_name": model_name,
                    "input_data": input_data
                }
                
                async with session.post(url, json=payload) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        error_text = await response.text()
                        raise Exception(f"Model server error: {error_text}")
                        
        except Exception as e:
            logger.error(f"Prediction request failed: {e}")
            raise
    
    async def _monitor_performance(self):
        """Monitor and log performance metrics."""
        while self.running:
            try:
                await asyncio.sleep(60)  # Log every minute
                
                if self.prediction_count > 0:
                    avg_latency = self.total_latency / self.prediction_count
                    error_rate = self.error_count / (self.prediction_count + self.error_count)
                    
                    logger.info(
                        f"Performance: {self.prediction_count} predictions, "
                        f"avg latency: {avg_latency:.2f}ms, "
                        f"error rate: {error_rate:.2%}"
                    )
                
            except Exception as e:
                logger.error(f"Performance monitoring error: {e}")
    
    async def start_chainlink_price_stream(self, symbols: List[str]):
        """Start streaming price data from Chainlink."""
        logger.info(f"Starting Chainlink price stream for: {symbols}")
        
        async def price_stream_task():
            while self.running:
                try:
                    # Simulate price data streaming (in production, use actual Chainlink API)
                    for symbol in symbols:
                        price_data = {
                            'symbol': symbol,
                            'price': 50000.0,  # Would be real price from Chainlink
                            'timestamp': datetime.now().isoformat(),
                            'source': 'chainlink'
                        }
                        
                        await self.submit_data('chainlink', price_data)
                    
                    await asyncio.sleep(5)  # 5-second intervals
                    
                except Exception as e:
                    logger.error(f"Chainlink price stream error: {e}")
                    await asyncio.sleep(10)
        
        # Add to tasks
        self.tasks.append(asyncio.create_task(price_stream_task()))
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics."""
        avg_latency = (
            self.total_latency / self.prediction_count 
            if self.prediction_count > 0 else 0.0
        )
        
        total_requests = self.prediction_count + self.error_count
        error_rate = self.error_count / total_requests if total_requests > 0 else 0.0
        
        return {
            'total_predictions': self.prediction_count,
            'total_errors': self.error_count,
            'average_latency_ms': avg_latency,
            'error_rate': error_rate,
            'active_streams': len(self.active_streams),
            'queues_size': {
                'chainlink': self.chainlink_queue.qsize(),
                'websocket': self.websocket_queue.qsize(),
                'api': self.api_queue.qsize()
            }
        }


# Example usage and testing
async def example_usage():
    """Example of how to use the real-time inference engine."""
    # Initialize engine
    engine = RealTimeInferenceEngine()
    
    # Add prediction callback
    def on_prediction(prediction: StreamingPrediction):
        logger.info(f"Prediction received: {prediction.model_name} -> {prediction.prediction}")
    
    # Configure streams
    arbitrage_stream = StreamConfig(
        model_name="arbitrage_price_prediction",
        input_source="chainlink",
        prediction_frequency=5,
        batch_size=1
    )
    
    engine.add_stream("arbitrage_btc", arbitrage_stream)
    engine.add_prediction_callback("arbitrage_btc", on_prediction)
    
    # Start engine
    await engine.start()
    
    # Start price streaming
    await engine.start_chainlink_price_stream(['BTC', 'ETH'])
    
    # Run for a while
    await asyncio.sleep(30)
    
    # Get stats
    stats = engine.get_performance_stats()
    logger.info(f"Performance stats: {stats}")
    
    # Stop engine
    await engine.stop()


if __name__ == "__main__":
    asyncio.run(example_usage())
