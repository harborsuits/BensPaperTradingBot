from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime
import asyncio
import random
from trading_bot.api.websocket_manager import enabled_manager

# Placeholder manager for EvoTester sessions
_sessions: Dict[str, Any] = {}
_results: Dict[str, Any] = {}

class EvoTesterConfig(BaseModel):
    populationSize: int
    generations: int
    crossoverRate: float
    mutationRate: float
    elitismCount: int
    fitnessMetric: str
    timeframe: str
    symbols: List[str]
    basedOnStrategyId: Optional[str] = None
    constraintsEnabled: bool
    customConfig: Optional[Dict[str, Any]] = None

class EvoTesterStatusModel(BaseModel):
    sessionId: str
    running: bool
    currentGeneration: int
    totalGenerations: int
    startTime: str
    estimatedCompletionTime: Optional[str] = None
    elapsedTime: Optional[str] = None
    progress: int
    bestFitness: float
    averageFitness: float
    status: str
    errorMessage: Optional[str] = None

class GenerationResultModel(BaseModel):
    generation: int
    bestFitness: float
    averageFitness: float
    diversityScore: float
    bestIndividual: Dict[str, Any]
    elapsedTime: str
    timestamp: str

class EvoTesterResultModel(BaseModel):
    sessionId: str
    config: EvoTesterConfig
    topStrategies: List[Dict[str, Any]]
    generationsData: List[GenerationResultModel]
    startTime: str
    endTime: str
    totalRuntime: str
    status: str
    errorMessage: Optional[str] = None

router = APIRouter(prefix="/evotester", tags=["EvoTester"])

# Endpoint to start a new EvoTest session
@router.post("/start", response_model=Dict[str, str])
async def start_evo_test(config: EvoTesterConfig, background_tasks: BackgroundTasks):
    session_id = str(uuid.uuid4())
    _sessions[session_id] = {
        "sessionId": session_id,
        "running": True,
        "currentGeneration": 0,
        "totalGenerations": config.generations,
        "startTime": datetime.utcnow().isoformat(),
        "progress": 0,
        "bestFitness": 0.0,
        "averageFitness": 0.0,
        "status": "running",
        "errorMessage": None
    }
    # Schedule background EvoTester execution
    background_tasks.add_task(run_evo_task, session_id, config)
    return {"sessionId": session_id, "message": "EvoTester started"}

# Endpoint to stop an ongoing EvoTest
@router.post("/{session_id}/stop", response_model=Dict[str, Any])
async def stop_evo_test(session_id: str):
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    _sessions[session_id]["running"] = False
    _sessions[session_id]["status"] = "aborted"
    return {"success": True, "message": "Session aborted"}

# Endpoint to get the status of an EvoTest session
@router.get("/{session_id}/status", response_model=EvoTesterStatusModel)
async def get_evo_status(session_id: str):
    status = _sessions.get(session_id)
    if not status:
        raise HTTPException(status_code=404, detail="Session not found")
    return status

# Endpoint to get the result of a completed EvoTest
@router.get("/{session_id}/result", response_model=EvoTesterResultModel)
async def get_evo_result(session_id: str):
    result = _results.get(session_id)
    if not result:
        raise HTTPException(status_code=404, detail="Result not available")
    return result

# Endpoint to list recent EvoTest sessions
@router.get("/recent", response_model=List[EvoTesterStatusModel])
async def get_recent_sessions(limit: int = 10):
    sessions = list(_sessions.values())
    return sessions[:limit]

# Helper functions for EvoTester background execution
def run_evo_task(session_id: str, config: EvoTesterConfig):
    "Schedule the async evolution process"
    asyncio.create_task(_run_evo(session_id, config))

async def _run_evo(session_id: str, config: EvoTesterConfig):
    "Background task to execute EvoTester and broadcast updates"
    generation_history: List[Dict[str, Any]] = []
    try:
        for gen in range(1, config.generations + 1):
            if not _sessions[session_id]["running"]:
                break
            await asyncio.sleep(1)
            # Simulate metrics
            best = random.random()
            avg = random.random()
            progress = int(gen / config.generations * 100)
            _sessions[session_id].update({
                "currentGeneration": gen,
                "bestFitness": best,
                "averageFitness": avg,
                "progress": progress
            })
            # Broadcast status update via WebSocket to evotester channel
            await enabled_manager.broadcast_to_channel("evotester", "evo_progress", _sessions[session_id])
            # Broadcast generation result
            gen_result = GenerationResultModel(
                generation=gen,
                bestFitness=best,
                averageFitness=avg,
                diversityScore=random.random(),
                bestIndividual={},
                elapsedTime="",
                timestamp=datetime.utcnow().isoformat()
            )
            generation_history.append(gen_result.dict())
            await enabled_manager.broadcast_to_channel("evotester", "evo_generation_complete", gen_result.dict())
        # Finalize result
        result = EvoTesterResultModel(
            sessionId=session_id,
            config=config,
            topStrategies=[],
            generationsData=generation_history,
            startTime=_sessions[session_id]["startTime"],
            endTime=datetime.utcnow().isoformat(),
            totalRuntime="",
            status="completed"
        )
        _results[session_id] = result.dict()
        await enabled_manager.broadcast_to_channel("evotester", "evo_complete", result.dict())
    except Exception as e:
        _sessions[session_id]["running"] = False
        _sessions[session_id]["status"] = "failed"
        _sessions[session_id]["errorMessage"] = str(e)
        await enabled_manager.broadcast_to_channel("evotester", "evo_error", _sessions[session_id])
