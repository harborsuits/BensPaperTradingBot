from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import asyncio
from datetime import datetime
import logging

from trading_bot.autonomous.ai_bot_competition import AIBotCompetition

logger = logging.getLogger(__name__)

# Global competition manager
competition_manager = AIBotCompetition()

class StartCompetitionRequest(BaseModel):
    evo_strategies: List[Dict[str, Any]]
    symbols: List[str]
    duration_days: Optional[int] = 7
    total_capital: Optional[float] = 10000.0

class UpdateBotPerformanceRequest(BaseModel):
    bot_id: str
    performance_data: Dict[str, Any]

router = APIRouter(prefix="/bot-competition", tags=["Bot Competition"])

@router.post("/start", response_model=Dict[str, str])
async def start_competition(request: StartCompetitionRequest, background_tasks: BackgroundTasks):
    """Start a new AI bot competition round"""
    try:
        round_id = await competition_manager.start_new_round(
            request.evo_strategies,
            request.symbols
        )

        # Start background monitoring
        background_tasks.add_task(monitor_competition, round_id)

        return {
            "round_id": round_id,
            "message": f"Competition started with {len(request.evo_strategies)} bots",
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Error starting competition: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/update-performance")
async def update_bot_performance(request: UpdateBotPerformanceRequest):
    """Update a bot's performance during competition"""
    try:
        await competition_manager.update_bot_performance(
            request.bot_id,
            request.performance_data
        )
        return {"status": "success", "message": f"Updated performance for bot {request.bot_id}"}
    except Exception as e:
        logger.error(f"Error updating bot performance: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/reallocate-capital")
async def reallocate_capital():
    """Trigger capital reallocation based on current performance"""
    try:
        await competition_manager.reallocate_capital()
        return {"status": "success", "message": "Capital reallocation completed"}
    except Exception as e:
        logger.error(f"Error reallocating capital: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/end-round")
async def end_round():
    """End the current competition round"""
    try:
        winner_bot_id = await competition_manager.end_round()
        return {
            "status": "success",
            "winner_bot_id": winner_bot_id,
            "message": f"Competition round ended. Winner: {winner_bot_id}"
        }
    except Exception as e:
        logger.error(f"Error ending round: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/leaderboard")
async def get_leaderboard():
    """Get current competition leaderboard"""
    try:
        leaderboard = competition_manager.get_leaderboard()
        return {
            "leaderboard": leaderboard,
            "total_bots": len(leaderboard),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting leaderboard: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stats")
async def get_competition_stats():
    """Get current competition statistics"""
    try:
        stats = competition_manager.get_round_stats()
        return {
            "stats": stats,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting competition stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/history")
async def get_competition_history():
    """Get competition history"""
    try:
        history = competition_manager.get_competition_history()
        return {
            "history": history,
            "total_rounds": len(history),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting competition history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/active-bots")
async def get_active_bots():
    """Get list of active competition bots"""
    try:
        active_bots = list(competition_manager.active_bots.values())
        return {
            "active_bots": [bot.to_dict() for bot in active_bots],
            "total_active": len(active_bots),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting active bots: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def monitor_competition(round_id: str):
    """Background task to monitor competition progress"""
    try:
        # Monitor for the duration of the competition
        while competition_manager.current_round and \
              competition_manager.current_round.round_id == round_id:

            # Periodic capital reallocation
            await asyncio.sleep(3600)  # Every hour
            await competition_manager.reallocate_capital()

            # Check if round should end
            if competition_manager.current_round:
                elapsed_days = (datetime.now() - competition_manager.current_round.start_time).days
                if elapsed_days >= competition_manager.current_round.duration_days:
                    await competition_manager.end_round()
                    break

    except Exception as e:
        logger.error(f"Error monitoring competition {round_id}: {e}")
