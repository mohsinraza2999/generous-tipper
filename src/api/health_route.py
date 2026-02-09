from fastapi import APIRouter
from datetime import datetime

router=APIRouter(tags=['health'])

@router.get("/health",response_model=dict,status_code=200)
async def check_health():
    """
    Basic health check endpoint.
    Returns service status and current timestamp.
    Useful for monitoring, Docker/K8s probes, and CI/CD checks.
    """
    return {
        "status": "ok",
        "service": "Generous Tipper Backend",
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }