from fastapi import FastAPI
from src.api import health_route
from src.api import predict_route

def create_app()-> FastAPI:
    
    app=FastAPI(title="Generous Tip Giver",
                description="The App Predict the Generous Tippers",
                version="0.0.1")

    app.include_router(router=health_route.router, tags=['health'])
    app.include_router(router=predict_route.router, tags=['prediction'])

    return app


app = create_app()

@app.on_event("startup")
async def startup_event():
    print("🚀 Generous Tipper API starting up...")

@app.on_event("shutdown")
async def shutdown_event():
    print("🛑 Generous Tipper API shutting down...")