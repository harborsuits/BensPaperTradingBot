
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

@app.get('/')
async def root():
    return {'message': 'Test API running', 'version': '1.0.0'}

@app.get('/api/portfolio')
async def get_portfolio():
    return {
        'totalValue': 100000,
        'dailyChange': 1500,
        'dailyChangePercent': 1.5,
        'cash': 25000,
        'invested': 75000
    }

@app.get('/api/strategies')
async def get_strategies():
    return [
        {
            'id': 'strat-1',
            'name': 'Momentum',
            'active': True,
            'exposure_pct': 0.12,
            'p_l_30d': 1250.50
        }
    ]

