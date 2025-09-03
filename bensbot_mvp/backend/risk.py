from typing import List, Dict
def compute_portfolio_heat(positions:List[Dict], equity:float)->float:
    if not equity: return 0.0
    gross = sum(abs(p["qty"]*p["last"]) for p in positions)
    return min(1.0, gross/float(equity))
def concentration_flag(positions:List[Dict])->bool:
    return len(positions)>=3 and any(abs(p["qty"]*p["last"]) > 0.5*sum(abs(x["qty"]*x["last"]) for x in positions) for p in positions)
