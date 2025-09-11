from dataclasses import dataclass
import numpy as np

@dataclass
class Params:
    fast:int=20; mid:int=50; slow:int=200; pullback_ema:int=20; adx_min:int=18

def generate_signal(df, p: Params):
    # df has columns: close, high, low, volume; plus indicators: sma20/50/200, ema20, adx
    uptrend = (df.sma20 > df.sma50) & (df.sma50 > df.sma200) & (df.adx >= p.adx_min)
    pullback = (df.close <= df.ema20) & (df.low <= df.ema20)
    momentum = df.close > df.close.shift(1)
    buy = uptrend & pullback & momentum
    sell = (df.close < df.ema20) | (df.close < df.sma50)
    return buy.astype(int), sell.astype(int)
