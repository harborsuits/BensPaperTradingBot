import os, sqlite3, json, threading, time
from typing import Any, Dict, List, Tuple

DB_URL = os.getenv("DATABASE_URL", "sqlite:///./bensbot.db").replace("sqlite:///","")

INIT_SQL = """
CREATE TABLE IF NOT EXISTS trades(
  id TEXT PRIMARY KEY,
  ts TEXT,
  strategy TEXT,
  symbol TEXT,
  side TEXT,
  qty REAL,
  avg_price REAL,
  pnl REAL
);
CREATE TABLE IF NOT EXISTS orders(
  id TEXT PRIMARY KEY,
  ts TEXT,
  symbol TEXT,
  side TEXT,
  qty REAL,
  type TEXT,
  limit_price REAL,
  status TEXT,
  details_json TEXT
);
CREATE TABLE IF NOT EXISTS events(
  ts TEXT,
  level TEXT,
  component TEXT,
  message TEXT,
  trace_id TEXT,
  context_json TEXT
);
CREATE TABLE IF NOT EXISTS jobs(
  id TEXT PRIMARY KEY,
  kind TEXT,
  status TEXT,
  progress INTEGER,
  result_ref TEXT,
  error TEXT,
  created_ts TEXT,
  updated_ts TEXT
);
"""

def get_conn():
    conn = sqlite3.connect(DB_URL, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn

_conn = get_conn()
for stmt in INIT_SQL.strip().split(";"):
    if stmt.strip():
        _conn.execute(stmt)
_conn.commit()

def insert_event(level:str, component:str, message:str, trace_id:str="", ctx:Dict[str,Any]=None):
    _conn.execute("INSERT INTO events VALUES(datetime('now'),'{}','{}','{}','{}','{}')"
                  .format(level,component,message,trace_id,json.dumps(ctx or {})))
    _conn.commit()

def upsert_job(job_id:str, kind:str, status:str, progress:int, result_ref:str=None, error:str=None):
    _conn.execute("""
    INSERT INTO jobs(id,kind,status,progress,result_ref,error,created_ts,updated_ts)
    VALUES(?,?,?,?,?,?,datetime('now'),datetime('now'))
    ON CONFLICT(id) DO UPDATE SET status=excluded.status, progress=excluded.progress,
      result_ref=excluded.result_ref, error=excluded.error, updated_ts=datetime('now')
    """, (job_id,kind,status,progress,result_ref,error))
    _conn.commit()

def get_job(job_id:str)->Tuple:
    cur=_conn.execute("SELECT id,kind,status,progress,result_ref,error FROM jobs WHERE id=?",(job_id,))
    return cur.fetchone()
