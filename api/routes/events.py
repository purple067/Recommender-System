

from fastapi import APIRouter
from pydantic import BaseModel
from datetime import datetime
import psycopg2

router = APIRouter()

class Event(BaseModel):
    user_id: int
    item_id: int
    event_type: str

@router.post("/events")
def log_event(event: Event):
    conn = psycopg2.connect(
        dbname="recsys",
        user="postgres",
        password="postgres",
        host="localhost",
        port=5432
    )
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO user_events (user_id, item_id, event_type, timestamp)
        VALUES (%s, %s, %s, %s)
    """, (event.user_id, event.item_id, event.event_type, datetime.utcnow()))

    conn.commit()
    cur.close()
    conn.close()

    return {"status": "logged"}
