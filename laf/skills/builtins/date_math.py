from datetime import datetime, timedelta

def date_add(days: int = 0):
    now = datetime.now()
    new_date = now + timedelta(days=days)
    return {
        "base": now.isoformat(),
        "days": days,
        "result": new_date.isoformat()
    }

def date_memory(result):
    return {
        "type": "date_operation",
        "importance": 0.4
    }
