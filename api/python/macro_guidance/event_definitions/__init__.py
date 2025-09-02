"""
Event Definitions Module

This module contains detailed implementations of specific macro economic events.
"""

from .cpi_report import CPIReport

# Map of event types to their implementations
EVENT_MAP = {
    "cpi": CPIReport
}

def get_event(event_type: str):
    """
    Get an event instance by type.
    
    Args:
        event_type: Type of economic event
        
    Returns:
        Instance of the specified event
    """
    event_class = EVENT_MAP.get(event_type.lower())
    if not event_class:
        raise ValueError(f"Unknown event type: {event_type}")
    
    return event_class() 