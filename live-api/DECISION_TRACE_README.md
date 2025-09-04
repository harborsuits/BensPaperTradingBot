# Decision Trace Integration Kit

This integration kit provides a unified data model for trade decisions, combining signals, evidence, context, and execution into a single auditable object.

## Overview

The Decision Trace Integration Kit consists of:

1. **Backend (FastAPI)**
   - Pydantic models for `DecisionTrace`
   - REST API endpoints for creating and listing traces
   - WebSocket support for real-time updates
   - Utility services for building and broadcasting traces

2. **Frontend (React/TypeScript)**
   - TypeScript types and Zod schema for `DecisionTrace`
   - React hook for consuming WebSocket stream with REST fallback
   - Components for displaying traces in a user-friendly way
   - Adapter for compatibility with existing Evidence Drawer

## Getting Started

### Backend Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Start the FastAPI server:
   ```bash
   ./start_api.sh
   ```
   This will start the server on port 8000 with auto-reload enabled.

3. Test the API:
   ```bash
   # Post a sample trace
   ./test_decision_trace.py
   
   # Get all traces
   ./test_decision_trace.py get
   ```

### Frontend Setup

1. The frontend components are already integrated into the new-trading-dashboard project.

2. Access the new Decision Trace page at:
   ```
   http://localhost:3003/decisions-v2
   ```

## API Reference

### REST Endpoints

- `GET /api/decision-traces`: Get all decision traces (with optional limit parameter)
- `POST /api/decision-traces`: Create a new decision trace

### WebSocket Endpoint

- `ws://localhost:8000/ws/decisions`: WebSocket endpoint for real-time updates

## Integration with Existing Code

### Backend Integration

To emit a `DecisionTrace` from your engine:

```python
from api.models.decision_trace import DecisionTrace, Plan, Execution
from api.routes.decisions import append_and_broadcast
from datetime import datetime, timezone

# Build and publish
now = datetime.now(timezone.utc).isoformat()

dt = DecisionTrace(
    trace_id=f"{now}-NVDA-001",
    as_of=now,
    symbol="NVDA",
    plan=Plan(
        action="OPEN_LONG", 
        entry={"type": "limit", "px": 5.2}, 
        sizing={"units": 2, "notional": 1040, "max_loss": 1040}
    ),
    explain_layman="Buy 2 call contracts because momentum is turning up from oversold; risk capped to premium.",
    explain_detail=[
        "RSI(14)=32 rising; MA cross-up in last 3 sessions.",
        "Policy/news tailwind on advanced chips.",
        "Risk gates OK; within per-trade cap.",
    ],
)

append_and_broadcast(dt)
```

### Frontend Integration

To use the `DecisionTrace` components in your UI:

```tsx
import { useDecisionFeed } from "@/hooks/useDecisionFeed";
import { DecisionCard } from "@/components/trade-decisions/DecisionCard";

function MyComponent() {
  const { decisions, status, error } = useDecisionFeed();
  
  return (
    <div>
      {decisions.map(decision => (
        <DecisionCard key={decision.trace_id} d={decision} />
      ))}
    </div>
  );
}
```

## Quality Checks

- **Validation**: Both frontend and backend validate traces against a schema
- **Evidence Requirement**: Traces must have at least one signal OR one news item
- **Instrument Label**: Set `plan.strategyLabel` for a user-friendly display name

## Troubleshooting

- **WebSocket Connection Issues**: Ensure your reverse proxy forwards WebSocket upgrade headers
- **Missing Data**: Check the console for validation errors; the UI will show a degraded banner
- **Backend Errors**: Check the FastAPI logs for details

## Next Steps

- Add persistence for traces (MongoDB or PostgreSQL)
- Enhance the UI with filtering and sorting options
- Add more detailed analytics and visualizations
- Integrate with your existing monitoring and alerting systems


