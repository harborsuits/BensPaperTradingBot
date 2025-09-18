# live-api


## Development

```bash
npm install
npm run dev   # builds TS and starts API at :4000
```

- UI runs at :3003 (Vite).
- API base is proxied via Vite (/api, /metrics, /ws -> :4000).
- To run full server: `npm start` (server.js).
- Minimal mock server: `node minimal_server.js` (also :4000).

