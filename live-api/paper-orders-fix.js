// Add this endpoint if it doesn't exist
app.get('/api/paper/orders/open', (req, res) => {
  const open = paperOrders.filter(o => !['filled', 'canceled', 'rejected'].includes(String(o.status || '').toLowerCase()));
  res.json(open.map(o => ({
    order_id: o.id,
    symbol: o.symbol,
    side: String(o.side || '').toUpperCase(),
    qty: o.qty,
    status: o.status,
    created_ts: Math.floor(new Date(o.submittedAt || asOf()).getTime() / 1000),
    limit_price: o.price,
  })));
});


