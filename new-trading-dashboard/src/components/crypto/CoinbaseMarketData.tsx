import React, { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Grid,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  CircularProgress,
  Chip,
  Divider,
  Button,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  SelectChangeEvent,
} from '@mui/material';
import { 
  Timeline, 
  TrendingUp, 
  TrendingDown, 
  Refresh, 
  ShowChart,
  ArrowUpward,
  ArrowDownward,
} from '@mui/icons-material';
import { ResponsiveContainer, AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip } from 'recharts';
import coinbaseApi, { TickerData, HistoricalCandle, MarketSummary } from '../../services/coinbaseApi';

// Timeframe options for chart
const timeframeOptions = [
  { value: '1m', label: '1 Minute' },
  { value: '5m', label: '5 Minutes' },
  { value: '15m', label: '15 Minutes' },
  { value: '1h', label: '1 Hour' },
  { value: '6h', label: '6 Hours' },
  { value: '1d', label: '1 Day' },
];

// Chart data formatter
const formatChartData = (candles: HistoricalCandle[]) => {
  return candles.map((candle) => ({
    timestamp: new Date(candle.timestamp).toLocaleTimeString(),
    price: candle.close,
    open: candle.open,
    high: candle.high,
    low: candle.low,
    volume: candle.volume,
  }));
};

const CoinbaseMarketData: React.FC = () => {
  // State variables
  const [loading, setLoading] = useState<boolean>(true);
  const [marketSummary, setMarketSummary] = useState<MarketSummary | null>(null);
  const [availableSymbols, setAvailableSymbols] = useState<string[]>([]);
  const [selectedSymbol, setSelectedSymbol] = useState<string>('BTC-USD');
  const [selectedTimeframe, setSelectedTimeframe] = useState<string>('1h');
  const [tickerData, setTickerData] = useState<TickerData | null>(null);
  const [chartData, setChartData] = useState<HistoricalCandle[]>([]);
  const [refreshKey, setRefreshKey] = useState<number>(0);

  // Initial data loading
  useEffect(() => {
    const loadInitialData = async () => {
      setLoading(true);
      try {
        // Load available symbols
        const products = await coinbaseApi.getProducts();
        if (products && products.length > 0) {
          // Filter to major USD pairs for the dropdown
          const usdPairs = products.filter(p => p.endsWith('-USD')).slice(0, 20);
          setAvailableSymbols(usdPairs);
        }

        // Load market summary
        const summary = await coinbaseApi.getMarketSummary();
        if (summary) {
          setMarketSummary(summary);
        }

        // Initial load of selected symbol
        await loadSymbolData('BTC-USD', '1h');
      } catch (error) {
        console.error('Error loading initial data:', error);
      } finally {
        setLoading(false);
      }
    };

    loadInitialData();
  }, [refreshKey]);

  // Load data for a specific symbol
  const loadSymbolData = async (symbol: string, timeframe: string) => {
    try {
      // Load ticker data
      const ticker = await coinbaseApi.getTicker(symbol);
      if (ticker) {
        setTickerData(ticker);
      }

      // Load chart data
      const candles = await coinbaseApi.getCandles(symbol, timeframe);
      if (candles && candles.length > 0) {
        setChartData(candles);
      }
    } catch (error) {
      console.error(`Error loading data for ${symbol}:`, error);
    }
  };

  // Handle symbol change
  const handleSymbolChange = (event: SelectChangeEvent) => {
    const newSymbol = event.target.value;
    setSelectedSymbol(newSymbol);
    loadSymbolData(newSymbol, selectedTimeframe);
  };

  // Handle timeframe change
  const handleTimeframeChange = (event: SelectChangeEvent) => {
    const newTimeframe = event.target.value;
    setSelectedTimeframe(newTimeframe);
    loadSymbolData(selectedSymbol, newTimeframe);
  };

  // Refresh data
  const handleRefresh = () => {
    setRefreshKey(prevKey => prevKey + 1);
  };

  return (
    <Box sx={{ p: 2 }}>
      <Card>
        <CardContent>
          <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
            <Typography variant="h5" component="h2">
              <img 
                src="https://www.coinbase.com/assets/images/coinbase-logo-black.png" 
                alt="Coinbase" 
                height="24px" 
                style={{ marginRight: '10px', verticalAlign: 'middle' }} 
              />
              Coinbase Market Data
            </Typography>
            <Button
              variant="outlined"
              color="primary"
              startIcon={<Refresh />}
              onClick={handleRefresh}
            >
              Refresh
            </Button>
          </Box>

          {loading ? (
            <Box display="flex" justifyContent="center" p={3}>
              <CircularProgress />
            </Box>
          ) : (
            <>
              {/* Market Summary */}
              <Card variant="outlined" sx={{ mb: 3 }}>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Market Overview
                  </Typography>
                  <TableContainer component={Paper} variant="outlined">
                    <Table size="small">
                      <TableHead>
                        <TableRow>
                          <TableCell>Coin</TableCell>
                          <TableCell align="right">Price</TableCell>
                          <TableCell align="right">24h Change</TableCell>
                          <TableCell align="right">24h Volume</TableCell>
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        {marketSummary && Object.entries(marketSummary.coins).map(([symbol, data]) => (
                          <TableRow key={symbol} 
                            hover 
                            onClick={() => {
                              setSelectedSymbol(symbol);
                              loadSymbolData(symbol, selectedTimeframe);
                            }}
                            sx={{ 
                              cursor: 'pointer',
                              backgroundColor: symbol === selectedSymbol ? 'rgba(0, 0, 0, 0.04)' : 'inherit' 
                            }}
                          >
                            <TableCell component="th" scope="row">
                              {symbol.split('-')[0]}
                            </TableCell>
                            <TableCell align="right">${data.price.toLocaleString()}</TableCell>
                            <TableCell align="right">
                              <Box display="flex" alignItems="center" justifyContent="flex-end">
                                {data.change_24h_pct > 0 ? (
                                  <ArrowUpward fontSize="small" color="success" />
                                ) : (
                                  <ArrowDownward fontSize="small" color="error" />
                                )}
                                <Typography
                                  variant="body2"
                                  color={data.change_24h_pct > 0 ? 'success.main' : 'error.main'}
                                >
                                  {Math.abs(data.change_24h_pct).toFixed(2)}%
                                </Typography>
                              </Box>
                            </TableCell>
                            <TableCell align="right">
                              {data.volume_24h.toLocaleString()}
                            </TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </TableContainer>
                </CardContent>
              </Card>

              {/* Symbol Selector and Timeframe */}
              <Grid container spacing={2} sx={{ mb: 3 }}>
                <Grid item xs={12} sm={6}>
                  <FormControl fullWidth variant="outlined" size="small">
                    <InputLabel>Trading Pair</InputLabel>
                    <Select
                      value={selectedSymbol}
                      onChange={handleSymbolChange}
                      label="Trading Pair"
                    >
                      {availableSymbols.map((symbol) => (
                        <MenuItem key={symbol} value={symbol}>
                          {symbol}
                        </MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                </Grid>
                <Grid item xs={12} sm={6}>
                  <FormControl fullWidth variant="outlined" size="small">
                    <InputLabel>Timeframe</InputLabel>
                    <Select
                      value={selectedTimeframe}
                      onChange={handleTimeframeChange}
                      label="Timeframe"
                    >
                      {timeframeOptions.map((option) => (
                        <MenuItem key={option.value} value={option.value}>
                          {option.label}
                        </MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                </Grid>
              </Grid>

              {/* Current Price */}
              {tickerData && (
                <Card variant="outlined" sx={{ mb: 3 }}>
                  <CardContent>
                    <Grid container spacing={2}>
                      <Grid item xs={12} md={6}>
                        <Typography variant="overline">Current Price</Typography>
                        <Typography variant="h4">${parseFloat(tickerData.price.toString()).toLocaleString()}</Typography>
                        <Box display="flex" alignItems="center" mt={1}>
                          {tickerData.change_24h > 0 ? (
                            <Chip
                              icon={<TrendingUp />}
                              label={`+$${Math.abs(tickerData.change_24h).toFixed(2)} (24h)`}
                              color="success"
                              size="small"
                            />
                          ) : (
                            <Chip
                              icon={<TrendingDown />}
                              label={`-$${Math.abs(tickerData.change_24h).toFixed(2)} (24h)`}
                              color="error"
                              size="small"
                            />
                          )}
                        </Box>
                      </Grid>
                      <Grid item xs={12} md={6}>
                        <Grid container spacing={1}>
                          <Grid item xs={6}>
                            <Typography variant="overline">24h High</Typography>
                            <Typography variant="body1">${tickerData.high_24h.toLocaleString()}</Typography>
                          </Grid>
                          <Grid item xs={6}>
                            <Typography variant="overline">24h Low</Typography>
                            <Typography variant="body1">${tickerData.low_24h.toLocaleString()}</Typography>
                          </Grid>
                          <Grid item xs={6}>
                            <Typography variant="overline">24h Volume</Typography>
                            <Typography variant="body1">{tickerData.volume_24h.toLocaleString()}</Typography>
                          </Grid>
                          <Grid item xs={6}>
                            <Typography variant="overline">Last Updated</Typography>
                            <Typography variant="body1">
                              {new Date(tickerData.timestamp).toLocaleTimeString()}
                            </Typography>
                          </Grid>
                        </Grid>
                      </Grid>
                    </Grid>
                  </CardContent>
                </Card>
              )}

              {/* Price Chart */}
              <Card variant="outlined">
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    <ShowChart /> Price Chart: {selectedSymbol} ({selectedTimeframe})
                  </Typography>
                  <Box height={400}>
                    {chartData.length > 0 ? (
                      <ResponsiveContainer width="100%" height="100%">
                        <AreaChart
                          data={formatChartData(chartData)}
                          margin={{ top: 10, right: 30, left: 0, bottom: 0 }}
                        >
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis dataKey="timestamp" />
                          <YAxis domain={['auto', 'auto']} />
                          <Tooltip />
                          <Area
                            type="monotone"
                            dataKey="price"
                            stroke="#1976d2"
                            fill="#1976d2"
                            fillOpacity={0.3}
                          />
                        </AreaChart>
                      </ResponsiveContainer>
                    ) : (
                      <Box display="flex" justifyContent="center" alignItems="center" height="100%">
                        <Typography variant="body1">No chart data available</Typography>
                      </Box>
                    )}
                  </Box>
                </CardContent>
              </Card>
            </>
          )}
        </CardContent>
      </Card>
    </Box>
  );
};

export default CoinbaseMarketData;
