import React, { useState } from 'react';
import { useQuery } from 'react-query';
import { 
  Card, 
  CardHeader, 
  CardContent, 
  Typography, 
  Button, 
  Grid, 
  Box, 
  Tabs, 
  Tab, 
  CircularProgress,
  Chip,
  Divider
} from '@mui/material';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import TrendingDownIcon from '@mui/icons-material/TrendingDown';
import WarningIcon from '@mui/icons-material/Warning';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import TimelineIcon from '@mui/icons-material/Timeline';
import { strategyApi, Strategy, StrategyRanking } from '../../services/strategyApi';
import { StatusBadge } from '../ui/StatusBadge';
import { TimeSeriesChart } from '../ui/TimeSeriesChart';
import { DataTable } from '../ui/DataTable';

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`volatility-tabpanel-${index}`}
      aria-labelledby={`volatility-tab-${index}`}
      {...other}
      style={{ width: '100%' }}
    >
      {value === index && (
        <Box sx={{ p: 3 }}>
          {children}
        </Box>
      )}
    </div>
  );
}

function a11yProps(index: number) {
  return {
    id: `volatility-tab-${index}`,
    'aria-controls': `volatility-tabpanel-${index}`,
  };
}

const VolatilityStrategyCard = ({ strategy }: { strategy: Strategy }) => {
  return (
    <Card sx={{ mb: 2, boxShadow: 3 }}>
      <CardHeader
        title={strategy.name}
        subheader={strategy.description}
        action={
          <StatusBadge status={strategy.status} />
        }
      />
      <CardContent>
        <Grid container spacing={2}>
          <Grid item xs={12} md={6}>
            <Typography variant="subtitle2" color="textSecondary">Performance</Typography>
            <Box display="flex" alignItems="center" mt={1}>
              <Typography variant="h6" color={strategy.performance.monthly >= 0 ? 'success.main' : 'error.main'}>
                {strategy.performance.monthly.toFixed(2)}%
              </Typography>
              {strategy.performance.monthly >= 0 ? 
                <TrendingUpIcon color="success" sx={{ ml: 1 }} /> : 
                <TrendingDownIcon color="error" sx={{ ml: 1 }} />}
            </Box>
          </Grid>
          <Grid item xs={12} md={6}>
            <Typography variant="subtitle2" color="textSecondary">Market Suitability</Typography>
            <Box display="flex" alignItems="center" mt={1}>
              <Typography variant="h6">
                {strategy.marketSuitability.toFixed(0)}%
              </Typography>
              {strategy.marketSuitability >= 70 ? 
                <CheckCircleIcon color="success" sx={{ ml: 1 }} /> : 
                <WarningIcon color="warning" sx={{ ml: 1 }} />}
            </Box>
          </Grid>
          {strategy.lastSignal && (
            <Grid item xs={12}>
              <Typography variant="subtitle2" color="textSecondary">Last Signal</Typography>
              <Box display="flex" alignItems="center" mt={1}>
                <Chip 
                  label={strategy.lastSignal.direction.toUpperCase()} 
                  color={
                    strategy.lastSignal.direction === 'long' ? 'success' : 
                    strategy.lastSignal.direction === 'short' ? 'error' : 'default'
                  }
                  size="small"
                  sx={{ mr: 1 }}
                />
                <Typography variant="body2">
                  Strength: {strategy.lastSignal.strength.toFixed(0)}% | 
                  {new Date(strategy.lastSignal.timestamp).toLocaleString()}
                </Typography>
              </Box>
            </Grid>
          )}
          <Grid item xs={12}>
            <Box display="flex" justifyContent="flex-end" mt={1}>
              <Button 
                variant="outlined" 
                size="small" 
                color={strategy.enabled ? 'error' : 'primary'}
                sx={{ mr: 1 }}
              >
                {strategy.enabled ? 'Disable' : 'Enable'}
              </Button>
              <Button 
                variant="contained" 
                size="small" 
                color="primary"
              >
                Details
              </Button>
            </Box>
          </Grid>
        </Grid>
      </CardContent>
    </Card>
  );
};

const VolatilityMetricsCard = () => {
  const [timeframe, setTimeframe] = useState<'day' | 'week' | 'month'>('week');
  
  // Sample VIX and IV data
  const vixData = [
    { time: '2025-04-10', value: 18.5 },
    { time: '2025-04-11', value: 19.2 },
    { time: '2025-04-14', value: 17.8 },
    { time: '2025-04-15', value: 16.9 },
    { time: '2025-04-16', value: 17.3 },
    { time: '2025-04-17', value: 18.1 },
    { time: '2025-04-18', value: 19.5 },
    { time: '2025-04-21', value: 20.3 },
    { time: '2025-04-22', value: 21.1 },
    { time: '2025-04-23', value: 22.7 },
    { time: '2025-04-24', value: 21.5 },
    { time: '2025-04-25', value: 20.8 },
    { time: '2025-04-28', value: 19.4 },
    { time: '2025-04-29', value: 18.7 },
    { time: '2025-04-30', value: 18.2 },
    { time: '2025-05-01', value: 17.9 },
    { time: '2025-05-02', value: 18.5 },
    { time: '2025-05-05', value: 19.1 },
    { time: '2025-05-06', value: 18.7 },
    { time: '2025-05-07', value: 17.4 },
    { time: '2025-05-08', value: 17.0 },
    { time: '2025-05-09', value: 16.8 },
    { time: '2025-05-12', value: 16.9 },
  ];
  
  // Sample implied volatility data for key symbols
  const symbolIVData = [
    { symbol: 'SPY', current: 22.3, historical: 18.5, percentile: 85 },
    { symbol: 'QQQ', current: 25.7, historical: 21.2, percentile: 78 },
    { symbol: 'AAPL', current: 34.1, historical: 29.8, percentile: 72 },
    { symbol: 'MSFT', current: 28.6, historical: 25.4, percentile: 65 },
    { symbol: 'AMZN', current: 42.8, historical: 37.1, percentile: 81 },
    { symbol: 'TSLA', current: 75.3, historical: 65.7, percentile: 88 },
  ];
  
  return (
    <Card sx={{ mb: 2, boxShadow: 3 }}>
      <CardHeader 
        title="Volatility Metrics" 
        subheader="Market volatility indicators for options strategies"
        action={
          <Box>
            <Button
              size="small"
              color={timeframe === 'day' ? 'primary' : 'inherit'}
              onClick={() => setTimeframe('day')}
              sx={{ minWidth: 30 }}
            >
              1D
            </Button>
            <Button
              size="small"
              color={timeframe === 'week' ? 'primary' : 'inherit'}
              onClick={() => setTimeframe('week')}
              sx={{ minWidth: 30 }}
            >
              1W
            </Button>
            <Button
              size="small"
              color={timeframe === 'month' ? 'primary' : 'inherit'}
              onClick={() => setTimeframe('month')}
              sx={{ minWidth: 30 }}
            >
              1M
            </Button>
          </Box>
        }
      />
      <CardContent>
        <Typography variant="subtitle2" color="textSecondary">VIX Index</Typography>
        <Box height={200} mt={1} mb={3}>
          <TimeSeriesChart 
            data={vixData}
            xKey="time"
            yKey="value"
            color="#8884d8"
            tooltipLabel="VIX"
            yAxisLabel="VIX Value"
          />
        </Box>
        
        <Typography variant="subtitle2" color="textSecondary">Symbol Implied Volatility</Typography>
        <Box mt={2}>
          <DataTable
            data={symbolIVData}
            columns={[
              { field: 'symbol', headerName: 'Symbol', width: 80 },
              { field: 'current', headerName: 'Current IV (%)', width: 120, 
                renderCell: (params) => `${params.value.toFixed(1)}%` },
              { field: 'historical', headerName: 'Historical IV (%)', width: 140,
                renderCell: (params) => `${params.value.toFixed(1)}%` },
              { field: 'percentile', headerName: 'IV Percentile', width: 120,
                renderCell: (params) => {
                  const value = params.value;
                  let color = 'default';
                  if (value > 80) color = 'success';
                  else if (value > 60) color = 'primary';
                  else if (value < 30) color = 'error';
                  
                  return (
                    <Chip 
                      label={`${value}%`} 
                      color={color as any}
                      size="small"
                    />
                  );
                }
              },
            ]}
            pageSize={6}
            hideFooter
          />
        </Box>
      </CardContent>
    </Card>
  );
};

const UpcomingEventsCard = () => {
  // Sample upcoming events that can impact volatility
  const upcomingEvents = [
    { date: '2025-05-15', type: 'Earnings', symbol: 'AAPL', impact: 'high', description: 'Apple Q2 Earnings' },
    { date: '2025-05-18', type: 'Economic', symbol: 'US', impact: 'high', description: 'FOMC Meeting Minutes' },
    { date: '2025-05-20', type: 'Earnings', symbol: 'NVDA', impact: 'high', description: 'NVIDIA Q1 Earnings' },
    { date: '2025-05-22', type: 'Economic', symbol: 'US', impact: 'medium', description: 'US GDP Report' },
    { date: '2025-05-25', type: 'Economic', symbol: 'EU', impact: 'medium', description: 'ECB Interest Rate Decision' },
    { date: '2025-05-28', type: 'Earnings', symbol: 'TSLA', impact: 'high', description: 'Tesla Q2 Earnings' },
  ];
  
  return (
    <Card sx={{ mb: 2, boxShadow: 3 }}>
      <CardHeader 
        title="Upcoming Volatility Events" 
        subheader="Events that may impact volatility strategies"
      />
      <CardContent>
        <DataTable
          data={upcomingEvents}
          columns={[
            { field: 'date', headerName: 'Date', width: 100 },
            { field: 'type', headerName: 'Type', width: 100,
              renderCell: (params) => (
                <Chip
                  label={params.value}
                  color={params.value === 'Earnings' ? 'primary' : 'secondary'}
                  size="small"
                />
              )
            },
            { field: 'symbol', headerName: 'Symbol', width: 80 },
            { field: 'impact', headerName: 'Impact', width: 100,
              renderCell: (params) => {
                const impact = params.value;
                return (
                  <Chip
                    label={impact}
                    color={
                      impact === 'high' ? 'error' :
                      impact === 'medium' ? 'warning' : 'success'
                    }
                    size="small"
                  />
                );
              }
            },
            { field: 'description', headerName: 'Description', width: 200 },
          ]}
          pageSize={6}
          hideFooter
        />
      </CardContent>
    </Card>
  );
};

export const VolatilityStrategyPanel: React.FC = () => {
  const [tabValue, setTabValue] = useState(0);
  
  // Fetch all strategies
  const { data: strategies, isLoading: isLoadingStrategies } = useQuery('strategies', strategyApi.getStrategies);
  
  // Filter volatility strategies
  const volatilityStrategies = strategies?.filter(
    s => s.category === 'volatility' || 
    s.name.toLowerCase().includes('straddle') || 
    s.name.toLowerCase().includes('strangle')
  ) || [];
  
  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };
  
  return (
    <Card sx={{ mb: 3, boxShadow: 2 }}>
      <CardHeader
        title={
          <Box display="flex" alignItems="center">
            <TimelineIcon sx={{ mr: 1 }} />
            <Typography variant="h6">Volatility Strategies</Typography>
          </Box>
        }
        subheader="Options strategies for profiting from market volatility"
      />
      <Divider />
      <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
        <Tabs 
          value={tabValue} 
          onChange={handleTabChange} 
          aria-label="volatility strategy tabs"
          variant="fullWidth"
        >
          <Tab label="Active Strategies" {...a11yProps(0)} />
          <Tab label="Volatility Metrics" {...a11yProps(1)} />
          <Tab label="Upcoming Events" {...a11yProps(2)} />
        </Tabs>
      </Box>
      
      <CardContent sx={{ p: 0 }}>
        <TabPanel value={tabValue} index={0}>
          {isLoadingStrategies ? (
            <Box display="flex" justifyContent="center" p={3}>
              <CircularProgress />
            </Box>
          ) : volatilityStrategies.length === 0 ? (
            <Box display="flex" justifyContent="center" p={3}>
              <Typography variant="body1" color="textSecondary">
                No volatility strategies found
              </Typography>
            </Box>
          ) : (
            <Box>
              {volatilityStrategies.map(strategy => (
                <VolatilityStrategyCard key={strategy.id} strategy={strategy} />
              ))}
            </Box>
          )}
        </TabPanel>
        
        <TabPanel value={tabValue} index={1}>
          <VolatilityMetricsCard />
        </TabPanel>
        
        <TabPanel value={tabValue} index={2}>
          <UpcomingEventsCard />
        </TabPanel>
      </CardContent>
    </Card>
  );
};

export default VolatilityStrategyPanel;
