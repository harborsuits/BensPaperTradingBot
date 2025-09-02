import React, { useState, useEffect } from 'react';
import { 
  Box, 
  Button, 
  Card, 
  Chip,
  CircularProgress, 
  Divider,
  FormControl,
  FormControlLabel,
  FormGroup,
  Grid,
  IconButton,
  MenuItem,
  Paper,
  Select,
  Switch,
  Tab,
  Tabs,
  Tooltip,
  Typography
} from '@mui/material';
import {
  AutoGraph,
  Analytics,
  Block,
  CheckCircle,
  Close,
  FilterAlt,
  Info,
  Pause,
  PlayArrow,
  Refresh,
  Warning
} from '@mui/icons-material';
import { DataGrid } from '@mui/x-data-grid';
import { apiFetch } from '../utils/api';

/**
 * Active Strategies Monitor Component
 * 
 * Provides a dashboard view of all active strategies with clear
 * distinction between paper and live strategies:
 * - Color-coded status indicators
 * - Filtering capabilities
 * - Performance metrics
 * - Controls for managing strategies
 */
const ActiveStrategiesMonitor = () => {
  // State
  const [loading, setLoading] = useState(false);
  const [strategies, setStrategies] = useState([]);
  const [filteredStrategies, setFilteredStrategies] = useState([]);
  const [filterMode, setFilterMode] = useState('all'); // 'all', 'paper', 'live'
  const [filterStatus, setFilterStatus] = useState('all'); // 'all', 'active', 'paused'
  const [showControls, setShowControls] = useState({
    paper: true,
    live: true
  });
  const [selectedStrategyIds, setSelectedStrategyIds] = useState([]);
  const [statsTimeframe, setStatsTimeframe] = useState('daily');

  // Load strategies on component mount
  useEffect(() => {
    fetchStrategies();
  }, []);

  // Update filtered strategies when filter changes
  useEffect(() => {
    applyFilters();
  }, [strategies, filterMode, filterStatus]);

  // Fetch strategies from API
  const fetchStrategies = async () => {
    setLoading(true);
    try {
      const response = await apiFetch('/api/strategies/list');
      if (response.success) {
        setStrategies(response.data.strategies || []);
      }
    } catch (error) {
      console.error('Error fetching strategies:', error);
    } finally {
      setLoading(false);
    }
  };

  // Apply filters to strategies
  const applyFilters = () => {
    let filtered = [...strategies];
    
    // Apply mode filter
    if (filterMode !== 'all') {
      const isPaper = filterMode === 'paper';
      filtered = filtered.filter(strategy => 
        (isPaper && strategy.phase === 'PAPER_TRADE') ||
        (!isPaper && strategy.phase === 'LIVE')
      );
    }
    
    // Apply status filter
    if (filterStatus !== 'all') {
      const isActive = filterStatus === 'active';
      filtered = filtered.filter(strategy => 
        (isActive && strategy.status === 'ACTIVE') ||
        (!isActive && strategy.status === 'PAUSED')
      );
    }
    
    setFilteredStrategies(filtered);
  };

  // Handle refresh button click
  const handleRefresh = () => {
    fetchStrategies();
  };

  // Handle filter changes
  const handleFilterModeChange = (event) => {
    setFilterMode(event.target.value);
  };

  const handleFilterStatusChange = (event) => {
    setFilterStatus(event.target.value);
  };

  // Handle control toggle
  const handleControlToggle = (type) => {
    setShowControls({
      ...showControls,
      [type]: !showControls[type]
    });
  };

  // Handle timeframe change
  const handleTimeframeChange = (event) => {
    setStatsTimeframe(event.target.value);
  };

  // Handle pause/resume strategy
  const handlePauseResumeStrategy = async (strategy) => {
    const action = strategy.status === 'ACTIVE' ? 'pause' : 'resume';
    try {
      const response = await apiFetch(`/api/strategies/${action}`, {
        method: 'POST',
        body: JSON.stringify({
          strategy_id: strategy.id
        })
      });
      
      if (response.success) {
        // Update strategy in state
        const updatedStrategies = strategies.map(s => {
          if (s.id === strategy.id) {
            return {
              ...s,
              status: action === 'pause' ? 'PAUSED' : 'ACTIVE'
            };
          }
          return s;
        });
        
        setStrategies(updatedStrategies);
      }
    } catch (error) {
      console.error(`Error ${action}ing strategy:`, error);
    }
  };

  // Handle close positions
  const handleClosePositions = async (strategy) => {
    if (!window.confirm(`Are you sure you want to close all positions for ${strategy.name}?`)) {
      return;
    }
    
    try {
      const response = await apiFetch('/api/strategies/close-positions', {
        method: 'POST',
        body: JSON.stringify({
          strategy_id: strategy.id
        })
      });
      
      if (response.success) {
        fetchStrategies(); // Refresh data
      }
    } catch (error) {
      console.error('Error closing positions:', error);
    }
  };

  // Handle bulk actions
  const handleBulkPauseResume = async (pause = true) => {
    const action = pause ? 'pause' : 'resume';
    if (!window.confirm(`Are you sure you want to ${action} all selected strategies?`)) {
      return;
    }
    
    try {
      const response = await apiFetch(`/api/strategies/bulk-${action}`, {
        method: 'POST',
        body: JSON.stringify({
          strategy_ids: selectedStrategyIds
        })
      });
      
      if (response.success) {
        fetchStrategies(); // Refresh data
      }
    } catch (error) {
      console.error(`Error bulk ${action}ing strategies:`, error);
    }
  };

  // Column definitions
  const columns = [
    {
      field: 'name',
      headerName: 'Strategy',
      width: 200,
      renderCell: (params) => {
        const isPaper = params.row.phase === 'PAPER_TRADE';
        return (
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Typography variant="body2">
              {params.value}
            </Typography>
            <Chip
              size="small"
              label={isPaper ? "PAPER" : "LIVE"}
              sx={{
                bgcolor: isPaper ? 'grey.400' : 'success.main',
                color: 'white',
                fontWeight: 'bold'
              }}
            />
          </Box>
        );
      }
    },
    {
      field: 'status',
      headerName: 'Status',
      width: 120,
      renderCell: (params) => {
        const isActive = params.value === 'ACTIVE';
        return (
          <Chip
            size="small"
            label={isActive ? "ACTIVE" : "PAUSED"}
            color={isActive ? "primary" : "default"}
            sx={{ fontWeight: 'bold' }}
          />
        );
      }
    },
    {
      field: 'asset_class',
      headerName: 'Asset Class',
      width: 130
    },
    {
      field: 'active_positions',
      headerName: 'Positions',
      type: 'number',
      width: 100,
      valueGetter: (params) => {
        return params.row.positions?.length || 0;
      }
    },
    {
      field: 'performance',
      headerName: 'Performance',
      width: 180,
      renderCell: (params) => {
        const metrics = params.row.performance_metrics || {};
        let returnValue, sharpRatio, winRate;
        
        switch (statsTimeframe) {
          case 'daily':
            returnValue = metrics.daily_return_pct || 0;
            sharpRatio = metrics.daily_sharpe || 0;
            winRate = metrics.daily_win_rate || 0;
            break;
          case 'weekly':
            returnValue = metrics.weekly_return_pct || 0;
            sharpRatio = metrics.weekly_sharpe || 0;
            winRate = metrics.weekly_win_rate || 0;
            break;
          case 'monthly':
            returnValue = metrics.monthly_return_pct || 0;
            sharpRatio = metrics.monthly_sharpe || 0;
            winRate = metrics.monthly_win_rate || 0;
            break;
          default:
            returnValue = metrics.total_return_pct || 0;
            sharpRatio = metrics.sharpe || 0;
            winRate = metrics.win_rate || 0;
        }
        
        return (
          <Box sx={{ display: 'flex', flexDirection: 'column' }}>
            <Typography
              variant="body2"
              color={returnValue >= 0 ? 'success.main' : 'error.main'}
              fontWeight="bold"
            >
              {returnValue.toFixed(2)}%
            </Typography>
            <Typography variant="caption" color="text.secondary">
              SR: {sharpRatio.toFixed(2)} | WR: {(winRate * 100).toFixed(0)}%
            </Typography>
          </Box>
        );
      }
    },
    {
      field: 'actions',
      headerName: 'Actions',
      width: 220,
      renderCell: (params) => {
        const isPaper = params.row.phase === 'PAPER_TRADE';
        const isActive = params.row.status === 'ACTIVE';
        
        // Only show appropriate controls based on settings
        if ((isPaper && !showControls.paper) || (!isPaper && !showControls.live)) {
          return null;
        }
        
        return (
          <Box sx={{ display: 'flex', gap: 1 }}>
            <Tooltip title={isActive ? "Pause Strategy" : "Resume Strategy"}>
              <IconButton
                size="small"
                color={isActive ? "primary" : "default"}
                onClick={() => handlePauseResumeStrategy(params.row)}
              >
                {isActive ? <Pause /> : <PlayArrow />}
              </IconButton>
            </Tooltip>
            
            <Tooltip title="Close All Positions">
              <IconButton
                size="small"
                color="error"
                onClick={() => handleClosePositions(params.row)}
                // For safety: Add extra confirmation for live positions
                disabled={params.row.positions?.length === 0}
              >
                <Close />
              </IconButton>
            </Tooltip>
            
            <Tooltip title="View Details">
              <IconButton
                size="small"
                color="info"
                onClick={() => window.location.href = `/strategies/${params.row.id}`}
              >
                <Info />
              </IconButton>
            </Tooltip>
            
            {!isPaper && (
              <Tooltip title="View Performance Analytics">
                <IconButton
                  size="small"
                  color="success"
                  onClick={() => window.location.href = `/analytics/strategy/${params.row.id}`}
                >
                  <Analytics />
                </IconButton>
              </Tooltip>
            )}
          </Box>
        );
      }
    }
  ];

  return (
    <Box sx={{ padding: 2 }}>
      <Paper elevation={2} sx={{ padding: 2, mb: 2 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <Typography variant="h5">Active Strategies Monitor</Typography>
          <Button 
            variant="outlined" 
            startIcon={<Refresh />}
            onClick={handleRefresh}
          >
            Refresh
          </Button>
        </Box>
        
        <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 2 }}>
          <Box sx={{ display: 'flex', gap: 2 }}>
            {/* Filter Controls */}
            <FormControl size="small" sx={{ minWidth: 120 }}>
              <Select
                value={filterMode}
                onChange={handleFilterModeChange}
                displayEmpty
                startAdornment={<FilterAlt fontSize="small" sx={{ mr: 1 }} />}
              >
                <MenuItem value="all">All Modes</MenuItem>
                <MenuItem value="paper">Paper Only</MenuItem>
                <MenuItem value="live">Live Only</MenuItem>
              </Select>
            </FormControl>
            
            <FormControl size="small" sx={{ minWidth: 120 }}>
              <Select
                value={filterStatus}
                onChange={handleFilterStatusChange}
                displayEmpty
              >
                <MenuItem value="all">All Status</MenuItem>
                <MenuItem value="active">Active Only</MenuItem>
                <MenuItem value="paused">Paused Only</MenuItem>
              </Select>
            </FormControl>
          </Box>
          
          <Box sx={{ display: 'flex', gap: 2, alignItems: 'center' }}>
            {/* Display Controls */}
            <FormGroup row>
              <FormControlLabel
                control={
                  <Switch
                    checked={showControls.paper}
                    onChange={() => handleControlToggle('paper')}
                    size="small"
                  />
                }
                label={
                  <Box sx={{ display: 'flex', alignItems: 'center' }}>
                    <Chip size="small" label="PAPER" sx={{ bgcolor: 'grey.400', color: 'white', fontWeight: 'bold', mr: 0.5 }} />
                    <Typography variant="body2">Controls</Typography>
                  </Box>
                }
              />
              <FormControlLabel
                control={
                  <Switch
                    checked={showControls.live}
                    onChange={() => handleControlToggle('live')}
                    size="small"
                  />
                }
                label={
                  <Box sx={{ display: 'flex', alignItems: 'center' }}>
                    <Chip size="small" label="LIVE" sx={{ bgcolor: 'success.main', color: 'white', fontWeight: 'bold', mr: 0.5 }} />
                    <Typography variant="body2">Controls</Typography>
                  </Box>
                }
              />
            </FormGroup>
            
            <FormControl size="small" sx={{ minWidth: 120 }}>
              <Select
                value={statsTimeframe}
                onChange={handleTimeframeChange}
                displayEmpty
              >
                <MenuItem value="daily">Daily Stats</MenuItem>
                <MenuItem value="weekly">Weekly Stats</MenuItem>
                <MenuItem value="monthly">Monthly Stats</MenuItem>
                <MenuItem value="total">Total Stats</MenuItem>
              </Select>
            </FormControl>
          </Box>
        </Box>
        
        {/* Bulk Actions */}
        {selectedStrategyIds.length > 0 && (
          <Box sx={{ display: 'flex', gap: 1, mb: 2, p: 1, bgcolor: 'grey.100', borderRadius: 1 }}>
            <Typography variant="body2" sx={{ mr: 2 }}>
              {selectedStrategyIds.length} strategies selected
            </Typography>
            <Button
              size="small"
              startIcon={<Pause />}
              onClick={() => handleBulkPauseResume(true)}
            >
              Pause All
            </Button>
            <Button
              size="small"
              startIcon={<PlayArrow />}
              onClick={() => handleBulkPauseResume(false)}
            >
              Resume All
            </Button>
            <Button
              size="small"
              color="error"
              startIcon={<Block />}
              onClick={() => {
                if (window.confirm(`Are you sure you want to stop all selected strategies?`)) {
                  // Implement stop functionality
                }
              }}
              sx={{ ml: 'auto' }}
            >
              Emergency Stop
            </Button>
          </Box>
        )}
      </Paper>

      {/* Strategies DataGrid */}
      <Paper elevation={2} sx={{ padding: 2 }}>
        <Box sx={{ height: 500, width: '100%' }}>
          {loading ? (
            <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%' }}>
              <CircularProgress />
            </Box>
          ) : (
            <DataGrid
              rows={filteredStrategies}
              columns={columns}
              pageSize={10}
              rowsPerPageOptions={[5, 10, 25]}
              checkboxSelection
              onSelectionModelChange={(newSelection) => {
                setSelectedStrategyIds(newSelection);
              }}
              selectionModel={selectedStrategyIds}
              density="compact"
              autoHeight
              disableSelectionOnClick
              getRowClassName={(params) => {
                if (params.row.phase === 'PAPER_TRADE') {
                  return 'paper-strategy-row';
                } else if (params.row.phase === 'LIVE') {
                  return 'live-strategy-row';
                }
                return '';
              }}
              sx={{
                '& .paper-strategy-row': {
                  bgcolor: 'rgba(0, 0, 0, 0.04)'
                },
                '& .live-strategy-row': {
                  bgcolor: 'rgba(76, 175, 80, 0.08)'
                }
              }}
            />
          )}
        </Box>
        
        {/* Legend */}
        <Box sx={{ display: 'flex', justifyContent: 'flex-end', mt: 2, gap: 2 }}>
          <Box sx={{ display: 'flex', alignItems: 'center' }}>
            <Box sx={{ width: 16, height: 16, bgcolor: 'rgba(0, 0, 0, 0.04)', mr: 1, borderRadius: '2px' }} />
            <Typography variant="body2">Paper Trading</Typography>
          </Box>
          <Box sx={{ display: 'flex', alignItems: 'center' }}>
            <Box sx={{ width: 16, height: 16, bgcolor: 'rgba(76, 175, 80, 0.08)', mr: 1, borderRadius: '2px' }} />
            <Typography variant="body2">Live Trading</Typography>
          </Box>
        </Box>
        
        {/* Safety Warning for Live Strategies */}
        {filteredStrategies.some(s => s.phase === 'LIVE') && (
          <Box sx={{ mt: 2, p: 1.5, bgcolor: 'warning.light', borderRadius: 1, display: 'flex', alignItems: 'center' }}>
            <Warning sx={{ mr: 1, color: 'warning.dark' }} />
            <Typography variant="body2">
              <strong>Safety Notice:</strong> Live trading strategies are using real money. Use caution when 
              making changes to these strategies.
            </Typography>
          </Box>
        )}
      </Paper>
    </Box>
  );
};

export default ActiveStrategiesMonitor;
