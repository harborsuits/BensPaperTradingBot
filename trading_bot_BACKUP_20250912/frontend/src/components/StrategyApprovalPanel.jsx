import React, { useState, useEffect } from 'react';
import { 
  Box, 
  Button, 
  Card, 
  CardContent, 
  CircularProgress, 
  Dialog,
  DialogActions,
  DialogContent,
  DialogContentText,
  DialogTitle,
  Divider,
  FormControl,
  FormControlLabel,
  FormLabel,
  Grid,
  IconButton,
  InputAdornment,
  MenuItem,
  Paper,
  Radio,
  RadioGroup,
  Select,
  Snackbar,
  Tab,
  Tabs,
  TextField,
  Typography,
  Alert
} from '@mui/material';
import {
  Check,
  Close,
  ArrowUpward,
  ArrowDownward,
  Refresh,
  Warning
} from '@mui/icons-material';
import { DataGrid } from '@mui/x-data-grid';
import { apiFetch } from '../utils/api';

/**
 * Strategy Approval Panel Component
 * 
 * Provides an interface for managing the approval workflow:
 * - View strategies in paper mode eligible for approval
 * - View pending approval requests
 * - Approve or reject strategies for live trading
 * - Manage position transitions
 * - View approval history
 */
const StrategyApprovalPanel = () => {
  const [loading, setLoading] = useState(false);
  const [tabValue, setTabValue] = useState(0);
  const [eligibleStrategies, setEligibleStrategies] = useState([]);
  const [pendingApprovals, setPendingApprovals] = useState([]);
  const [approvalHistory, setApprovalHistory] = useState([]);
  const [selectedStrategy, setSelectedStrategy] = useState(null);
  const [approvalDialogOpen, setApprovalDialogOpen] = useState(false);
  const [rejectionDialogOpen, setRejectionDialogOpen] = useState(false);
  const [transitionMode, setTransitionMode] = useState('CLOSE_PAPER_START_FLAT');
  const [positionLimit, setPositionLimit] = useState(2);
  const [liveBroker, setLiveBroker] = useState('');
  const [rejectionReason, setRejectionReason] = useState('');
  const [brokers, setBrokers] = useState([]);
  const [notification, setNotification] = useState({ open: false, message: '', severity: 'info' });

  // Load data on component mount
  useEffect(() => {
    fetchEligibleStrategies();
    fetchPendingApprovals();
    fetchApprovalHistory();
    fetchBrokers();
  }, []);

  // Fetch available live brokers
  const fetchBrokers = async () => {
    try {
      const response = await apiFetch('/api/brokers/list');
      if (response.success) {
        // Filter to only include live brokers (not paper)
        const liveBrokers = response.data.brokers.filter(broker => 
          !broker.id.includes('paper') && broker.status === 'connected'
        );
        setBrokers(liveBrokers);
        if (liveBrokers.length > 0) {
          setLiveBroker(liveBrokers[0].id);
        }
      }
    } catch (error) {
      console.error('Error fetching brokers:', error);
      showNotification('Error fetching brokers', 'error');
    }
  };

  // Fetch strategies eligible for approval
  const fetchEligibleStrategies = async () => {
    setLoading(true);
    try {
      const response = await apiFetch('/api/strategies/approval/eligible');
      if (response.success) {
        setEligibleStrategies(response.data.strategies || []);
      } else {
        showNotification(response.message, 'error');
      }
    } catch (error) {
      console.error('Error fetching eligible strategies:', error);
      showNotification('Error fetching eligible strategies', 'error');
    } finally {
      setLoading(false);
    }
  };

  // Fetch pending approval requests
  const fetchPendingApprovals = async () => {
    try {
      const response = await apiFetch('/api/strategies/approval/pending');
      if (response.success) {
        // Convert object to array for DataGrid
        const pendingArray = Object.entries(response.data.pending_approvals || {}).map(
          ([id, data]) => ({
            id,
            strategy_id: id,
            ...data
          })
        );
        setPendingApprovals(pendingArray);
      }
    } catch (error) {
      console.error('Error fetching pending approvals:', error);
    }
  };

  // Fetch approval history
  const fetchApprovalHistory = async () => {
    try {
      const response = await apiFetch('/api/strategies/approval/history');
      if (response.success) {
        setApprovalHistory(response.data.history || []);
      }
    } catch (error) {
      console.error('Error fetching approval history:', error);
    }
  };

  // Handle tab change
  const handleTabChange = (event, newValue) => {
    setTabValue(newValue);
  };

  // Open approval dialog
  const handleOpenApprovalDialog = (strategy) => {
    setSelectedStrategy(strategy);
    setApprovalDialogOpen(true);
  };

  // Open rejection dialog
  const handleOpenRejectionDialog = (strategy) => {
    setSelectedStrategy(strategy);
    setRejectionDialogOpen(true);
  };

  // Close dialogs
  const handleCloseDialogs = () => {
    setApprovalDialogOpen(false);
    setRejectionDialogOpen(false);
    setSelectedStrategy(null);
    setRejectionReason('');
  };

  // Submit approval request
  const handleRequestApproval = async (strategy) => {
    try {
      const response = await apiFetch('/api/strategies/approval/request', {
        method: 'POST',
        body: JSON.stringify({
          strategy_id: strategy.id,
          requested_by: localStorage.getItem('username') || 'unknown_user',
          notes: `Requesting approval for ${strategy.name}`
        })
      });

      if (response.success) {
        showNotification(`Approval requested for ${strategy.name}`, 'success');
        fetchEligibleStrategies();
        fetchPendingApprovals();
      } else {
        showNotification(response.message, 'error');
      }
    } catch (error) {
      console.error('Error requesting approval:', error);
      showNotification('Error requesting approval', 'error');
    }
  };

  // Submit strategy approval
  const handleApproveStrategy = async () => {
    if (!selectedStrategy) return;

    try {
      const response = await apiFetch('/api/strategies/approval/approve', {
        method: 'POST',
        body: JSON.stringify({
          strategy_id: selectedStrategy.id || selectedStrategy.strategy_id,
          approved_by: localStorage.getItem('username') || 'unknown_user',
          notes: `Approved with ${positionLimit}% position limit`,
          position_transition_mode: transitionMode,
          position_limit_pct: positionLimit / 100, // Convert to decimal
          live_broker_id: liveBroker
        })
      });

      if (response.success) {
        showNotification(`Strategy ${selectedStrategy.name || selectedStrategy.strategy_id} approved for live trading`, 'success');
        handleCloseDialogs();
        // Refresh all data
        fetchEligibleStrategies();
        fetchPendingApprovals();
        fetchApprovalHistory();
      } else {
        showNotification(response.message, 'error');
      }
    } catch (error) {
      console.error('Error approving strategy:', error);
      showNotification('Error approving strategy', 'error');
    }
  };

  // Submit approval rejection
  const handleRejectApproval = async () => {
    if (!selectedStrategy || !rejectionReason) {
      showNotification('Please provide a rejection reason', 'warning');
      return;
    }

    try {
      const response = await apiFetch('/api/strategies/approval/reject', {
        method: 'POST',
        body: JSON.stringify({
          strategy_id: selectedStrategy.id || selectedStrategy.strategy_id,
          rejected_by: localStorage.getItem('username') || 'unknown_user',
          reason: rejectionReason
        })
      });

      if (response.success) {
        showNotification(`Approval rejected for ${selectedStrategy.name || selectedStrategy.strategy_id}`, 'info');
        handleCloseDialogs();
        fetchPendingApprovals();
        fetchApprovalHistory();
      } else {
        showNotification(response.message, 'error');
      }
    } catch (error) {
      console.error('Error rejecting approval:', error);
      showNotification('Error rejecting approval', 'error');
    }
  };

  // Show notification
  const showNotification = (message, severity = 'info') => {
    setNotification({
      open: true,
      message,
      severity
    });
  };

  // Close notification
  const handleCloseNotification = () => {
    setNotification({ ...notification, open: false });
  };

  // Refresh data
  const handleRefresh = () => {
    fetchEligibleStrategies();
    fetchPendingApprovals();
    fetchApprovalHistory();
  };

  // Column definitions for eligible strategies table
  const eligibleColumns = [
    { field: 'id', headerName: 'ID', width: 120 },
    { field: 'name', headerName: 'Name', width: 200 },
    { field: 'status', headerName: 'Status', width: 150 },
    { field: 'phase', headerName: 'Phase', width: 150 },
    {
      field: 'performance',
      headerName: 'Performance',
      width: 150,
      renderCell: (params) => {
        const metrics = params.row.performance_metrics || {};
        const winRate = metrics.win_rate || 0;
        const totalReturn = metrics.total_return_pct || 0;
        
        return (
          <Box>
            <Typography variant="body2" color={totalReturn >= 0 ? 'success.main' : 'error.main'}>
              {totalReturn.toFixed(2)}% / {(winRate * 100).toFixed(0)}% WR
            </Typography>
          </Box>
        );
      }
    },
    {
      field: 'actions',
      headerName: 'Actions',
      width: 200,
      renderCell: (params) => (
        <Button
          variant="contained"
          color="primary"
          size="small"
          onClick={() => handleRequestApproval(params.row)}
        >
          Request Approval
        </Button>
      )
    }
  ];

  // Column definitions for pending approvals table
  const pendingColumns = [
    { field: 'strategy_id', headerName: 'Strategy ID', width: 150 },
    { 
      field: 'requested_by',
      headerName: 'Requested By',
      width: 150 
    },
    {
      field: 'requested_at',
      headerName: 'Requested At',
      width: 200,
      valueFormatter: (params) => {
        return new Date(params.value).toLocaleString();
      }
    },
    { field: 'notes', headerName: 'Notes', width: 250 },
    {
      field: 'actions',
      headerName: 'Actions',
      width: 200,
      renderCell: (params) => (
        <Box>
          <Button
            variant="contained"
            color="success"
            size="small"
            onClick={() => handleOpenApprovalDialog(params.row)}
            sx={{ mr: 1 }}
          >
            Approve
          </Button>
          <Button
            variant="contained"
            color="error"
            size="small"
            onClick={() => handleOpenRejectionDialog(params.row)}
          >
            Reject
          </Button>
        </Box>
      )
    }
  ];

  // Column definitions for approval history table
  const historyColumns = [
    { field: 'strategy_id', headerName: 'Strategy ID', width: 150 },
    { field: 'requested_by', headerName: 'Requested By', width: 150 },
    {
      field: 'status',
      headerName: 'Status',
      width: 120,
      renderCell: (params) => {
        const status = params.row.status;
        let color;
        
        switch (status) {
          case 'approved':
            color = 'success.main';
            break;
          case 'rejected':
            color = 'error.main';
            break;
          default:
            color = 'text.secondary';
        }
        
        return (
          <Typography variant="body2" sx={{ color }}>
            {status.toUpperCase()}
          </Typography>
        );
      }
    },
    {
      field: 'approved_by',
      headerName: 'Approved By',
      width: 150,
      valueGetter: (params) => params.row.approved_by || 'N/A'
    },
    {
      field: 'approved_at',
      headerName: 'Approved At', 
      width: 200,
      valueGetter: (params) => params.row.approved_at ? new Date(params.row.approved_at).toLocaleString() : 'N/A'
    },
    {
      field: 'rejected_by',
      headerName: 'Rejected By',
      width: 150,
      valueGetter: (params) => params.row.rejected_by || 'N/A'
    },
    {
      field: 'rejection_reason',
      headerName: 'Rejection Reason',
      width: 200,
      valueGetter: (params) => params.row.rejection_reason || 'N/A'
    }
  ];

  return (
    <Box sx={{ padding: 2 }}>
      <Paper elevation={2} sx={{ padding: 2, mb: 2 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <Typography variant="h5">Strategy Approval Workflow</Typography>
          <Button 
            variant="outlined" 
            startIcon={<Refresh />}
            onClick={handleRefresh}
          >
            Refresh
          </Button>
        </Box>

        <Tabs value={tabValue} onChange={handleTabChange}>
          <Tab label="Eligible Strategies" />
          <Tab 
            label={
              <Box sx={{ display: 'flex', alignItems: 'center' }}>
                <span>Pending Approvals</span>
                {pendingApprovals.length > 0 && (
                  <Box
                    sx={{
                      ml: 1,
                      bgcolor: 'warning.main',
                      color: 'warning.contrastText',
                      borderRadius: '50%',
                      width: 20,
                      height: 20,
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      fontSize: '0.75rem'
                    }}
                  >
                    {pendingApprovals.length}
                  </Box>
                )}
              </Box>
            } 
          />
          <Tab label="Approval History" />
        </Tabs>
      </Paper>

      {/* Eligible Strategies Tab */}
      {tabValue === 0 && (
        <Paper elevation={2} sx={{ padding: 2 }}>
          <Typography variant="h6" sx={{ mb: 2 }}>
            Paper Trading Strategies Eligible for Approval
          </Typography>
          <Box sx={{ height: 400, width: '100%' }}>
            {loading ? (
              <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%' }}>
                <CircularProgress />
              </Box>
            ) : (
              <DataGrid
                rows={eligibleStrategies}
                columns={eligibleColumns}
                pageSize={5}
                rowsPerPageOptions={[5, 10, 20]}
                disableSelectionOnClick
                autoHeight
              />
            )}
          </Box>
        </Paper>
      )}

      {/* Pending Approvals Tab */}
      {tabValue === 1 && (
        <Paper elevation={2} sx={{ padding: 2 }}>
          <Typography variant="h6" sx={{ mb: 2 }}>
            Pending Approval Requests
          </Typography>
          <Box sx={{ height: 400, width: '100%' }}>
            {pendingApprovals.length === 0 ? (
              <Box sx={{ 
                display: 'flex', 
                justifyContent: 'center', 
                alignItems: 'center', 
                height: 200,
                flexDirection: 'column' 
              }}>
                <Check sx={{ fontSize: 48, color: 'success.main', mb: 2 }} />
                <Typography variant="h6" color="text.secondary">
                  No pending approval requests
                </Typography>
              </Box>
            ) : (
              <DataGrid
                rows={pendingApprovals}
                columns={pendingColumns}
                pageSize={5}
                rowsPerPageOptions={[5, 10, 20]}
                disableSelectionOnClick
                autoHeight
              />
            )}
          </Box>
        </Paper>
      )}

      {/* Approval History Tab */}
      {tabValue === 2 && (
        <Paper elevation={2} sx={{ padding: 2 }}>
          <Typography variant="h6" sx={{ mb: 2 }}>
            Approval History
          </Typography>
          <Box sx={{ height: 400, width: '100%' }}>
            {approvalHistory.length === 0 ? (
              <Box sx={{ 
                display: 'flex', 
                justifyContent: 'center', 
                alignItems: 'center', 
                height: 200,
                flexDirection: 'column' 
              }}>
                <Typography variant="h6" color="text.secondary">
                  No approval history yet
                </Typography>
              </Box>
            ) : (
              <DataGrid
                rows={approvalHistory.map((item, index) => ({ ...item, id: index }))}
                columns={historyColumns}
                pageSize={5}
                rowsPerPageOptions={[5, 10, 20]}
                disableSelectionOnClick
                autoHeight
              />
            )}
          </Box>
        </Paper>
      )}

      {/* Approval Dialog */}
      <Dialog
        open={approvalDialogOpen}
        onClose={handleCloseDialogs}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>
          Approve Strategy for Live Trading
        </DialogTitle>
        <DialogContent>
          <DialogContentText sx={{ mb: 2 }}>
            You are about to approve the strategy <strong>{selectedStrategy?.name || selectedStrategy?.strategy_id}</strong> for live trading. 
            This will transition the strategy from paper trading to live trading with real money.
          </DialogContentText>
          
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <FormControl fullWidth sx={{ mb: 2 }}>
                <FormLabel>Position Transition Mode</FormLabel>
                <RadioGroup
                  value={transitionMode}
                  onChange={(e) => setTransitionMode(e.target.value)}
                >
                  <FormControlLabel 
                    value="CLOSE_PAPER_START_FLAT" 
                    control={<Radio />} 
                    label="Close paper positions, start live flat" 
                  />
                  <FormControlLabel 
                    value="MIRROR_TO_LIVE" 
                    control={<Radio />} 
                    label="Mirror paper positions to live account" 
                  />
                  <FormControlLabel 
                    value="WAIT_FOR_FLAT" 
                    control={<Radio />} 
                    label="Only allow if strategy has no open positions" 
                  />
                </RadioGroup>
              </FormControl>
            </Grid>
            
            <Grid item xs={12} md={6}>
              <FormControl fullWidth sx={{ mb: 2 }}>
                <FormLabel>Position Size Limit</FormLabel>
                <TextField
                  value={positionLimit}
                  onChange={(e) => setPositionLimit(Number(e.target.value))}
                  type="number"
                  InputProps={{
                    endAdornment: <InputAdornment position="end">%</InputAdornment>,
                  }}
                  inputProps={{
                    min: 0,
                    max: 100,
                    step: 0.1
                  }}
                  helperText="Maximum position size as % of portfolio"
                />
              </FormControl>
              
              <FormControl fullWidth sx={{ mt: 2 }}>
                <FormLabel>Live Broker</FormLabel>
                <Select
                  value={liveBroker}
                  onChange={(e) => setLiveBroker(e.target.value)}
                >
                  {brokers.map((broker) => (
                    <MenuItem key={broker.id} value={broker.id}>
                      {broker.name} ({broker.id})
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>
          </Grid>
          
          <Box sx={{ mt: 2, p: 2, bgcolor: 'warning.light', borderRadius: 1 }}>
            <Typography variant="body2" sx={{ display: 'flex', alignItems: 'center' }}>
              <Warning sx={{ mr: 1 }} />
              <strong>Warning:</strong> This will allow the strategy to trade with real money. 
              Make sure you have reviewed its performance thoroughly.
            </Typography>
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={handleCloseDialogs}>Cancel</Button>
          <Button 
            onClick={handleApproveStrategy} 
            variant="contained" 
            color="primary"
          >
            Approve for Live Trading
          </Button>
        </DialogActions>
      </Dialog>

      {/* Rejection Dialog */}
      <Dialog
        open={rejectionDialogOpen}
        onClose={handleCloseDialogs}
      >
        <DialogTitle>
          Reject Approval Request
        </DialogTitle>
        <DialogContent>
          <DialogContentText sx={{ mb: 2 }}>
            You are rejecting the approval request for <strong>{selectedStrategy?.name || selectedStrategy?.strategy_id}</strong>.
          </DialogContentText>
          <TextField
            autoFocus
            margin="dense"
            label="Reason for Rejection"
            fullWidth
            multiline
            rows={3}
            value={rejectionReason}
            onChange={(e) => setRejectionReason(e.target.value)}
            required
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={handleCloseDialogs}>Cancel</Button>
          <Button 
            onClick={handleRejectApproval} 
            variant="contained" 
            color="error"
            disabled={!rejectionReason}
          >
            Reject Request
          </Button>
        </DialogActions>
      </Dialog>

      {/* Notification */}
      <Snackbar
        open={notification.open}
        autoHideDuration={6000}
        onClose={handleCloseNotification}
      >
        <Alert 
          onClose={handleCloseNotification} 
          severity={notification.severity}
        >
          {notification.message}
        </Alert>
      </Snackbar>
    </Box>
  );
};

export default StrategyApprovalPanel;
