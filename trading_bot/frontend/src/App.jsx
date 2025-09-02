import React from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import { 
  AppBar, 
  Box, 
  CssBaseline, 
  Divider, 
  Drawer, 
  IconButton, 
  List, 
  ListItem, 
  ListItemButton, 
  ListItemIcon, 
  ListItemText, 
  Toolbar, 
  Typography, 
  useTheme 
} from '@mui/material';
import {
  Menu as MenuIcon,
  Dashboard as DashboardIcon,
  BarChart as BarChartIcon,
  Settings as SettingsIcon,
  Sync as SyncIcon,
  MonitorHeart as MonitorHeartIcon,
  ThumbUpAlt as ThumbUpIcon
} from '@mui/icons-material';

// Import our strategy management components
import StrategyApprovalPanel from './components/StrategyApprovalPanel';
import ActiveStrategiesMonitor from './components/ActiveStrategiesMonitor';

// Placeholder components for other sections
const Dashboard = () => <Box p={3}><Typography variant="h4">Dashboard</Typography></Box>;
const Performance = () => <Box p={3}><Typography variant="h4">Performance Analytics</Typography></Box>;
const Settings = () => <Box p={3}><Typography variant="h4">Settings</Typography></Box>;

const drawerWidth = 240;

function App() {
  const [mobileOpen, setMobileOpen] = React.useState(false);
  const theme = useTheme();

  const handleDrawerToggle = () => {
    setMobileOpen(!mobileOpen);
  };

  const drawer = (
    <div>
      <Toolbar>
        <Typography variant="h6" noWrap component="div">
          Trading Bot
        </Typography>
      </Toolbar>
      <Divider />
      <List>
        <ListItem disablePadding component={Link} to="/" style={{ textDecoration: 'none', color: 'inherit' }}>
          <ListItemButton>
            <ListItemIcon>
              <DashboardIcon />
            </ListItemIcon>
            <ListItemText primary="Dashboard" />
          </ListItemButton>
        </ListItem>
        
        <ListItem disablePadding component={Link} to="/strategies" style={{ textDecoration: 'none', color: 'inherit' }}>
          <ListItemButton>
            <ListItemIcon>
              <MonitorHeartIcon />
            </ListItemIcon>
            <ListItemText primary="Strategy Monitor" />
          </ListItemButton>
        </ListItem>
        
        <ListItem disablePadding component={Link} to="/strategy-approval" style={{ textDecoration: 'none', color: 'inherit' }}>
          <ListItemButton>
            <ListItemIcon>
              <ThumbUpIcon />
            </ListItemIcon>
            <ListItemText primary="Strategy Approval" />
          </ListItemButton>
        </ListItem>
        
        <ListItem disablePadding component={Link} to="/performance" style={{ textDecoration: 'none', color: 'inherit' }}>
          <ListItemButton>
            <ListItemIcon>
              <BarChartIcon />
            </ListItemIcon>
            <ListItemText primary="Performance" />
          </ListItemButton>
        </ListItem>
      </List>
      <Divider />
      <List>
        <ListItem disablePadding component={Link} to="/settings" style={{ textDecoration: 'none', color: 'inherit' }}>
          <ListItemButton>
            <ListItemIcon>
              <SettingsIcon />
            </ListItemIcon>
            <ListItemText primary="Settings" />
          </ListItemButton>
        </ListItem>
      </List>
    </div>
  );

  return (
    <Router>
      <Box sx={{ display: 'flex' }}>
        <CssBaseline />
        <AppBar
          position="fixed"
          sx={{
            width: { sm: `calc(100% - ${drawerWidth}px)` },
            ml: { sm: `${drawerWidth}px` },
          }}
        >
          <Toolbar>
            <IconButton
              color="inherit"
              aria-label="open drawer"
              edge="start"
              onClick={handleDrawerToggle}
              sx={{ mr: 2, display: { sm: 'none' } }}
            >
              <MenuIcon />
            </IconButton>
            <Typography variant="h6" noWrap component="div">
              Trading Bot Control Panel
            </Typography>
          </Toolbar>
        </AppBar>
        <Box
          component="nav"
          sx={{ width: { sm: drawerWidth }, flexShrink: { sm: 0 } }}
          aria-label="navigation"
        >
          <Drawer
            variant="temporary"
            open={mobileOpen}
            onClose={handleDrawerToggle}
            ModalProps={{
              keepMounted: true, // Better open performance on mobile
            }}
            sx={{
              display: { xs: 'block', sm: 'none' },
              '& .MuiDrawer-paper': { boxSizing: 'border-box', width: drawerWidth },
            }}
          >
            {drawer}
          </Drawer>
          <Drawer
            variant="permanent"
            sx={{
              display: { xs: 'none', sm: 'block' },
              '& .MuiDrawer-paper': { boxSizing: 'border-box', width: drawerWidth },
            }}
            open
          >
            {drawer}
          </Drawer>
        </Box>
        <Box
          component="main"
          sx={{ flexGrow: 1, p: 3, width: { sm: `calc(100% - ${drawerWidth}px)` } }}
        >
          <Toolbar />
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/strategies" element={<ActiveStrategiesMonitor />} />
            <Route path="/strategy-approval" element={<StrategyApprovalPanel />} />
            <Route path="/performance" element={<Performance />} />
            <Route path="/settings" element={<Settings />} />
          </Routes>
        </Box>
      </Box>
    </Router>
  );
}

export default App;
