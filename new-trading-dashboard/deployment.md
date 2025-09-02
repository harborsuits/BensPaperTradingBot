# BenBot Trading Dashboard Deployment Guide

This guide outlines how to deploy the BenBot Trading Dashboard for production use.

## Environment Configuration

Create a `.env.production` file in the root directory with the following variables (replace with your actual values):

```
# API Configuration 
REACT_APP_API_URL=https://api.yourdomain.com
REACT_APP_WS_URL=wss://api.yourdomain.com

# Feature Flags
REACT_APP_ENABLE_DEMO_MODE=false
REACT_APP_ENABLE_LOGS=true
REACT_APP_ENABLE_REAL_TRADING=true

# Performance Settings
REACT_APP_WS_RECONNECT_INTERVAL=5000
REACT_APP_DATA_REFRESH_INTERVAL=60000
REACT_APP_MAX_LOG_ENTRIES=500

# Authentication
REACT_APP_AUTH_TOKEN_KEY=benbot_prod_auth_token
REACT_APP_AUTH_EXPIRY_KEY=benbot_prod_auth_expiry
```

## Build Process

1. Install dependencies:
```bash
npm install
```

2. Build for production:
```bash
npm run build
```

This will create a `build` folder with all static files optimized for production.

## Deployment Options

### Option 1: Nginx Server

1. Install Nginx on your server
2. Copy the contents of the `build` folder to your Nginx web root (typically `/var/www/html`)
3. Configure Nginx to serve the React application:

```nginx
server {
    listen 80;
    server_name yourdomain.com;
    root /var/www/html;
    index index.html;
    
    location / {
        try_files $uri $uri/ /index.html;
    }
    
    # Configure API proxy if needed
    location /api {
        proxy_pass http://your-backend-server:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }
    
    # Configure WebSocket proxy
    location /ws {
        proxy_pass http://your-backend-server:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
    }
}
```

4. Restart Nginx:
```bash
sudo systemctl restart nginx
```

### Option 2: Docker Deployment

1. Create a Dockerfile in the project root:

```dockerfile
FROM node:18-alpine as build
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=build /app/build /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

2. Create nginx.conf:

```nginx
server {
    listen 80;
    root /usr/share/nginx/html;
    index index.html;
    
    location / {
        try_files $uri $uri/ /index.html;
    }
}
```

3. Build and run the Docker container:
```bash
docker build -t benbot-trading-dashboard .
docker run -p 80:80 benbot-trading-dashboard
```

### Option 3: Netlify/Vercel Deployment

1. Connect your GitHub repository to Netlify or Vercel
2. Configure the build settings:
   - Build command: `npm run build`
   - Publish directory: `build`
3. Configure environment variables in the platform's dashboard
4. Deploy the application

## Performance Optimizations for Production

1. Enable GZIP compression on your web server
2. Configure proper cache headers:

```nginx
location ~* \.(js|css|png|jpg|jpeg|gif|svg|ico)$ {
    expires 7d;
    add_header Cache-Control "public, max-age=604800";
}
```

3. Set up a Content Delivery Network (CDN) for static assets
4. Enable HTTP/2 for better performance:

```nginx
server {
    listen 443 ssl http2;
    # SSL configuration...
}
```

## Monitoring and Logging

1. Set up application monitoring using tools like Sentry or LogRocket
2. Configure server monitoring for your hosting environment
3. Set up log rotation for Nginx logs
4. Implement health check endpoints for your backend

## Security Considerations

1. Always use HTTPS in production
2. Set secure and HTTP-only flags on cookies
3. Implement proper rate limiting on your API
4. Use a Web Application Firewall (WAF)
5. Regularly update dependencies to patch security vulnerabilities
6. Implement Content Security Policy (CSP) headers

## Backup and Disaster Recovery

1. Set up regular backups of your database and configuration
2. Document the restore process
3. Set up monitoring and alerts for system failures
4. Test your disaster recovery plan periodically

For any questions or issues regarding the deployment process, please contact the development team.
