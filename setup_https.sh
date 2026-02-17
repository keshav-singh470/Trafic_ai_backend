#!/bin/bash

# ANPR System - Production Setup Script (Ubuntu/Debian)
# This script installs Nginx, configures Reverse Proxy, and sets up SSL.

DOMAIN="ai-trafic.servepics.com"
EMAIL="ranjeetray6120@gmail.com" # Updated based on repo owner usually, can ask user
APP_PORT=8000

echo ">>> Updating System..."
sudo apt update && sudo apt install -y nginx certbot python3-certbot-nginx

echo ">>> Creating Nginx Configuration for $DOMAIN..."
sudo tee /etc/nginx/sites-available/anpr-system > /dev/null <<EOL
server {
    server_name $DOMAIN;
    client_max_body_size 100M;

    location / {
        proxy_pass http://127.0.0.1:$APP_PORT;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host \$host;
        proxy_cache_bypass \$http_upgrade;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }
}
EOL

echo ">>> Enabling Configuration..."
sudo ln -sf /etc/nginx/sites-available/anpr-system /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t && sudo systemctl restart nginx

echo ">>> Setting up SSL with Let's Encrypt..."
# Non-interactive mode
sudo certbot --nginx -d $DOMAIN --non-interactive --agree-tos -m "$EMAIL" --redirect

echo ">>> Setup Complete!"
echo "Your API is now served at https://$DOMAIN"
