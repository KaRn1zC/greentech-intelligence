# =============================================================================
# GreenTech Intelligence — Frontend Dockerfile (multi-stage, nginx)
# =============================================================================
# Build : docker build -t greentech-frontend ./frontend
# Run   : docker run -p 80:80 greentech-frontend
# =============================================================================

# --- Stage 1 : Build de l'application React ---
FROM node:20-alpine AS builder

WORKDIR /app

# Copier les fichiers de dependances en premier (cache Docker)
COPY package.json package-lock.json ./

# Installer les dependances
RUN npm ci --ignore-scripts

# Copier le code source
COPY . .

# Build de production (genere dist/)
RUN npx vite build

# --- Stage 2 : Serveur NGINX pour les fichiers statiques ---
FROM nginx:1.27-alpine AS runtime

# Copier la config NGINX custom
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Copier les fichiers buildes depuis le stage builder
COPY --from=builder /app/dist /usr/share/nginx/html

# Utilisateur non-root (nginx user existe deja dans l'image)
RUN chown -R nginx:nginx /usr/share/nginx/html && \
    chown -R nginx:nginx /var/cache/nginx && \
    chown -R nginx:nginx /var/log/nginx && \
    touch /var/run/nginx.pid && \
    chown -R nginx:nginx /var/run/nginx.pid

USER nginx

EXPOSE 80

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD ["wget", "--no-verbose", "--tries=1", "--spider", "http://localhost:80/", "||", "exit", "1"]

CMD ["nginx", "-g", "daemon off;"]
