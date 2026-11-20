# DermAI Production Frontend Dockerfile
# Stage 1: Build static distribution
FROM node:20-alpine AS builder

WORKDIR /app

# Install dependencies with lockfile caching
COPY package*.json ./
RUN npm ci

# Copy source code and build production assets
COPY . .
RUN npm run build

# Stage 2: Serve with lightweight Nginx
FROM nginx:1.27-alpine AS runner

# Copy customized Nginx configuration
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Copy compiled assets from builder
COPY --from=builder /app/dist /usr/share/nginx/html

EXPOSE 80

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD wget -q -O /dev/null http://localhost:80/ || exit 1

CMD ["nginx", "-g", "daemon off;"]
