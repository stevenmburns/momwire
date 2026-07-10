# Build the static Astro/Starlight primer site, then serve it with nginx.
# Deploy from this directory: `cd site && fly deploy` (see fly.toml).

# ---- build ----
FROM node:22-bookworm-slim AS build
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
RUN npm run build      # type-checks then emits the static site to /app/dist

# ---- serve ----
FROM nginx:1.27-alpine AS serve
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
