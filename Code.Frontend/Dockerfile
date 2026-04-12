FROM node:20-alpine AS build

WORKDIR /srv/app
COPY Code.Frontend/package.json Code.Frontend/package-lock.json ./
RUN npm ci
COPY Code.Frontend/ ./
RUN npm run build

FROM caddy:2.8-alpine

COPY Code.Frontend/Caddyfile /etc/caddy/Caddyfile
COPY --from=build /srv/app/dist /srv
