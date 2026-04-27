# Stage 1: Build frontend
FROM oven/bun:1-alpine AS frontend
WORKDIR /app/web
COPY web/package.json web/bun.lock* ./
RUN bun install --frozen-lockfile
COPY web/ ./
RUN bun run build

# Stage 2: Build Go binary
FROM golang:1.25-alpine AS backend
WORKDIR /app
RUN apk add --no-cache build-base
COPY go.mod go.sum ./
RUN go mod download
COPY . .
COPY --from=frontend /app/web/dist ./web/dist
RUN CGO_ENABLED=1 go build -ldflags="-s -w" -o copilot-go .

# Stage 3: Runtime
FROM alpine:3.21
RUN apk add --no-cache ca-certificates tzdata libstdc++
WORKDIR /app
COPY --from=backend /app/copilot-go .

EXPOSE 3000 4141
VOLUME /root/.local/share/copilot-api

ENTRYPOINT ["./copilot-go"]