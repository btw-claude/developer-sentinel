# Docker Deployment Example

Production-ready Docker configuration for Developer Sentinel.

## What This Example Provides

- **Dockerfile** - Multi-stage build for minimal image size
- **docker-compose.yml** - Complete deployment configuration
- **Health checks** - Automatic container health monitoring
- **Volume mounts** - Persistent logs and working directories

## Files

```
docker/
├── README.md              # This file
├── Dockerfile             # Multi-stage Docker build
├── docker-compose.yml     # Compose configuration
├── .env.example           # Environment template
└── orchestrations/
    └── example.yaml       # Example orchestration
```

## Quick Start

### 1. Configure Environment

```bash
cp .env.example .env
vim .env
```

### 2. Build and Run

```bash
# Build the image
docker-compose build

# Run in foreground (see logs)
docker-compose up

# Run in background
docker-compose up -d
```

### 3. Monitor

```bash
# View logs
docker-compose logs -f

# Check health status
docker-compose ps

# Stop the service
docker-compose down
```

## Configuration

### Environment Variables

All configuration is done through environment variables. See `.env.example` for the complete list.

Key variables:
- `JIRA_*` - Jira API credentials
- `SENTINEL_*` - Sentinel configuration
- `ANTHROPIC_API_KEY` - Claude API key (required for agent execution)

### Volume Mounts

The docker-compose file mounts these directories:

| Host Path | Container Path | Purpose |
|-----------|----------------|---------|
| `./orchestrations` | `/app/orchestrations` | Orchestration configs |
| `./workdir` | `/app/workdir` | Agent working directories |
| `./logs` | `/app/logs` | Execution logs |

### Health Checks

The container includes health checks that verify:
- Service is running
- Can connect to Jira API
- Agent execution is functional

## Production Considerations

### Security

1. **Never commit `.env` files** - They contain secrets
2. **Use Docker secrets** for production deployments
3. **Restrict network access** - Only allow necessary outbound connections

### Resource Limits

```yaml
# Add to docker-compose.yml for resource limits
services:
  sentinel:
    deploy:
      resources:
        limits:
          memory: 1G
          cpus: '0.5'
```

### Logging

For production, enable JSON logging for log aggregation:

```bash
SENTINEL_LOG_JSON=true
```

### High Availability

For HA deployments, consider:
- Running multiple instances with unique workdir paths
- Using external storage for logs (S3, CloudWatch, etc.)
- Container orchestration (Kubernetes, ECS)

## Building Custom Images

### Add Custom Dependencies

```dockerfile
# Extend the Dockerfile
FROM developer-sentinel:latest

# Install additional packages
RUN pip install your-custom-package
```

### Multi-Architecture Builds

```bash
# Build for multiple architectures
docker buildx build --platform linux/amd64,linux/arm64 -t developer-sentinel:latest .
```

## Troubleshooting

### Container won't start

```bash
# Check logs for errors
docker-compose logs sentinel

# Common issues:
# - Missing environment variables
# - Invalid API credentials
# - Permission issues with volume mounts
```

### Health check failing

```bash
# Check detailed health status
docker inspect --format='{{json .State.Health}}' developer-sentinel

# Verify API connectivity from container
docker-compose exec sentinel curl -I https://your-jira.atlassian.net
```

### Out of memory

```bash
# Increase memory limit
docker-compose up -d --memory=2g

# Or adjust in docker-compose.yml
```

## Next Steps

- Configure orchestrations for your workflows
- Set up monitoring and alerting
- Review the [main documentation](../../README.md) for all options
