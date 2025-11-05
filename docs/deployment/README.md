# Deployment Guide

This guide provides comprehensive instructions for deploying the GraphRAG system across different environments, from local development to production.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Local Development Setup](#local-development-setup)
3. [Docker Deployment](#docker-deployment)
4. [Kubernetes Deployment](#kubernetes-deployment)
5. [Railway Deployment](#railway-deployment)
6. [Configuration Management](#configuration-management)
7. [Monitoring and Logging](#monitoring-and-logging)
8. [Scaling](#scaling)
9. [Backup and Recovery](#backup-and-recovery)
10. [Security Considerations](#security-considerations)
11. [Troubleshooting](#troubleshooting)

## Prerequisites

### Hardware Requirements

| Component | Development | Staging | Production |
|-----------|-------------|---------|------------|
| CPU | 4 cores | 8 cores | 16+ cores |
| Memory | 8GB | 16GB | 64GB+ |
| Storage | 50GB | 200GB | 1TB+ |
| Network | 100Mbps | 1Gbps | 10Gbps |

### Software Requirements

- Docker 20.10+
- Docker Compose 2.0+
- Kubernetes 1.22+ (for production)
- Helm 3.0+ (for Kubernetes)
- Terraform 1.0+ (for infrastructure as code)
- Python 3.10+
- Node.js 16+ (for frontend)

## Local Development Setup

### 1. Clone the Repository

```bash
git clone https://github.com/your-org/graphrag-crypto.git
cd graphrag-crypto
```

### 2. Set Up Environment Variables

Create a `.env` file in the project root with the following variables:

```bash
# Application
DEBUG=true
ENVIRONMENT=development

# API
API_V1_STR=/api/v1
API_PREFIX=/api
API_AUTH_REQUIRED=false  # Set to true in production

# Database
POSTGRES_SERVER=db
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_DB=crypto
TIMESCALEDB_URI=postgresql+asyncpg://postgres:postgres@db:5432/crypto

# Redis
REDIS_HOST=redis
REDIS_PORT=6379
REDIS_DB=0

# Neo4j
NEO4J_URI=bolt://neo4j:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password

# LLM Services (optional for local development)
TOGETHER_API_KEY=your_together_api_key
```

### 3. Start Services with Docker Compose

```bash
docker compose up -d --build
```

This will start the following services:
- `app`: FastAPI application
- `db`: TimescaleDB database
- `redis`: Redis for caching and pub/sub
- `neo4j`: Neo4j graph database
- `adminer`: Web-based database management (http://localhost:8080)

### 4. Verify the Installation

1. Check if all services are running:
   ```bash
   docker compose ps
   ```

2. Check the application logs:
   ```bash
   docker compose logs -f app
   ```

3. Test the health check endpoint:
   ```bash
   curl http://localhost:8000/health
   ```

### 5. Initialize the Database

Run database migrations and initialize the schema:

```bash
docker compose exec app alembic upgrade head
```

### 6. Access the Services

- **FastAPI Application**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Adminer (Database UI)**: http://localhost:8080
  - System: PostgreSQL
  - Server: db
  - Username: postgres
  - Password: postgres
  - Database: crypto

## Pydantic v2 Migration Notes

This project uses Pydantic v2 with the following key changes:

1. **Settings Configuration**:
   - Uses `SettingsConfigDict` instead of inner `Config` class
   - Updated field validators to use `@field_validator` decorator

2. **Model Updates**:
   - All models now use `model_config` instead of `Config`
   - Updated type hints and validators for v2 compatibility

3. **Dependencies**:
   - Requires `pydantic-extra-types` for additional field types
   - Updated SQLAlchemy integration for async support

```bash
cp .env.example .env
# Edit .env with your configuration
```

## Local Development Setup

### 1. Clone the Repository

```bash
git clone https://github.com/your-org/graphrag-crypto.git
cd graphrag-crypto
```

This will start:
- Graphiti (Neo4j)
- TimescaleDB
- Redis
- API Service
- Agent Service
- Web Interface

### 4. Initialize the Database

```bash
# Apply database migrations
docker-compose exec api alembic upgrade head

# Load initial data
docker-compose exec api python -m scripts.load_initial_data
```

### 5. Verify the Installation

```bash
# Check service status
docker-compose ps

# Test API
curl http://localhost:8000/v1/health
```

## Docker Deployment

### Production Docker Compose

```bash
# Build and start production services
docker-compose -f docker-compose.prod.yml up -d --build
```

### Environment Variables

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `NEO4J_URI` | Neo4j connection string | Yes | |
| `NEO4J_USER` | Neo4j username | Yes | |
| `NEO4J_PASSWORD` | Neo4j password | Yes | |
| `TIMESCALEDB_URI` | TimescaleDB connection string | Yes | |
| `REDIS_URL` | Redis connection URL | Yes | |
| `TOGETHER_API_KEY` | Together.AI API key | Yes | |
| `JWT_SECRET_KEY` | Secret for JWT token generation | Yes | |
| `LOG_LEVEL` | Logging level | No | INFO |
| `ENVIRONMENT` | Deployment environment | No | development |

## Kubernetes Deployment

### Prerequisites

- Kubernetes cluster
- `kubectl` configured to access the cluster
- Helm installed

### 1. Install Dependencies

```bash
# Add required Helm repositories
helm repo add bitnami https://charts.bitnami.com/bitnami
helm repo add timescale https://charts.timescale.com/
helm repo update

# Install dependencies
helm install redis bitnami/redis --namespace graphrag --create-namespace
helm install timescaledb timescale/timescaledb-single --namespace graphrag
```

### 2. Deploy GraphRAG

```bash
# Create namespace
kubectl create namespace graphrag

# Install with Helm
helm install graphrag ./charts/graphrag \
  --namespace graphrag \
  --set environment=production \
  --set image.tag=latest \
  --set secrets.togetherApiKey="$TOGETHER_API_KEY"
```

### 3. Verify Deployment

```bash
# Check pods
kubectl get pods -n graphrag

# Check services
kubectl get svc -n graphrag

# View logs
kubectl logs -l app.kubernetes.io/name=api -n graphrag
```

## Railway Deployment

### 1. Install Railway CLI

```bash
npm i -g @railway/cli
railway login
```

### 2. Link Project

```bash
railway link $PROJECT_ID
```

### 3. Deploy

```bash
# Set environment variables
railway env set NODE_ENV production
railway env set PORT $PORT

# Deploy
railway up
```

### 4. Set Up Database Plugins

1. Add TimescaleDB plugin
2. Add Redis plugin
3. Add Neo4j plugin
4. Configure environment variables

## Configuration Management

### Environment Configuration

Configuration is managed through environment variables with sensible defaults. Required variables must be set in production.

### Secrets Management

Sensitive values are managed using Kubernetes Secrets or Railway's secret management:

```bash
# Kubernetes
kubectl create secret generic graphrag-secrets \
  --from-literal=neo4j-password=yourpassword \
  --from-literal=together-api-key=yourapikey \
  -n graphrag

# Railway
railway env set NEO4J_PASSWORD=yourpassword
railway env set TOGETHER_API_KEY=yourapikey
```

### Feature Flags

Feature flags can be set via environment variables:

```bash
FEATURE_NEW_ANALYSIS_ENGINE=false
ENABLE_EXPERIMENTAL_AGENTS=false
```

## Monitoring and Logging

### Prometheus and Grafana

```bash
# Install monitoring stack
helm install monitoring prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --create-namespace

# Access Grafana
kubectl port-forward svc/monitoring-grafana 3000:80 -n monitoring
```

### Logging with Loki

```bash
# Install Loki stack
helm install loki grafana/loki-stack \
  --namespace monitoring \
  --set promtail.enabled=true
```

### Application Logs

Logs are structured in JSON format and include:
- Timestamp
- Log level
- Service name
- Request ID
- Contextual data

## Scaling

### Horizontal Pod Autoscaling

```yaml
# hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: api
  namespace: graphrag
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: api
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

### Database Scaling

#### TimescaleDB

```yaml
# timescaledb-values.yaml
replicaCount: 2
persistence:
  size: 100Gi
  storageClass: gp2
resources:
  requests:
    cpu: 2
    memory: 4Gi
  limits:
    cpu: 4
    memory: 8Gi
```

#### Neo4j

```yaml
# neo4j-values.yaml
neo4j:
  name: neo4j
  password: yourpassword
  resources:
    requests:
      cpu: 2
      memory: 4Gi
    limits:
      cpu: 4
      memory: 8Gi
  volumes:
    data:
      mode: storage
      size: 100Gi
```

## Backup and Recovery

### Database Backups

#### TimescaleDB Backups

```bash
# Create backup
kubectl exec -it timescaledb-0 -n graphrag -- pg_dump -Fc -d $DATABASE_NAME > backup.dump

# Restore from backup
cat backup.dump | kubectl exec -i timescaledb-0 -n graphrag -- pg_restore -d $DATABASE_NAME
```

#### Neo4j Backups

```bash
# Create backup
kubectl exec -it neo4j-0 -n graphrag -- cypher-shell -u neo4j -p $PASSWORD "CALL apoc.export.cypher.all('backup.cypher')"

# Restore from backup
kubectl cp backup.cypher neo4j-0:/var/lib/neo4j/import/ -n graphrag
kubectl exec -it neo4j-0 -n graphrag -- cypher-shell -u neo4j -p $PASSWORD -f /var/lib/neo4j/import/backup.cypher
```

### Automated Backups with Velero

```bash
# Install Velero
velero install \
  --provider aws \
  --plugins velero/velero-plugin-for-aws:v1.0.0 \
  --bucket $BUCKET \
  --backup-location-config region=$REGION \
  --snapshot-location-config region=$REGION

# Create backup schedule
velero create schedule daily-backup \
  --schedule="@daily" \
  --include-namespaces graphrag \
  --ttl 720h0m0s
```

## Security Considerations

### Network Policies

```yaml
# network-policy.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: default-deny
  namespace: graphrag
spec:
  podSelector: {}
  policyTypes:
  - Ingress
  - Egress
```

### Pod Security Policies

```yaml
# psp.yaml
apiVersion: policy/v1beta1
kind: PodSecurityPolicy
metadata:
  name: restricted
spec:
  privileged: false
  allowPrivilegeEscalation: false
  requiredDropCapabilities:
    - ALL
  volumes:
    - 'configMap'
    - 'emptyDir'
    - 'secret'
    - 'persistentVolumeClaim'
  hostNetwork: false
  hostIPC: false
  hostPID: false
  runAsUser:
    rule: 'MustRunAsNonRoot'
  seLinux:
    rule: 'RunAsAny'
  supplementalGroups:
    rule: 'MustRunAs'
    ranges:
      - min: 1
        max: 65535
  fsGroup:
    rule: 'MustRunAs'
    ranges:
      - min: 1
        max: 65535
```

### TLS Termination

```yaml
# ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: graphrag
  namespace: graphrag
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/force-ssl-redirect: "true"
spec:
  tls:
  - hosts:
    - api.graphrag.example.com
    secretName: graphrag-tls
  rules:
  - host: api.graphrag.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: api
            port:
              number: 80
```

## Troubleshooting

### Common Issues

1. **API Not Starting**
   - Check database connection
   - Verify environment variables
   - Check logs: `kubectl logs -l app=api -n graphrag`

2. **High CPU/Memory Usage**
   - Check HPA status: `kubectl get hpa -n graphrag`
   - Analyze metrics in Grafana
   - Consider increasing resources

3. **Database Connection Issues**
   - Verify database pods are running
   - Check network policies
   - Test connection manually

### Debugging Commands

```bash
# Get pod status
kubectl get pods -n graphrag

# View logs
kubectl logs -l app=api -n graphrag

# Port forward to service
kubectl port-forward svc/api 8000:80 -n graphrag

# Exec into container
kubectl exec -it <pod-name> -n graphrag -- /bin/bash

# Describe pod
kubectl describe pod <pod-name> -n graphrag
```

### Getting Help

For additional support:
1. Check the [troubleshooting guide](TROUBLESHOOTING.md)
2. Open an issue on GitHub
3. Join our [Discord server](https://discord.gg/graphrag)

## License

This project is licensed under the MIT License - see the [LICENSE](../../LICENSE) file for details.
