# Lifting API - Руководство по развёртыванию

## Архитектура

```
┌─────────────────────────────────────────────────────────────┐
│                      КЛИЕНТЫ                                │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐        │
│  │ Python  │  │  Web    │  │ Mobile  │  │  Другие │        │
│  │  SDK    │  │  App    │  │  App    │  │   API   │        │
│  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘        │
└───────┼────────────┼────────────┼────────────┼──────────────┘
        │            │            │            │
        ▼            ▼            ▼            ▼
┌─────────────────────────────────────────────────────────────┐
│                    LOAD BALANCER                            │
│              (nginx / Cloudflare / AWS ALB)                 │
└─────────────────────────┬───────────────────────────────────┘
                          │
        ┌─────────────────┼─────────────────┐
        ▼                 ▼                 ▼
┌───────────────┐ ┌───────────────┐ ┌───────────────┐
│   API Node 1  │ │   API Node 2  │ │   API Node N  │
│   (FastAPI)   │ │   (FastAPI)   │ │   (FastAPI)   │
│   Port 8000   │ │   Port 8000   │ │   Port 8000   │
└───────┬───────┘ └───────┬───────┘ └───────┬───────┘
        │                 │                 │
        └─────────────────┼─────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                     REDIS (опционально)                     │
│           Rate limiting / Sessions / Cache                  │
└─────────────────────────────────────────────────────────────┘
```

---

## 1. Быстрый старт

### Локальный запуск

```bash
cd server
pip install -r requirements.txt

# Запуск (разработка)
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Запуск (production)
gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000
```

### Docker

```bash
cd server
docker build -t lifting-api .
docker run -d -p 8000:8000 --name lifting lifting-api
```

### Docker Compose (с Redis для rate limiting)

```yaml
# docker-compose.yml
version: '3.8'

services:
  api:
    build: ./server
    ports:
      - "8000:8000"
    environment:
      - REDIS_URL=redis://redis:6379
      - API_KEYS=prod-key-1,prod-key-2,enterprise-key
      - DEFAULT_RATE_LIMIT=1000
    depends_on:
      - redis
    deploy:
      replicas: 3  # 3 инстанса для масштабирования

  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - api

volumes:
  redis_data:
```

---

## 2. Конфигурация

### Переменные окружения

```bash
# server/.env (создать файл)

# Сервер
HOST=0.0.0.0
PORT=8000
DEBUG=false

# API ключи (через запятую)
API_KEYS=demo-key,prod-key-abc123,enterprise-xyz789

# Rate Limiting (запросов в минуту)
DEFAULT_RATE_LIMIT=100        # Обычные ключи
PREMIUM_RATE_LIMIT=1000       # Премиум ключи
ENTERPRISE_RATE_LIMIT=10000   # Enterprise

# Премиум/Enterprise ключи
PREMIUM_KEYS=prod-key-abc123
ENTERPRISE_KEYS=enterprise-xyz789

# Redis (для распределённого rate limiting)
REDIS_URL=redis://localhost:6379

# CORS (разрешённые домены)
CORS_ORIGINS=https://app.example.com,https://admin.example.com
```

### Изменение IP/порта

```bash
# Локально
uvicorn app.main:app --host 192.168.1.100 --port 9000

# Docker
docker run -p 9000:8000 lifting-api

# В коде клиента
client = LiftingClient(
    api_key="your-key",
    base_url="http://192.168.1.100:9000"
)
```

---

## 3. API Ключи

### Создание ключей

```python
import secrets

# Генерация безопасного ключа
api_key = f"lft_{secrets.token_urlsafe(32)}"
print(api_key)  # lft_Abc123...xyz (43 символа)
```

### Типы ключей

| Тип | Лимит | Цена | Для кого |
|-----|-------|------|----------|
| `demo-*` | 100/мин | Бесплатно | Тестирование |
| `prod-*` | 1000/мин | $29/мес | Разработчики |
| `enterprise-*` | 10000/мин | Custom | Компании |

### Добавление ключей

```bash
# Через переменные окружения
export API_KEYS="demo-key,lft_newkey123,lft_anotherkey"

# Или в .env файле
API_KEYS=demo-key,lft_newkey123,lft_anotherkey
```

### Интеграция с базой данных (будущее)

```python
# server/app/auth.py - расширение для БД

from sqlalchemy import select
from .database import AsyncSession, APIKey

async def get_api_key_from_db(key: str, db: AsyncSession) -> Optional[APIKeyInfo]:
    """Получить ключ из БД."""
    result = await db.execute(
        select(APIKey).where(APIKey.key == key, APIKey.is_active == True)
    )
    api_key = result.scalar_one_or_none()

    if api_key:
        return APIKeyInfo(
            key=api_key.key,
            tier=api_key.tier,
            rate_limit=api_key.rate_limit,
            user_id=api_key.user_id,
        )
    return None
```

---

## 4. Rate Limiting

### Как это работает

```
Sliding Window Algorithm:
┌──────────────────────────────────────────────────────────────┐
│  Окно: 60 секунд                                             │
│  ┌────┬────┬────┬────┬────┬────┬────┬────┬────┬────┐        │
│  │ 10 │ 15 │ 8  │ 12 │ 20 │ 5  │ 10 │ 8  │ 7  │ 5  │ = 100  │
│  └────┴────┴────┴────┴────┴────┴────┴────┴────┴────┘        │
│   -60s -54s -48s -42s -36s -30s -24s -18s -12s -6s   now    │
│                                                              │
│  Если сумма >= лимит → 429 Too Many Requests                │
│  Retry-After: секунды до освобождения слота                 │
└──────────────────────────────────────────────────────────────┘
```

### Ответ при превышении лимита

```http
HTTP/1.1 429 Too Many Requests
Retry-After: 45
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 0
X-RateLimit-Reset: 1703001245

{
  "detail": "Rate limit exceeded. Try again in 45 seconds."
}
```

### Клиент автоматически обрабатывает

```python
# client/client.py уже делает это:

if response.status_code == 429:
    retry_after = float(response.headers.get("Retry-After", 60))
    time.sleep(min(retry_after, 60))
    # автоматический retry
```

---

## 5. Примеры использования API

### Python SDK (рекомендуется)

```python
from client import LiftingClient
from tasks import AnomalyDetector, use_remote
import numpy as np

# Способ 1: Прямой клиент
client = LiftingClient(
    api_key="lft_your_key_here",
    base_url="https://api.lifting.io"
)

# Нормализация данных
data = np.random.randn(100, 10)
shapes = client.normalize(data)

# Расчёт расстояний
distances = client.distance(shapes, target=shapes[0])

# Центроид
center = client.centroid(shapes)

# Sync Lifting
signal = np.sin(np.linspace(0, 10, 256))
metrics = client.sync_compute(signal)
print(f"Time center: {metrics['time_center']}")


# Способ 2: Через backend (прозрачная интеграция)
use_remote(api_key="lft_your_key", base_url="https://api.lifting.io")

detector = AnomalyDetector(threshold_sigma=2.0)
detector.fit(shapes)  # Автоматически через API!
results = detector.predict(new_shapes)
```

### Batch операции (эффективно!)

```python
# Одним запросом вместо трёх
results = client.batch([
    {"op": "normalize", "data": data.tolist()},
    {"op": "centroid", "shapes": "@0.shapes"},      # Ссылка на результат #0
    {"op": "distance", "shapes": "@0.shapes", "target": "@1.centroid"},
])

shapes = np.array(results[0]["shapes"])
centroid = np.array(results[1]["centroid"])
distances = np.array(results[2]["distances"])
```

### Async Python

```python
import asyncio
from client.client import AsyncLiftingClient

async def process_data():
    async with AsyncLiftingClient(api_key="your-key") as client:
        # Параллельная обработка
        tasks = [
            client.normalize(data1),
            client.normalize(data2),
            client.normalize(data3),
        ]
        results = await asyncio.gather(*tasks)
    return results

shapes = asyncio.run(process_data())
```

### cURL (для тестирования)

```bash
# Health check
curl https://api.lifting.io/health

# Нормализация
curl -X POST https://api.lifting.io/geometry/normalize \
  -H "X-API-Key: your-key" \
  -H "Content-Type: application/json" \
  -d '{"data": [[1, 2, 3], [4, 5, 6]]}'

# Sync Lifting
curl -X POST https://api.lifting.io/sync/compute \
  -H "X-API-Key: your-key" \
  -H "Content-Type: application/json" \
  -d '{"signal": [0.1, 0.5, 0.9, 0.5, 0.1], "transform": "fft"}'
```

### JavaScript/TypeScript

```typescript
class LiftingClient {
  constructor(
    private apiKey: string,
    private baseUrl: string = 'https://api.lifting.io'
  ) {}

  async normalize(data: number[][]): Promise<number[][]> {
    const response = await fetch(`${this.baseUrl}/geometry/normalize`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-API-Key': this.apiKey,
      },
      body: JSON.stringify({ data }),
    });

    if (response.status === 429) {
      const retryAfter = response.headers.get('Retry-After') || '60';
      await new Promise(r => setTimeout(r, parseInt(retryAfter) * 1000));
      return this.normalize(data);  // retry
    }

    const result = await response.json();
    return result.shapes;
  }
}

// Использование
const client = new LiftingClient('your-key');
const shapes = await client.normalize([[1, 2, 3], [4, 5, 6]]);
```

---

## 6. Масштабирование

### Горизонтальное масштабирование

```yaml
# docker-compose.prod.yml
services:
  api:
    image: lifting-api:latest
    deploy:
      replicas: 10              # 10 инстансов
      resources:
        limits:
          cpus: '2'
          memory: 4G
      restart_policy:
        condition: on-failure
    environment:
      - REDIS_URL=redis://redis-cluster:6379
```

### Kubernetes

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: lifting-api
spec:
  replicas: 10
  selector:
    matchLabels:
      app: lifting-api
  template:
    spec:
      containers:
      - name: api
        image: lifting-api:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            cpu: "500m"
            memory: "512Mi"
          limits:
            cpu: "2000m"
            memory: "4Gi"
        env:
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: redis-secret
              key: url
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: lifting-api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: lifting-api
  minReplicas: 3
  maxReplicas: 50
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

### Почему это работает?

```
✓ Stateless API
  └── Каждый запрос независим
  └── Нет сессий на сервере
  └── Любой node может обработать любой запрос

✓ Async FastAPI
  └── Non-blocking I/O
  └── Тысячи одновременных соединений
  └── uvloop для максимальной производительности

✓ Redis для rate limiting
  └── Распределённый счётчик
  └── Все nodes видят одни лимиты
  └── Atomic операции

✓ Каждый клиент получает свой ответ
  └── API Key идентифицирует клиента
  └── Request ID для трейсинга
  └── Изолированные вычисления
```

---

## 7. Интеграция с веб-интерфейсом и платежами

### Архитектура с платежами

```
┌─────────────────────────────────────────────────────────────────┐
│                         WEB APP                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │   Next.js   │  │   Stripe    │  │   Auth0/    │             │
│  │  Frontend   │  │  Checkout   │  │   Clerk     │             │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘             │
└─────────┼────────────────┼────────────────┼─────────────────────┘
          │                │                │
          ▼                ▼                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      BACKEND API                                │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    FastAPI                               │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐ │   │
│  │  │  /auth   │  │ /billing │  │  /keys   │  │ /lifting │ │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘ │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────┬───────────────────────────────────────┘
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
    ┌──────────┐   ┌──────────┐   ┌──────────────┐
    │ Postgres │   │  Stripe  │   │ Lifting API  │
    │  (users, │   │   API    │   │   (math)     │
    │   keys)  │   │          │   │              │
    └──────────┘   └──────────┘   └──────────────┘
```

### Модели для платежей (подготовка)

```python
# server/app/models/billing.py (для будущего)

from sqlalchemy import Column, String, Integer, DateTime, ForeignKey, Enum
from datetime import datetime
import enum

class SubscriptionTier(enum.Enum):
    FREE = "free"
    PRO = "pro"
    ENTERPRISE = "enterprise"

class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True)
    email = Column(String, unique=True, index=True)
    stripe_customer_id = Column(String, unique=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class Subscription(Base):
    __tablename__ = "subscriptions"

    id = Column(String, primary_key=True)
    user_id = Column(String, ForeignKey("users.id"))
    tier = Column(Enum(SubscriptionTier), default=SubscriptionTier.FREE)
    stripe_subscription_id = Column(String)
    current_period_end = Column(DateTime)

class APIKey(Base):
    __tablename__ = "api_keys"

    id = Column(String, primary_key=True)
    user_id = Column(String, ForeignKey("users.id"))
    key = Column(String, unique=True, index=True)  # lft_xxx
    name = Column(String)  # "Production", "Development"
    tier = Column(Enum(SubscriptionTier))
    rate_limit = Column(Integer)  # requests per minute
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_used = Column(DateTime)

class UsageLog(Base):
    __tablename__ = "usage_logs"

    id = Column(Integer, primary_key=True)
    api_key_id = Column(String, ForeignKey("api_keys.id"))
    endpoint = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)
    response_time_ms = Column(Integer)
    # Для billing: считаем requests per month
```

### Stripe Webhook (будущее)

```python
# server/app/routes/webhooks.py

from fastapi import APIRouter, Request, HTTPException
import stripe

router = APIRouter()

@router.post("/stripe")
async def stripe_webhook(request: Request):
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature")

    event = stripe.Webhook.construct_event(
        payload, sig_header, settings.STRIPE_WEBHOOK_SECRET
    )

    if event["type"] == "checkout.session.completed":
        session = event["data"]["object"]
        user_id = session["client_reference_id"]

        # Создать/обновить подписку
        await create_subscription(user_id, tier="pro")

        # Создать API ключ
        api_key = await create_api_key(user_id, tier="pro", rate_limit=1000)

        # Отправить email с ключом
        await send_welcome_email(user_id, api_key)

    elif event["type"] == "customer.subscription.deleted":
        # Downgrade to free
        subscription = event["data"]["object"]
        await downgrade_user(subscription["customer"])

    return {"status": "ok"}
```

### Dashboard API (будущее)

```python
# server/app/routes/dashboard.py

@router.get("/usage")
async def get_usage(api_key: APIKeyInfo = Depends(get_api_key)):
    """Статистика использования для dashboard."""
    return {
        "current_month": {
            "requests": 15234,
            "limit": 100000,
            "percentage": 15.2,
        },
        "by_endpoint": {
            "/geometry/normalize": 8000,
            "/geometry/distance": 5000,
            "/sync/compute": 2234,
        },
        "daily": [
            {"date": "2024-01-15", "requests": 512},
            {"date": "2024-01-16", "requests": 489},
            # ...
        ]
    }

@router.get("/keys")
async def list_keys(user: User = Depends(get_current_user)):
    """Список API ключей пользователя."""
    return await get_user_keys(user.id)

@router.post("/keys")
async def create_key(name: str, user: User = Depends(get_current_user)):
    """Создать новый API ключ."""
    return await create_api_key(user.id, name=name)

@router.delete("/keys/{key_id}")
async def revoke_key(key_id: str, user: User = Depends(get_current_user)):
    """Отозвать API ключ."""
    await revoke_api_key(key_id, user.id)
    return {"status": "revoked"}
```

---

## 8. Безопасность

### Checklist

- [ ] HTTPS везде (Let's Encrypt / Cloudflare)
- [ ] Rate limiting включён
- [ ] API ключи достаточно длинные (32+ символов)
- [ ] Нет чувствительных данных в логах
- [ ] CORS настроен правильно
- [ ] Валидация входных данных (Pydantic)
- [ ] SQL injection защита (SQLAlchemy ORM)
- [ ] Secrets в переменных окружения, не в коде

### Nginx конфиг

```nginx
# nginx.conf
server {
    listen 443 ssl http2;
    server_name api.lifting.io;

    ssl_certificate /etc/letsencrypt/live/api.lifting.io/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/api.lifting.io/privkey.pem;

    # Security headers
    add_header X-Content-Type-Options nosniff;
    add_header X-Frame-Options DENY;
    add_header X-XSS-Protection "1; mode=block";

    location / {
        proxy_pass http://api:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }
}
```

---

## 9. Мониторинг

### Prometheus метрики

```python
# server/app/metrics.py
from prometheus_client import Counter, Histogram, generate_latest

REQUEST_COUNT = Counter(
    'lifting_requests_total',
    'Total requests',
    ['endpoint', 'status', 'api_key_tier']
)

REQUEST_LATENCY = Histogram(
    'lifting_request_latency_seconds',
    'Request latency',
    ['endpoint']
)

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type="text/plain")
```

### Логирование

```python
# Структурированные логи (JSON)
import structlog

logger = structlog.get_logger()

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)

    logger.info(
        "request",
        path=request.url.path,
        method=request.method,
        status=response.status_code,
        duration_ms=int((time.time() - start) * 1000),
        api_key=request.headers.get("X-API-Key", "")[:8] + "...",
    )

    return response
```

---

## FAQ

**Q: Можно ли использовать без Redis?**
A: Да, по умолчанию rate limiting in-memory. Redis нужен только для распределённого развёртывания.

**Q: Как добавить новый endpoint?**
A: Создать router в `server/app/routes/`, добавить в `main.py`.

**Q: Как тестировать локально?**
A: `uvicorn app.main:app --reload`, клиент на `localhost:8000`.

**Q: Какая максимальная нагрузка?**
A: ~5000 RPS на одном инстансе (зависит от операции). Масштабируется горизонтально.

**Q: Как мигрировать на production?**
A: Docker → Kubernetes, добавить HTTPS, настроить мониторинг.
