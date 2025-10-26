# IRIS ASSISTANT

## Tổng quan

**IRIS Assistant** là nền tảng AI Assistant có khả năng mở rộng cho nhiều domain nghiệp vụ, được xây dựng trên nền tảng Clean Architecture và Domain-Driven Design.

### Thông tin dự án

- **Tên dự án**: IRIS Assistant
- **Phiên bản**: 1.0.0
- **Kiến trúc**: Clean Architecture + Domain-Driven Design
- **Tech Stack**: FastAPI, SQLAlchemy (Async), PostgreSQL, Redis, AI/RAG
- **Python Version**: 3.11+

### Mục tiêu

Xây dựng hệ thống Assistant thông minh có khả năng:
- Trả lời câu hỏi về chính sách, quy định nội bộ (HR domain)
- Xử lý tài liệu đa định dạng (PDF, DOCX, TXT, Markdown)
- Tìm kiếm ngữ nghĩa với RAG (Retrieval Augmented Generation)
- Mở rộng cho nhiều domain: HR, Document Management, Authentication, ...
- Scale horizontally khi chuyển sang microservices

### Tech Stack

**Backend Framework:**
- FastAPI 0.104.1 - Modern async web framework
- Uvicorn - ASGI server with hot reload
- Pydantic 2.5+ - Data validation và settings management

**Database & Cache:**
- PostgreSQL - Primary database
- asyncpg - Async PostgreSQL driver
- SQLAlchemy 2.0+ - Async ORM framework
- Alembic - Database migrations
- Redis 5.0+ - Caching và session management

**AI/ML Stack:**
- LiteLLM - Primary AI gateway (multi-provider support)
- OpenAI - Fallback provider
- Azure OpenAI, Anthropic, Google AI - Optional providers
- ChromaDB 0.4.18 - Vector database
- Custom RAG Pipeline - Retrieval Augmented Generation
- Structure-aware chunking - Intelligent document processing

**Development & Testing:**
- Pytest + pytest-asyncio - Testing framework
- Black + isort - Code formatting
- Structlog - Structured logging
- Dependency Injector 4.41.0 - DI container
- python-dotenv - Environment management

---

## Mục lục

- [I. Kiến trúc dự án](#i-kiến-trúc-dự-án)
  - [1. Kiến trúc vật lý](#1-kiến-trúc-vật-lý)
  - [2. Tầng Core](#2-tầng-core)
  - [3. Tầng Domains](#3-tầng-domains)
  - [4. Tầng Infrastructure](#4-tầng-infrastructure)
  - [5. Tầng Interface](#5-tầng-interface)
- [II. Chi tiết các domain](#ii-chi-tiết-các-domain)
  - [1. Domain Auth](#1-domain-auth)
  - [2. Domain HR](#2-domain-hr)
  - [3. Domain Document](#3-domain-document)
  - [4. RAG Pipeline](#4-rag-pipeline)
- [III. Getting Started](#iii-getting-started)
- [IV. Development Guide](#iv-development-guide)
- [V. Testing](#v-testing)
- [VI. Deployment](#vi-deployment)

---

# I. KIẾN TRÚC DỰ ÁN

## 1. Kiến trúc vật lý

Dự án được tổ chức theo mô hình 4 tầng với sự phân tách rõ ràng giữa các concerns:

```
project_root/
├── core/               # Shared technical components
│   ├── ai/            # AI pipeline và models
│   ├── config/        # Configuration management
│   ├── shared/        # Base classes, exceptions, logger
│   └── events/        # Event bus (future)
│
├── domains/           # Business domains
│   ├── auth/         # Authentication & Authorization
│   ├── hr/           # HR Management
│   └── document/     # Document Processing
│
├── infrastructure/    # Technical infrastructure
│   ├── database/     # Database engine, session
│   ├── cache/        # Redis client
│   └── di/           # Dependency Injection container
│
├── interface/         # API layer
│   ├── api/          # Routes, middlewares
│   └── server.py     # FastAPI application factory
│
├── tests/            # Test suites
├── migrations/       # Alembic migrations
├── docs/             # Documentation
└── main.py           # Application entry point
```

### Dependency Flow

```
┌────────────────────┐
│     Interface      │  ← HTTP/API Layer
│  (FastAPI routes)  │
└────────┬───────────┘
         │
┌────────▼───────────┐
│    Application     │  ← Use Cases & Business Logic
│ (UseCases, DTOs)   │
└────────┬───────────┘
         │
┌────────▼───────────┐
│      Domain        │  ← Pure Business Domain
│ (Entities, Repos)  │
└────────┬───────────┘
         │
┌────────▼───────────┐
│  Infrastructure    │  ← Technical Implementation
│ (DB, Cache, AI)    │
└────────────────────┘
```

**Quy tắc phụ thuộc:**
- Tất cả dependencies đều hướng vào trong (vào Domain)
- Domain không phụ thuộc vào bất kỳ layer nào khác
- Infrastructure implements interfaces định nghĩa trong Domain
- Interface layer chỉ biết về Application layer

## 2. Tầng Core

Tầng core chứa các components kỹ thuật được chia sẻ giữa các domain.

### 2.1 core/ai - AI Pipeline

```
core/ai/
├── models/
│   ├── llm_gateway.py           # LLM provider abstraction
│   └── embedding_service.py     # Text → Vector embeddings
│
├── vectorstore/
│   ├── base.py                  # VectorStore interface
│   ├── pgvector_adapter.py      # PostgreSQL pgvector
│   └── chroma_adapter.py        # ChromaDB adapter
│
├── steps/
│   ├── intent_detector.py       # Query intent detection
│   ├── retrieval.py             # Context retrieval
│   ├── generation.py            # Response generation
│   ├── query_preprocessor.py    # Query preprocessing
│   ├── embedding.py             # Embedding orchestration
│   └── chunking/                # Document chunking strategies
│       ├── strategies.py        # Chunking strategies
│       ├── markdown_chunker.py  # Structure-aware chunking
│       ├── factory.py           # Strategy factory
│       └── models.py            # Data models
│
├── reasoning/                    # Advanced reasoning (Phase 2)
│   ├── reasoning_engine.py
│   ├── reasoning_bank.py
│   └── followup_manager.py
│
├── context/
│   └── session_context.py       # Session management (Phase 2)
│
└── pipeline/
    └── rag_pipeline.py          # End-to-end RAG orchestrator
```

**RAG Pipeline Flow:**
```
Query → Intent Detection → Retrieval → Context Building → Generation → Response
```

**Embedding Service vs Embedding Step:**

- **embedding_service.py** (core/ai/models/):
  - Adapter layer cho LiteLLM/OpenAI
  - Chịu trách nhiệm gọi API và tạo embeddings
  - Hỗ trợ fallback và batch processing
  - Infrastructure concern

- **embedding.py** (core/ai/steps/):
  - Orchestration logic trong pipeline
  - Chuẩn hóa input, gọi EmbeddingService
  - Kết hợp metadata với embeddings
  - Application concern

### 2.2 core/config - Configuration

```
core/config/
└── settings.py              # Pydantic Settings với lazy loading
```

**Cấu hình được tổ chức theo nhóm:**
- DatabaseSettings - PostgreSQL connection
- RedisSettings - Cache configuration
- JWTSettings - Authentication tokens
- AzureADSettings - Azure AD integration
- AIModelSettings - AI model management
- LiteLLMSettings - Primary AI provider
- OpenAISettings - Fallback provider
- Azure/Anthropic/Google - Optional providers

**Pattern:** Singleton với lazy loading để tối ưu performance.

### 2.3 core/shared - Shared Components

```
core/shared/
├── base_entity.py          # BaseEntity, AggregateRoot, ValueObject
├── base_repository.py      # Repository interfaces
├── base_usecase.py         # UseCase patterns (CQRS)
├── exceptions.py           # Exception hierarchy
└── logger.py               # Structured logging
```

**Exception Hierarchy:**
```
BaseException
├── DomainException          # Business logic errors
├── ApplicationException     # Use case errors
├── InfrastructureException  # Technical errors
└── InterfaceException       # API errors
```

**Base Classes:**

- **BaseEntity**: Identity-based equality, audit fields
- **AggregateRoot**: Domain event management
- **ValueObject**: Value-based equality
- **BaseRepository**: Generic CRUD operations
- **BaseUseCase**: Command pattern với logging tự động

## 3. Tầng Domains

Mỗi domain là một bounded context độc lập, có thể tách thành microservice.

### Domain Structure

```
domains/<domain_name>/
├── domain/                 # Pure business logic
│   ├── entities/          # Domain entities
│   ├── repositories/      # Repository interfaces
│   └── services/          # Domain services
│
├── application/           # Use cases
│   ├── usecases/         # Business use cases
│   ├── dtos/             # Data transfer objects
│   └── services/         # Application services
│
├── infrastructure/        # Technical implementation
│   ├── repositories_impl/ # Repository implementations
│   ├── models/           # ORM models
│   └── adapters/         # External system adapters
│
└── interface/            # API layer
    ├── routers/          # FastAPI routers
    ├── controllers/      # Request handlers
    └── schemas/          # Pydantic schemas
```

### 3.1 Domain Layer

**Nguyên tắc:**
- Không phụ thuộc vào framework, database, hoặc external services
- Chứa pure business logic
- Định nghĩa interfaces, không implement technical details

**Components:**

- **Entities**: Business objects với identity (Employee, Policy, Document)
- **Repositories**: Data access interfaces
- **Services**: Business logic spanning multiple entities

### 3.2 Application Layer

**Nguyên tắc:**
- Orchestrate domain objects để thực hiện use cases
- Không chứa business logic (business logic ở Domain)
- Không chứa technical details (technical details ở Infrastructure)

**Components:**

- **Use Cases**: Specific business operations (CreateEmployeeUseCase, QueryHRPolicyUseCase)
- **DTOs**: Data transfer objects cho input/output
- **Application Services**: Context building, orchestration

**CQRS Pattern:**
- **Query Use Cases**: Read-only operations (QueryUseCase)
- **Command Use Cases**: State-changing operations (CommandUseCase)

### 3.3 Infrastructure Layer

**Nguyên tắc:**
- Implement interfaces từ Domain layer
- Chứa tất cả technical details
- Phụ thuộc vào Domain, không ngược lại

**Components:**

- **Repository Implementations**: SQLAlchemy implementations
- **ORM Models**: Database table mappings
- **Adapters**: External service integrations

### 3.4 Interface Layer

**Nguyên tắc:**
- HTTP/API concerns only
- Không chứa business logic
- Transform HTTP requests → DTOs → Use Cases

**Components:**

- **Routers**: FastAPI route definitions
- **Controllers**: Request/response handling
- **Schemas**: Pydantic request/response models

## 4. Tầng Infrastructure

Tầng infrastructure chung cho toàn hệ thống.

```
infrastructure/
├── database/
│   ├── engine.py          # SQLAlchemy async engine
│   └── base.py            # ORM base classes, mixins
│
├── cache/
│   └── redis_client.py    # Redis async client wrapper
│
└── di/
    └── container.py       # DI container tổng
```

**Database Engine:**
- Async SQLAlchemy engine
- Connection pooling
- Auto commit/rollback
- Startup/shutdown lifecycle management

**Redis Cache:**
- Async Redis client
- JSON serialization support
- TTL management
- Pattern-based deletion

**DI Container:**
- CoreContainer: config, db, cache
- InfrastructureContainer: shared services
- ApplicationContainer: domain containers
- Auto-wiring cho interface và domains

## 5. Tầng Interface

```
interface/
├── api/
│   ├── routes_register.py    # Central route registration
│   └── middlewares.py        # Request logging, auth, CORS
│
└── server.py                 # FastAPI application factory
```

**FastAPI Application:**
- Application factory pattern
- CORS middleware
- Exception handlers (custom + global)
- Startup/shutdown events
- Health check endpoints

**Middlewares:**
- Request logging với request ID
- Timing và performance tracking
- Authentication (future)
- Error handling

---

# II. CHI TIẾT CÁC DOMAIN

## 1. Domain Auth

**Status**: Structure ready, implementation in progress

```
domains/auth/
├── domain/
│   ├── entities/
│   │   └── user.py                    # User entity
│   ├── repositories/
│   │   └── user_repository.py         # User repository interface
│   └── services/
│       └── auth_service.py            # Auth business logic
│
├── application/
│   ├── usecases/
│   │   ├── register_user_usecase.py   # User registration
│   │   ├── login_user_usecase.py      # User login
│   │   └── refresh_token_usecase.py   # Token refresh
│   └── dtos/
│       ├── register_dto.py
│       └── login_dto.py
│
├── infrastructure/
│   ├── repositories_impl/
│   │   └── user_repository_impl.py    # SQLAlchemy implementation
│   ├── orm/
│   │   └── user_model.py              # User ORM model
│   └── security/
│       ├── jwt_handler.py             # JWT generation/validation
│       └── password_hasher.py         # Password hashing (bcrypt)
│
└── interface/
    ├── controllers/
    │   └── auth_controller.py
    └── routes/
        └── auth_routes.py             # /auth/login, /auth/register
```

**Features:**
- User registration với email/password
- JWT-based authentication
- Token refresh mechanism
- Azure AD integration (optional)
- Password hashing với bcrypt

## 2. Domain HR

**Status**: Fully implemented

```
domains/hr/
├── domain/
│   ├── entities/
│   │   ├── employee.py
│   │   ├── policy.py
│   │   ├── department.py
│   │   └── payroll.py
│   └── repositories/
│       ├── employee_repository.py
│       ├── policy_repository.py
│       ├── department_repository.py
│       └── payroll_repository.py
│
├── application/
│   ├── usecases/
│   │   ├── queries/
│   │   │   ├── hr_assistant_query.py     # Main RAG query handler
│   │   │   ├── query_hr_policy.py
│   │   │   ├── query_leave_policy.py
│   │   │   ├── get_employee_info.py
│   │   │   └── ...
│   │   ├── actions/
│   │   │   ├── create_leave_request.py
│   │   │   ├── approve_leave_request.py
│   │   │   └── update_employee_record.py
│   │   └── hr_assistant_orchestrator.py  # Phase 2: Agent
│   ├── dtos/
│   │   └── hr_assistant_dto.py
│   └── services/
│       └── hr_context_builder.py         # Context enrichment
│
├── infrastructure/
│   ├── repositories_impl/               # SQLAlchemy implementations
│   └── models/                          # ORM models
│
└── interface/
    ├── routers/
    │   └── hr_assistant_router.py
    └── controllers/
        └── hr_assistant_controller.py
```

**Key Use Case: HR Assistant Query**

```python
# Flow: User Query → Context Building → RAG Pipeline → Response

HRQueryRequest
  ↓
HRAssistantQueryUseCase
  ↓
HRContextBuilder (enrich context với employee/dept info)
  ↓
RAGPipeline.run(query, context, filters)
  ↓
  ├─ Intent Detection
  ├─ Retrieval (search HR documents)
  └─ Generation (LLM generates answer)
  ↓
HRQueryResponse (answer + sources + metadata)
```

**Features:**
- HR policy Q&A với RAG
- Employee information queries
- Leave policy queries
- Intent-based routing (Phase 2)
- Action execution (Phase 2)

## 3. Domain Document

**Status**: Fully implemented

```
domains/document/
├── domain/
│   ├── entities/
│   │   ├── document.py              # Document entity
│   │   └── document_chunk.py        # Document chunk
│   └── repositories/
│       └── document_repository.py   # Document data access
│
├── application/
│   ├── usecases/
│   │   ├── upload_document.py       # Upload & process
│   │   ├── process_document.py      # Reprocess document
│   │   └── index_document.py        # Index to vectorstore
│   ├── dtos/
│   │   └── document_dto.py
│   └── services/
│       └── document_pipeline_service.py  # Processing orchestration
│
├── infrastructure/
│   ├── parsers/
│   │   ├── base_parser.py           # Parser interface
│   │   ├── pdf_parser.py            # PDF parsing
│   │   ├── docx_parser.py           # DOCX parsing
│   │   ├── txt_parser.py            # Text parsing
│   │   ├── loader.py                # File loading
│   │   ├── extractor_unstructured.py # Unstructured API
│   │   ├── normalizer.py            # Text normalization
│   │   ├── markdownizer.py          # Markdown conversion
│   │   └── chunker.py               # Document chunking
│   ├── repositories_impl/
│   │   └── document_repository_impl.py
│   └── storage/
│       └── file_storage_adapter.py  # File storage (local/S3)
│
└── interface/
    └── routers/
        └── document_router.py       # /documents/upload
```

**Document Processing Pipeline:**

```
1. Upload
   ↓ file (PDF/DOCX/TXT)
2. Parse
   ↓ raw text + metadata
3. Normalize
   ↓ cleaned text
4. Markdownize (with LLM)
   ↓ structured markdown
5. Chunk
   ↓ logical chunks với headers
6. Embed
   ↓ vector embeddings
7. Index
   ↓ PostgreSQL + vectorstore
```

**Features:**
- Multi-format document support (PDF, DOCX, TXT, MD)
- Structure-aware chunking
- LLM-powered markdown conversion
- Vector embeddings với ChromaDB
- Metadata extraction và storage
- Semantic search ready

## 4. RAG Pipeline

**Location**: `core/ai/pipeline/rag_pipeline.py`

**Architecture:**

```python
class RAGPipeline:
    """
    End-to-end Retrieval Augmented Generation pipeline
    
    Components:
    - IntentDetector: Classify query intent
    - Retrieval: Find relevant documents
    - Generation: Generate response with LLM
    """
    
    async def run(
        query: str,
        conversation_history: List[Dict],
        filters: Dict
    ) -> RAGResult:
        # 1. Detect intent
        intent = await intent_detector.detect_intent(query)
        
        # 2. Retrieve context
        context = await retrieval.get_context_for_query(
            query, intent, top_k, max_tokens
        )
        
        # 3. Generate response
        response = await generation.generate_rag_response(
            query, context, intent, history
        )
        
        return RAGResult(answer, sources, metadata)
```

**Configuration:**

```python
RAGConfig(
    top_k=3,                    # Final documents to LLM
    search_limit=15,            # Initial search results
    min_score=0.0,              # Minimum similarity score
    max_context_tokens=3000,    # Max context length
    temperature=0.7,            # LLM temperature
    max_tokens=1000,            # Max response tokens
    detect_intent=True,         # Enable intent detection
    use_fallback=True,          # Enable provider fallback
    include_metadata=True       # Include metadata in response
)
```

**Features:**
- Intent-based retrieval optimization
- Multi-provider fallback (LiteLLM → OpenAI)
- Context token management
- Metadata tracking
- Batch processing support

---

# III. GETTING STARTED

## Prerequisites

- Python 3.11 hoặc cao hơn
- PostgreSQL 14+ (với pgvector extension)
- Redis 5.0+
- Git

## Installation

### 1. Clone repository

```bash
git clone <repository-url>
cd clean_architecture_skeleton
```

### 2. Setup môi trường Python

**Sử dụng môi trường Env3_11 (recommended):**

```bash
# Windows
Env3_11\Scripts\activate

# Linux/Mac
source Env3_11/bin/activate
```

**Hoặc tạo môi trường mới:**

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc
venv\Scripts\activate     # Windows
```

### 3. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### 4. Setup database

**Start PostgreSQL (Docker):**

```bash
docker-compose up -d postgres
```

**Hoặc connect đến PostgreSQL hiện có**, đảm bảo có database `kinhcong`:

```sql
CREATE DATABASE kinhcong;
```

### 5. Setup Redis

```bash
docker-compose up -d redis
```

### 6. Configuration

**Tạo file `.env` từ template:**

```bash
cp env.example .env
```

**Edit `.env` với cấu hình của bạn:**

```bash
# Application
APP_NAME=IRIS_ASSISTANT
ENVIRONMENT=dev
DEBUG=True
HOST=0.0.0.0
PORT=8001
RELOAD=True

# Database
DATABASE_URL=postgresql+asyncpg://db_user:db_pw@localhost:5433/db_name
ECHO_SQL=False

# Redis
REDIS_URL=redis://localhost:6379/0

# JWT
SECRET_KEY=your-secret-key-here-change-in-production
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# AI Models
PRIMARY_AI_PROVIDER=litellm
LITELLM_API_KEY=your-litellm-key
LITELLM_BASE_URL=https://your-site.com.vn
OPENAI_API_KEY=your-openai-key-for-fallback

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=console
```

### 7. Run database migrations

```bash
alembic upgrade head
```

### 8. Start application

**Development mode (với auto-reload):**

```bash
python main.py
```

**Hoặc sử dụng uvicorn trực tiếp:**

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8001
```

### 9. Verify installation

**Check health endpoint:**

```bash
curl http://localhost:8001/health
```

**Expected response:**

```json
{
  "status": "healthy",
  "app": "IRIS_ASSISTANT",
  "environment": "dev"
}
```

**Access API documentation:**

- Swagger UI: http://localhost:8001/docs
- ReDoc: http://localhost:8001/redoc

---

# IV. DEVELOPMENT GUIDE

## Project Structure

### Thêm Domain Mới

**Bước 1: Tạo domain structure**

```bash
mkdir -p domains/new_domain/{domain,application,infrastructure,interface}
mkdir -p domains/new_domain/domain/{entities,repositories,services}
mkdir -p domains/new_domain/application/{usecases,dtos,services}
mkdir -p domains/new_domain/infrastructure/{repositories_impl,models,adapters}
mkdir -p domains/new_domain/interface/{routers,controllers,schemas}
```

**Bước 2: Implement domain entities**

```python
# domains/new_domain/domain/entities/my_entity.py
from core.shared.base_entity import BaseEntity

class MyEntity(BaseEntity):
    def __init__(self, name: str, **kwargs):
        super().__init__(**kwargs)
        self._name = name
    
    @property
    def name(self) -> str:
        return self._name
```

**Bước 3: Define repository interface**

```python
# domains/new_domain/domain/repositories/my_repository.py
from abc import abstractmethod
from core.shared.base_repository import BaseRepository
from ..entities.my_entity import MyEntity

class MyRepository(BaseRepository[MyEntity, str]):
    @abstractmethod
    async def find_by_name(self, name: str) -> MyEntity:
        pass
```

**Bước 4: Create use case**

```python
# domains/new_domain/application/usecases/create_my_entity.py
from core.shared.base_usecase import BaseUseCase

class CreateMyEntityUseCase(BaseUseCase[CreateDTO, ResponseDTO]):
    def __init__(self, repository: MyRepository):
        self.repository = repository
    
    async def execute(self, dto: CreateDTO) -> ResponseDTO:
        entity = MyEntity(name=dto.name)
        saved = await self.repository.create(entity)
        return ResponseDTO.from_entity(saved)
```

**Bước 5: Implement infrastructure**

```python
# domains/new_domain/infrastructure/repositories_impl/my_repository_impl.py
from domains.new_domain.domain.repositories import MyRepository

class MyRepositoryImpl(MyRepository):
    async def find_by_name(self, name: str) -> MyEntity:
        # SQLAlchemy implementation
        pass
```

**Bước 6: Create API router**

```python
# domains/new_domain/interface/routers/my_router.py
from fastapi import APIRouter

router = APIRouter()

@router.post("/")
async def create_entity(request: CreateRequest):
    # Controller logic
    pass
```

**Bước 7: Register routes**

```python
# interface/api/routes_register.py
from domains.new_domain.interface.routers.my_router import router as new_domain_router

def register_routes(app: FastAPI):
    # ...
    app.include_router(
        new_domain_router,
        prefix="/api/v1/new-domain",
        tags=["New Domain"]
    )
```

## Coding Standards

### Code Style

- **PEP 8** compliance
- **Type hints** cho tất cả functions
- **Docstrings** cho public methods (Google style)
- **Black** formatter (line length: 100)
- **isort** cho import organization

### Naming Conventions

- **Classes**: PascalCase (UserRepository, CreateEmployeeUseCase)
- **Functions/Methods**: snake_case (get_by_id, create_employee)
- **Constants**: UPPER_SNAKE_CASE (MAX_RETRY_COUNT, DEFAULT_TIMEOUT)
- **Private members**: _leading_underscore (_internal_method)

### Import Order

```python
# 1. Standard library
import os
from typing import Optional

# 2. Third-party
from fastapi import APIRouter
from sqlalchemy import select

# 3. Local application
from core.shared.base_entity import BaseEntity
from domains.hr.domain.entities import Employee
```

## Database Migrations

### Tạo migration mới

```bash
alembic revision --autogenerate -m "description of changes"
```

### Apply migrations

```bash
# Upgrade to latest
alembic upgrade head

# Upgrade one version
alembic upgrade +1

# Downgrade one version
alembic downgrade -1
```

### Check migration status

```bash
alembic current
alembic history
```

## Logging

### Sử dụng logger

```python
from core.shared.logger import get_logger

logger = get_logger(__name__, component="my_component", domain="hr")

# Structured logging
logger.info(
    "Processing document",
    document_id=doc.id,
    file_size=len(content),
    user_id=user.id
)

logger.error(
    "Failed to process",
    error=str(e),
    exc_info=True
)
```

### Log levels

- **DEBUG**: Detailed information for debugging
- **INFO**: General information about system operation
- **WARNING**: Warning messages for non-critical issues
- **ERROR**: Error messages for failures
- **CRITICAL**: Critical issues requiring immediate attention

---

# V. TESTING

## Test Structure

```
tests/
├── conftest.py                    # Pytest configuration & fixtures
├── debug/                         # Debug scripts (not tests)
├── domains/
│   ├── hr/
│   │   ├── test_hr_query.py
│   │   └── test_hr_entities.py
│   └── document/
│       └── test_upload_document.py
├── integration/
│   ├── test_rag_pipeline.py
│   └── test_database.py
└── api/
    └── test_hr_endpoints.py
```

## Running Tests

### Run all tests

```bash
pytest
```

### Run with coverage

```bash
pytest --cov=core --cov=domains --cov=infrastructure --cov-report=html
```

### Run specific test file

```bash
pytest tests/domains/hr/test_hr_query.py
```

### Run with markers

```bash
# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration

# Skip slow tests
pytest -m "not slow"
```

## Writing Tests

### Unit Test Example

```python
# tests/domains/hr/application/test_hr_query.py
import pytest
from unittest.mock import Mock, AsyncMock

@pytest.mark.asyncio
async def test_hr_query_success():
    # Arrange
    mock_rag = Mock()
    mock_rag.run = AsyncMock(return_value=RAGResult(answer="Test"))
    
    usecase = HRAssistantQueryUseCase()
    usecase.rag_pipeline = mock_rag
    
    request = HRQueryRequest(query="Test query")
    
    # Act
    response = await usecase.execute(request)
    
    # Assert
    assert response.answer == "Test"
    mock_rag.run.assert_called_once()
```

### Integration Test Example

```python
# tests/integration/test_rag_pipeline.py
@pytest.mark.asyncio
@pytest.mark.integration
async def test_rag_pipeline_end_to_end():
    pipeline = get_rag_pipeline()
    
    result = await pipeline.run("Test query")
    
    assert result.answer is not None
    assert result.num_documents > 0
```

### Fixtures

```python
# tests/conftest.py
import pytest
from sqlalchemy.ext.asyncio import create_async_engine

@pytest.fixture
async def db_session():
    """Provide test database session"""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    # Setup
    yield session
    # Teardown
    await engine.dispose()

@pytest.fixture
def mock_llm():
    """Mock LLM client"""
    mock = Mock()
    mock.chat.completions.create = AsyncMock(
        return_value={"choices": [{"message": {"content": "Test"}}]}
    )
    return mock
```

## Test Coverage Goals

- Core: > 80%
- Domain: > 80%
- Application: > 75%
- Infrastructure: > 60%
- Interface: > 50%

---

# VI. DEPLOYMENT

## Docker Deployment

### Build Docker image

```bash
# Development
docker build -f Dockerfile.dev -t iris-assistant:dev .

# Production
docker build -f Dockerfile.prod -t iris-assistant:prod .
```

### Run with Docker Compose

```bash
# Development
docker-compose -f docker-compose.yml up -d

# Production
docker-compose -f docker-compose.prod.yml up -d

# With Docker Hub
docker-compose -f docker-compose.hub.yml up -d
```

### Environment-specific deployments

```bash
# Development
ENVIRONMENT=dev docker-compose up -d

# Staging
ENVIRONMENT=staging docker-compose up -d

# Production
ENVIRONMENT=prod docker-compose up -d
```

## Manual Deployment

### 1. Setup production environment

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure production settings

**Edit `.env`:**

```bash
ENVIRONMENT=prod
DEBUG=False
RELOAD=False
LOG_LEVEL=WARNING
```

### 3. Run migrations

```bash
alembic upgrade head
```

### 4. Start with Gunicorn (recommended)

```bash
gunicorn main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8001 \
  --timeout 120 \
  --access-logfile - \
  --error-logfile -
```

## Production Checklist

- [ ] Environment variables set correctly
- [ ] Database migrations applied
- [ ] Redis connection configured
- [ ] AI API keys configured
- [ ] Secrets not in version control
- [ ] HTTPS enabled (reverse proxy)
- [ ] CORS configured for production domains
- [ ] Logging level set appropriately
- [ ] Health checks configured
- [ ] Backup strategy in place
- [ ] Monitoring enabled

## Monitoring

### Health Check Endpoints

```bash
# Basic health
curl http://localhost:8001/health

# Detailed health (if implemented)
curl http://localhost:8001/health/detailed
```

### Logs

```bash
# Docker logs
docker-compose logs -f app

# Application logs
tail -f logs/app.log
```

---

## Triển khai theo giai đoạn

### Phase 1: Foundation (COMPLETED)

- Core architecture setup
- Base classes và patterns
- Database và cache infrastructure
- Logging và configuration
- HR Assistant với RAG
- Document processing pipeline

### Phase 2: Advanced Features (IN PROGRESS)

- Authentication implementation
- Authorization & RBAC
- Conversation memory management
- Advanced reasoning engine
- Agent-based orchestration
- Multi-turn dialogue support

### Phase 3: Production Ready

- Comprehensive testing (coverage > 70%)
- Performance optimization
- Monitoring và observability
- Security hardening
- Documentation completion
- Deployment automation

### Phase 4: Scaling

- Microservices migration
- Distributed tracing
- Advanced caching strategies
- Multi-tenancy support
- Load balancing
- High availability setup

---

## Tài liệu bổ sung

- [Feedback 26/10/2025](docs/FEEDBACK_26_10_25.md) - Code review chi tiết
- [Phase 1 Complete](PHASE1_COMPLETE.md) - Tổng kết Phase 1
- [Testing Guide](tests/README.md) - Hướng dẫn testing
- [Architecture Docs](docs/README.md) - Chi tiết kiến trúc

---

## Contributing

### Quy trình đóng góp

1. Fork repository
2. Tạo feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Tạo Pull Request

### Code Review Checklist

- [ ] Code follows coding standards
- [ ] Tests written và pass
- [ ] Documentation updated
- [ ] No linter warnings
- [ ] Changelog updated (if applicable)

---

## License

Proprietary - All rights reserved.

---

## Support & Contact

Để được hỗ trợ hoặc báo cáo vấn đề:
- Tạo issue trên GitHub
- Liên hệ team qua email
- Check documentation trong folder `docs/`

---

**Cập nhật lần cuối**: 26/10/2025
