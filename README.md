# Astra AI Platform

> Production-oriented AI engineering platform built with Python and FastAPI, evolving toward RAG, agentic workflows, MCP, evaluation, observability, and cloud deployment.

Astra is a portfolio-grade AI engineering project designed to demonstrate how modern LLM-powered systems are designed, implemented, secured, tested, evaluated, and deployed in a production-oriented environment.

The project is being developed incrementally, with each stage introducing a real engineering capability.

---

## 🎯 Project Vision

Astra aims to evolve from a modular FastAPI backend into an AI platform capable of:

- LLM-powered applications
- Retrieval-Augmented Generation (RAG)
- Semantic and hybrid search
- Vector retrieval
- Reranking
- Agentic workflows
- Tool calling
- Model Context Protocol (MCP)
- Authentication and authorization
- AI evaluation
- Observability and monitoring
- Security controls
- Docker-based deployment
- CI/CD
- Cloud deployment

The goal is not to build a simple chatbot.

The goal is to demonstrate the engineering required to build reliable AI systems.

---

# 🏗️ Architecture

Current architecture:

```text
                    Client
                      |
                      v
                 FastAPI API
                      |
              +-------+-------+
              |               |
              v               v
       Authentication     API Routers
              |               |
              |               v
              |        Application Services
              |               |
              |               v
              |          Repositories
              |               |
              |               v
              |            Database
              |
              v
          Authorization


Future AI Architecture:

                    Client
                      |
                      v
                 FastAPI API
                      |
              +-------+-------+
              |               |
              v               v
       Authentication     AI Gateway
                              |
                     +--------+--------+
                     |        |        |
                     v        v        v
                    RAG     Agents   Tools
                     |        |        |
                     v        v        v
                 Retrieval LangGraph MCP
                     |
                     v
              Vector / Hybrid Search
                     |
                     v
                 PostgreSQL
                 + pgvector
                     |
                     v
                    LLM