# Supermarket Management System

[English](README.md) | [简体中文](README.zh-CN.md)

<p align="center">
  <a href="reports/system-analysis-design/screenshots/系统运行界面完整截图_20260614_220302/03-首页概览.png"><img src="reports/system-analysis-design/screenshots/系统运行界面完整截图_20260614_220302/03-首页概览.png" alt="Dashboard overview page" width="49%" /></a>
  <a href="reports/system-analysis-design/screenshots/系统运行界面完整截图_20260614_220302/04-商品管理.png"><img src="reports/system-analysis-design/screenshots/系统运行界面完整截图_20260614_220302/04-商品管理.png" alt="Product management page" width="49%" /></a>
</p>

**A full store management system built with Flask and SQLite for a software development and management course design.**

The system covers products, inventory, checkout, sales, finance, announcements, business analysis and an intelligent assistant, plus second-phase modules for members, employees, suppliers and system parameters. The engineering side uses uv-managed dependencies, a pytest suite with a 100 percent coverage gate on the backend models and second-phase services, and a complete analysis package with reports, diagrams and the defense deck.

**Status.** Course design complete. 11 automated tests pass, and the coverage report shows 530 statements at 100 percent.

## My Role

Built the Flask application end to end: models, routes, services, templates and static assets (`app/`), the SQL schema and seed data (`data/`), and the pytest suite with the backend coverage gate (`tests/`). Produced the analysis and design report, report figures and defense deck (`reports/`), the exported and editable system diagram sets (`supermarket-management-diagrams/`, `supermarket-management-diagrams-drawio-editable/`), and the course deliverable notes (`docs/course-deliverables/`).

## Modules

- Products, CRUD, shelf status, CSV and Excel import and inventory initialization
- Inventory, summaries, stock movement ledger and low-stock alerts
- Checkout, product search, cart settlement, stock validation and sales order generation
- Sales, order list, filtering and order details
- Finance, income and expense ledger, daily reconciliation, payables and monthly snapshots
- Business analysis, sales overview, trends, best sellers and category share
- Announcements, publish and take down, target roles and read status
- Intelligent assistant for inventory, sales, product and help questions
- Members, profiles, tiers, points adjustment and account status
- Employees, profiles, positions and scheduling
- Suppliers, contacts and settlement cycles
- System parameters, inventory alerts and receipt text

## Quick Start

Python 3.12 or 3.13 with uv.

```powershell
uv sync
uv run python run.py
```

Open `http://127.0.0.1:5000`.

Default accounts.

- Administrator `admin`, password `admin123`
- Cashier `cashier01`, password `123456`

The first launch creates the database tables, default users and categories, and demo data for the second-phase modules at `data/supermarket.db`.

## Tests

```powershell
uv run python -m compileall -q app
uv run pytest
uv run coverage run -m pytest
uv run coverage report
```

The suite covers login and registration, products, inventory, checkout, sales, finance, announcements, second-phase master data, error paths and page access control.

## Documentation

- [Course deliverables](docs/course-deliverables/README.md), startup notes, requirements modeling, test case design and the defense checklist.
- [System analysis and design report](reports/system-analysis-design/), the final report, defense deck and report figures.
- [System diagrams](supermarket-management-diagrams/), analysis and design diagrams as PNG images, with editable sources in [supermarket-management-diagrams-drawio-editable](supermarket-management-diagrams-drawio-editable/).

## Repository Layout

```text
app/       Flask application with models, routes, services, templates and static assets
tests/     Automated test suite
data/      SQL schema, seed data and the SQLite database
docs/      Course deliverables
reports/   Lab report, defense deck and report figures
supermarket-management-diagrams/                  Exported PNG diagrams
supermarket-management-diagrams-drawio-editable/  Editable drawio sources
run.py            Application entry point
config.py         Application configuration
pyproject.toml    Dependencies, pytest and coverage configuration
```

The database, demo accounts and seed data serve local development and coursework demonstration.

*Bohan Yu, Software Development and Management course design.*
