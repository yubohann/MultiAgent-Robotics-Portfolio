# Supermarket Management System

A full-featured supermarket store management system built for my *Software Development and Management* course design at [REDACTED]. It covers products, inventory, checkout, sales, finance, announcements, business analysis, an intelligent assistant, and the second-phase modules: members, employees, suppliers, and system parameters.

## About this work

- **Author**: Bohan Yu (Bohan Yu)
- **Course**: Software Development and Management — course design (课程设计)
- **Stack**: Flask + Flask-SQLAlchemy + SQLite + Jinja2 + vanilla JavaScript
- **Engineering discipline**: uv-managed dependencies, automated pytest suite with a 100% coverage gate, and full system-analysis documentation (reports, diagrams, defense PPT) in this repository.

## Tech stack

| Layer | Technology |
| --- | --- |
| Backend | Flask + Flask-SQLAlchemy |
| Database | SQLite (`data/supermarket.db`) |
| Frontend | Jinja2 templates + vanilla JavaScript + Bootstrap-style pages |
| Data exchange | CSV / Excel import (openpyxl) |
| Tooling | uv package management, pytest, coverage (100% gate) |

## Repository structure

```text
app/                     Flask application (models / routes / services / templates / static)
tests/                   Automated tests (auth, products, inventory, checkout, finance, announcements, second-phase modules, error paths)
data/                    Database file and SQL scripts
scripts/                 Helper scripts
docs/                    Course-design documents and defense materials
reports/                 Lab reports and defense PPT
supermarket-management-diagrams/       System analysis & design diagrams (PNG)
supermarket-management-diagrams-drawio-editable/  drawio editable sources
tools/                   Utility scripts
run.py                   Application entry point
pyproject.toml           Project configuration (deps / pytest / coverage)
```

## Quick start

Requirements:

- Python `>=3.12,<3.14`
- Recommended package manager: `uv`
- Database: SQLite

```powershell
cd supermarket-management-system
uv sync
uv run python run.py
```

Then open `http://127.0.0.1:5000`.

Default accounts:

- Administrator: `admin` / `admin123`
- Cashier: `cashier01` / `123456`

The first launch creates the database tables, default users and categories, and demo data for the second-phase modules. The database lives at `data/supermarket.db`.

## Tests

```powershell
uv run python -m compileall -q app
uv run pytest
uv run coverage run -m pytest
uv run coverage report
```

The automated tests in `tests/` cover login/registration, products, inventory, checkout, sales, finance, announcements, second-phase master data, error paths, and page access control. Coverage is measured on the backend models and second-phase core services, with `fail_under = 100` in the coverage config — currently at 100%.

## Main modules

- Products: CRUD, on/off shelf, CSV/Excel import, inventory initialization
- Inventory: summary, lists, stock movement ledger, low-stock alerts
- Checkout: product search, cart settlement, stock validation, sales order generation
- Sales: order list, filtering, order details
- Finance: income/expense ledger, daily reconciliation, payables, monthly snapshots
- Business analysis: sales overview, trends, best sellers, category share
- Announcements: publish, on/off line, target roles, read status
- Intelligent assistant: inventory / sales / product / help Q&A
- Members: profiles, tiers, points adjustment, enable/disable
- Employees: profiles, positions, scheduling, enable/disable
- Suppliers: profiles, contacts, settlement cycles, enable/disable
- System: store parameters, inventory-alert switch, receipt text, etc.

## Course deliverables

- Final lab report: `reports/system-analysis-design/`
- Defense PPT: `reports/system-analysis-design/超市管理系统_答辩PPT_20260602.pptx`
- Defense flow checklist: `docs/course-deliverables/defense-flow-checklist.md`
- Report figures: `reports/system-analysis-design/images/`
- System analysis & design PNG diagrams: `supermarket-management-diagrams/`
- drawio editable diagrams: `supermarket-management-diagrams-drawio-editable/`
- Module-level UML use-case supplements: `supermarket-management-diagrams/01-环境与用例/模块级用例图/`
- Phase supplements: `docs/course-deliverables/`
- Automated & functional test materials: `tests/`, `docs/course-deliverables/test-case-design.md`

## Acceptance notes

The repository is aligned with the course-design defense flow: reports, PPT, diagrams, test source, code-review records, and phase supplements are all committed. Before acceptance, re-run `uv run pytest`, `uv run coverage report`, and `git status --short` so the test results and coverage match the defense materials.

*Bohan Yu — Software Development and Management course design.*