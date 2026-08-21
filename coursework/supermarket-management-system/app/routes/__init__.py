from app.routes.analytics import register_routes as register_analytics_routes
from app.routes.announcements import register_routes as register_announcements_routes
from app.routes.assistant import register_routes as register_assistant_routes
from app.routes.auth import register_routes as register_auth_routes
from app.routes.cashier import register_routes as register_cashier_routes
from app.routes.finance import register_routes as register_finance_routes
from app.routes.inventory import register_routes as register_inventory_routes
from app.routes.product import register_routes as register_product_routes
from app.routes.sales import register_routes as register_sales_routes
from app.routes.second_phase import register_routes as register_second_phase_routes


def register_routes(app):
    register_auth_routes(app)
    register_product_routes(app)
    register_inventory_routes(app)
    register_sales_routes(app)
    register_finance_routes(app)
    register_analytics_routes(app)
    register_announcements_routes(app)
    register_assistant_routes(app)
    register_cashier_routes(app)
    register_second_phase_routes(app)
