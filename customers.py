from fastapi import APIRouter
from services.data_service import get_customers_list

router = APIRouter(tags=["Customers"])


@router.get("/customers")
def customers():
    return get_customers_list()