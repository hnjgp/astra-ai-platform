from fastapi import APIRouter, Depends

from fastapi.security import OAuth2PasswordRequestForm


from dependency import (
    get_db,
    get_current_user,
    require_admin
)

from schemas import (
    UserCreate,
    AdminCreate
)

from services.user_service import (
    register_user,
    login_user,
    create_admin
)

router = APIRouter(
    prefix="/users",
    tags=["users"]
)


@router.post("/register")
def register(
    request: UserCreate,
    db=Depends(get_db)
):
    user = register_user(
        db,
        request.username,
        request.password
    )

    return {
        "id": user.id,
        "username": user.username
    }


@router.post("/login")
def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db=Depends(get_db)
):
    access_token = login_user(
        db,
        form_data.username,
        form_data.password
    )

    return {
        "access_token": access_token,
        "token_type": "bearer"
    }


@router.get("/")
def get_users(
    db=Depends(get_db)
):
    return {
        "message": "users endpoint"
    }


@router.get("/profile")
def profile(
    user=Depends(get_current_user)
):
    return {
        "id": user.id,
        "username": user.username
    }

@router.post("/admin")
def create_admin_api(
    request: AdminCreate,
    db=Depends(get_db),
    current_user=Depends(require_admin)
):
    user = create_admin(
        db,
        request.username,
        request.password
    )

    return {
        "id": user.id,
        "username": user.username,
        "role": user.role
    }