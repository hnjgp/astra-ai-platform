from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    status
)

from fastapi.security import OAuth2PasswordRequestForm

from database import get_db

from services.user_service import (
    register_user,
    login_user,
    create_admin,
    list_users,
    remove_user,
    set_user_role,
)
from time import monotonic

from schemas import RoleUpdate, UserCreate

from security import (
    decode_access_token
)

from repository.user import (
    get_user_by_id
)


router = APIRouter(
    prefix="/users",
    tags=["users"]
)

_failed_logins: dict[str, list[float]] = {}
LOGIN_WINDOW_SECONDS = 60
MAX_FAILED_LOGINS = 5


def get_current_user(
    token: str = Depends(decode_access_token),
    db=Depends(get_db)
):

    user = get_user_by_id(
        db,
        int(token["sub"])
    )

    if user is None:
        raise HTTPException(
            status_code=401,
            detail="invalid token"
        )

    return user



def admin_required(
    user=Depends(get_current_user)
):

    if user.role != "admin":
        raise HTTPException(
            status_code=403,
            detail="admin access required"
        )

    return user



@router.post("/register")
def register(
    request: UserCreate,
    db=Depends(get_db)
):

    return register_user(
        db,
        request.username,
        request.password
    )



@router.post("/login")
def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db=Depends(get_db)
):
    now = monotonic()
    attempts = [stamp for stamp in _failed_logins.get(form_data.username, []) if now - stamp < LOGIN_WINDOW_SECONDS]
    if len(attempts) >= MAX_FAILED_LOGINS:
        raise HTTPException(status_code=429, detail="too many failed login attempts")

    try:
        token = login_user(db, form_data.username, form_data.password)
    except Exception as exc:
        from exceptions import InvalidCredentialsError
        if isinstance(exc, InvalidCredentialsError):
            attempts.append(now)
            _failed_logins[form_data.username] = attempts
        raise
    _failed_logins.pop(form_data.username, None)

    return {
        "access_token": token,
        "token_type": "bearer"
    }



@router.get("/profile")
def profile(
    user=Depends(get_current_user)
):

    return {
        "id": user.id,
        "username": user.username,
        "role": user.role
    }



@router.post("/admin")
def create_new_admin(
    request: UserCreate,
    db=Depends(get_db),
    admin=Depends(admin_required)
):

    return create_admin(
        db,
        request.username,
        request.password
    )


@router.post("")
@router.post("/")
def create_new_user(
    request: UserCreate,
    db=Depends(get_db),
    admin=Depends(admin_required),
):
    return register_user(db, request.username, request.password)


@router.get("")
@router.get("/")
def get_all_users(admin=Depends(admin_required), db=Depends(get_db)):
    return [user_response(user) for user in list_users(db)]


@router.get("/{user_id}")
def get_user(user_id: int, admin=Depends(admin_required), db=Depends(get_db)):
    user = get_user_by_id(db, user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="user not found")
    return user_response(user)


@router.patch("/{user_id}/role")
def update_role(
    user_id: int,
    request: RoleUpdate,
    admin=Depends(admin_required),
    db=Depends(get_db),
):
    user = get_user_by_id(db, user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="user not found")
    return user_response(set_user_role(db, user, request.role))


@router.delete("/{user_id}")
def delete_user(user_id: int, admin=Depends(admin_required), db=Depends(get_db)):
    if admin.id == user_id:
        raise HTTPException(status_code=400, detail="admin cannot delete itself")
    user = get_user_by_id(db, user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="user not found")
    remove_user(db, user)
    return {"message": "user deleted"}


def user_response(user):
    return {"id": user.id, "username": user.username, "role": user.role}
