from fastapi import HTTPException, status

from repository.user import (
    get_user_by_username,
    create_user,
    get_user_by_id,
    get_users,
    delete_user as repository_delete_user,
)

from security import (
    hash_password,
    verify_password,
    create_access_token
)

from models import User
from exceptions import InvalidCredentialsError


def register_user(
    db,
    username: str,
    password: str
):
    existing_user = get_user_by_username(
        db,
        username
    )

    if existing_user:
        from exceptions import UserAlreadyExistsError
        raise UserAlreadyExistsError()

    user = User(
        username=username,
        password=hash_password(password),
        role="user"
    )

    db.add(user)
    db.commit()
    db.refresh(user)

    return {
        "id": user.id,
        "username": user.username,
        "role": user.role
    }


def authenticate_user(
    db,
    username: str,
    password: str
):

    user = get_user_by_username(
        db,
        username
    )

    if user is None:
        raise InvalidCredentialsError()


    if not verify_password(
        password,
        user.password
    ):
        raise InvalidCredentialsError()


    return user



def login_user(
    db,
    username: str,
    password: str
):

    user = authenticate_user(
        db,
        username,
        password
    )


    token = create_access_token(
        {
            "sub": str(user.id),
            "role": user.role
        }
    )

    return token



def create_admin_user(
    db,
    username: str,
    password: str
):

    existing_user = get_user_by_username(
        db,
        username
    )

    if existing_user:
        from exceptions import UserAlreadyExistsError
        raise UserAlreadyExistsError()


    user = User(
        username=username,
        password=hash_password(password),
        role="admin"
    )


    db.add(user)
    db.commit()
    db.refresh(user)

    return user


# Public service API used by the user router and administrative workflows.
def create_admin(
    db,
    username: str,
    password: str
):
    return create_admin_user(db, username, password)


def list_users(db):
    return get_users(db)


def set_user_role(db, user: User, role: str):
    user.role = role
    db.commit()
    db.refresh(user)
    return user


def remove_user(db, user: User):
    repository_delete_user(db, user)
