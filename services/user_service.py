from exceptions import (
    UserAlreadyExistsError,
    InvalidCredentialsError
)


from security import (
    hash_password,
    verify_password,
    create_access_token
)

from repository.user import (
    get_user_by_username,
    create_user
)


def register_user(
    db,
    username: str,
    password: str
):
    existing_user = get_user_by_username(
        db,
        username
    )

    if existing_user is not None:
        raise UserAlreadyExistsError()

    hashed_password = hash_password(
        password
    )

    user = create_user(
        db,
        username,
        hashed_password
    )

    return user


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

    access_token = create_access_token({
        "sub": str(user.id)
    })

    return access_token



def create_admin(
    db,
    username: str,
    password: str
):
    existing_user = get_user_by_username(
        db,
        username
    )

    if existing_user is not None:
        raise UserAlreadyExistsError()

    hashed_password = hash_password(password)

    user = create_user(
        db,
        username,
        hashed_password
    )

    user.role = "admin"

    db.commit()
    db.refresh(user)

    return user