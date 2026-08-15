from sqlalchemy import select

from models import User


def get_user_by_username(
    db,
    username: str
):
    result = db.execute(
        select(User).where(
            User.username == username
        )
    )

    return result.scalar_one_or_none()


def get_user_by_id(
    db,
    user_id: int
):
    return db.get(
        User,
        user_id
    )


def get_users(db):
    result = db.execute(
        select(User)
    )

    return result.scalars().all()


def create_user(
    db,
    username: str,
    password: str
):
    user = User(
        username=username,
        password=password
    )

    db.add(user)
    db.commit()
    db.refresh(user)

    return user