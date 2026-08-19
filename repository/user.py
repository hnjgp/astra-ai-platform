from sqlalchemy.orm import Session
from models import User




def get_user_by_username(
    db: Session,
    username: str
):

    return (
        db.query(User)
        .filter(
            User.username == username
        )
        .first()
    )



def get_user_by_id(
    db: Session,
    user_id: int
):

    return (
        db.query(User)
        .filter(
            User.id == user_id
        )
        .first()
    )






def create_user(
    db,
    username,
    password,
    role="user"
):

    user = User(
        username=username,
        password=password,
        role=role
    )

    db.add(user)
    db.commit()
    db.refresh(user)

    return user


def get_users(db: Session):
    return db.query(User).order_by(User.id).all()


def delete_user(db: Session, user: User):
    db.delete(user)
    db.commit()
