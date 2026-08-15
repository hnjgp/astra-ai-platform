from sqlalchemy import select

from models import Document


def get_document_by_id(
    db,
    document_id: int
):
    result = db.execute(
        select(Document).where(
            Document.id == document_id
        )
    )

    return result.scalar_one_or_none()


def get_documents(db):
    result = db.execute(
        select(Document)
    )

    return result.scalars().all()


def create_document(
    db,
    title: str,
    category: str
):
    document = Document(
        title=title,
        category=category
    )

    db.add(document)
    db.commit()
    db.refresh(document)

    return document


def update_document(
    db,
    document,
    title: str,
    category: str
):
    document.title = title
    document.category = category

    db.commit()
    db.refresh(document)

    return document


def delete_document(
    db,
    document
):
    db.delete(document)
    db.commit()