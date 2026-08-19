from models import Document


def create_document(db, title, category, body=None, is_private=False, owner_id=None):

    document = Document(
        title=title,
        category=category,
        body=body,
        is_private=is_private,
        owner_id=owner_id,
    )

    db.add(document)
    db.commit()
    db.refresh(document)

    return document



def get_documents(db):

    return db.query(Document).all()






def get_document_by_id(
    db,
    document_id
):
    return (
        db.query(Document)
        .filter(
            Document.id == document_id
        )
        .first()
    )



def update_document(db, document, **changes):

    for field, value in changes.items():
        if value is not None:
            setattr(document, field, value)

    db.commit()
    db.refresh(document)

    return document


def delete_document(db, document):

    db.delete(document)
    db.commit()

    return document
