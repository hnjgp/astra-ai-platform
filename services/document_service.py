from exceptions import DocumentNotFoundError

from repository.document import (
    get_document_by_id,
    get_documents as repository_get_documents,
    create_document as repository_create_document,
    update_document as repository_update_document,
    delete_document as repository_delete_document
)


def create_document(
    db,
    title: str,
    category: str,
    body: str | None = None,
    is_private: bool = False,
    owner_id: int | None = None,
):
    return repository_create_document(
        db,
        title,
        category, body, is_private, owner_id
    )


def get_document(
    db,
    document_id: int
):
    document = get_document_by_id(
        db,
        document_id
    )

    if document is None:
        raise DocumentNotFoundError()

    return document


def get_documents(db):
    return repository_get_documents(db)


def update_document(db, document, **changes):
    return repository_update_document(
        db, document, **changes
    )


def delete_document(
    db,
    document
):
    return repository_delete_document(
        db,
        document
    )
