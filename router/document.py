from fastapi import APIRouter, Depends, HTTPException

from database import get_db
from dependency import check_permission, get_current_user, get_optional_current_user

from schemas import (
    DocumentCreate,
    DocumentUpdate,
    DocumentResponse
)

from services.document_service import (
    create_document,
    get_documents,
    get_document,
    update_document,
    delete_document,
)


router = APIRouter(
    prefix="/documents",
    tags=["documents"]
)


@router.post(
    "/",
    response_model=DocumentResponse
)
def create_document_api(
    request: DocumentCreate,
    db=Depends(get_db),
    user=Depends(get_optional_current_user),
):
    return create_document(
        db,
        request.title,
        request.category,
        request.body,
        request.is_private,
        user.id if user else None,
    )


@router.get(
    "/",
    response_model=list[DocumentResponse]
)
def get_documents_api(
    db=Depends(get_db),
    user=Depends(get_optional_current_user),
):
    documents = get_documents(db)
    if user is not None and user.role == "admin":
        return documents
    return [
        document for document in documents
        if not document.is_private or (user is not None and document.owner_id == user.id)
    ]


@router.get(
    "/{document_id}",
    response_model=DocumentResponse
)
def get_document_api(
    document_id: int,
    db=Depends(get_db),
    user=Depends(get_optional_current_user),
):
    document = get_document(db, document_id)
    ensure_document_access(document, user)
    return document


@router.put(
    "/{document_id}",
    response_model=DocumentResponse
)
def update_document_api(
    document_id: int,
    request: DocumentUpdate,
    db=Depends(get_db),
    user=Depends(get_optional_current_user),
):
    document = get_document(db, document_id)
    ensure_document_owner(document, user)
    return update_document(db, document, **request.model_dump(exclude_unset=True))


@router.delete(
    "/{document_id}"
)
def delete_document_api(
    document_id: int,
    db=Depends(get_db),
    user=Depends(get_current_user)
):
    document = get_document(
        db,
        document_id
    )

    ensure_document_owner(document, user)
    delete_document(db, document)

    return {
        "message": "document deleted"
    }


@router.get(
    "/secure/{document_id}",
    response_model=DocumentResponse
)
def secure_document(
    document_id: int,
    permission=Depends(check_permission),
    db=Depends(get_db)
):
    return get_document(
        db,
        document_id
    )


def ensure_document_access(document, user):
    if document.is_private and user is None:
        raise HTTPException(status_code=401, detail="Not authenticated")
    if document.is_private and user.role != "admin" and user.id != document.owner_id:
        raise HTTPException(status_code=403, detail="document permission denied")


def ensure_document_owner(document, user):
    # Legacy public documents have no owner and remain editable through the
    # original unauthenticated endpoint.  Owned documents require their owner
    # (or an administrator).
    if document.owner_id is None and not document.is_private:
        return
    if user is None:
        raise HTTPException(status_code=401, detail="Not authenticated")
    if user.role != "admin" and user.id != document.owner_id:
        raise HTTPException(status_code=403, detail="document permission denied")
