from fastapi import APIRouter, Depends

from dependency import (
    check_permission,
    get_db,
    require_admin
)

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
    db=Depends(get_db)
):
    return create_document(
        db,
        request.title,
        request.category
    )


@router.get(
    "/",
    response_model=list[DocumentResponse]
)
def get_documents_api(
    db=Depends(get_db)
):
    return get_documents(db)


@router.get(
    "/{document_id}",
    response_model=DocumentResponse
)
def get_document_api(
    document_id: int,
    db=Depends(get_db)
):
    return get_document(
        db,
        document_id
    )


@router.put(
    "/{document_id}",
    response_model=DocumentResponse
)
def update_document_api(
    document_id: int,
    request: DocumentUpdate,
    db=Depends(get_db)
):
    document = get_document(
        db,
        document_id
    )

    return update_document(
        db,
        document,
        request.title,
        request.category
    )


@router.delete(
    "/{document_id}"
)
def delete_document_api(
    document_id: int,
    db=Depends(get_db),
    user=Depends(require_admin)
):
    document = get_document(
        db,
        document_id
    )

    delete_document(
        db,
        document
    )

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