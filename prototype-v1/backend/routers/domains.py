"""
routers/domains.py — what label set this deployment serves.

GET /domains
  The active domain and every domain pack installed alongside it.

This exists so the frontend stops keeping its own copy of the label list.
constants.js used to hold the ten legal labels with a comment asking whoever
changed the backend to remember to update them by hand; the label dropdown
now comes from here instead, which is the same place model_service reads its
id2label from.

The `domains` list is what a domain selector needs: one deployment serves one
model, so `active` is the domain predictions are actually made in, and the
rest of the list is what else is installed.
"""

import logging
from typing import List, Optional

from fastapi import APIRouter, Header, status
from pydantic import BaseModel, Field

from config.domains import active_domain, registry

logger = logging.getLogger(__name__)
router = APIRouter()


class DomainLabel(BaseModel):
    id: int = Field(..., description="Index into the model's output logits")
    name: str
    description: str = ""
    related: List[str] = []


class DomainSummary(BaseModel):
    key: str = Field(..., description="Pack filename, e.g. 'legal' for domains/legal.json")
    display_name: str
    description: str = ""
    base_model: str = Field("", description="Checkpoint the domain was fine-tuned from")
    num_labels: int
    labels: List[DomainLabel] = Field(..., description="In id order — the logit order")


class DomainsResponse(BaseModel):
    active: str = Field(..., description="Key of the domain this deployment serves")
    domains: List[DomainSummary]


def _summarize(domain) -> DomainSummary:
    return DomainSummary(
        key=domain.key,
        display_name=domain.display_name,
        description=domain.description,
        base_model=domain.base_model,
        num_labels=domain.num_labels,
        labels=[
            DomainLabel(
                id=l.id, name=l.name, description=l.description, related=list(l.related)
            )
            for l in domain.labels
        ],
    )


@router.get(
    "",
    response_model=DomainsResponse,
    status_code=status.HTTP_200_OK,
    summary="List installed domains and which one is active",
    description=(
        "Return every installed domain pack and the key of the one this "
        "deployment serves. Labels are listed in id order, which is the order "
        "of the model's output logits. Accessible to all authenticated roles."
    ),
)
async def get_domains(
    x_role: Optional[str] = Header(default=None, alias="X-Role"),
) -> DomainsResponse:
    active = active_domain()
    return DomainsResponse(
        active=active.key,
        domains=[_summarize(d) for d in registry().values()],
    )
