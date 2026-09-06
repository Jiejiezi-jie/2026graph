from dataclasses import dataclass

from app.config import Settings
from app.services.dataset_service import DatasetService
from app.services.workspace_service import WorkspaceService


@dataclass(frozen=True)
class AppContext:
    settings: Settings
    dataset_service: DatasetService
    workspace_service: WorkspaceService
