"""Browser automation tools for Canopus."""

from canopus.tools.navigation import (
    NavigateToTool,
    GoBackTool,
    GoForwardTool,
    RefreshTool,
    GetSnapshotTool,
    ListTabsTool,
    NewTabTool,
    SwitchTabTool,
    CloseTabTool,
)
from canopus.tools.interaction import (
    ClickTool,
    TypeTextTool,
    PressKeyTool,
    ScrollTool,
    HoverTool,
    DoubleClickTool,
    SelectOptionTool,
)
from canopus.tools.extraction import (
    ScreenshotTool,
    ExtractTextTool,
    EvaluateJsTool,
    GetLinksTool,
    GetPageInfoTool,
)
from canopus.tools.forms import FillFormTool
from canopus.tools.analysis import WaitForElementTool, FindElementsTool
from canopus.tools.files import (
    UploadFileTool,
    SetDownloadPathTool,
    WaitForDownloadTool,
)
from canopus.tools.storage import (
    GetCookiesTool,
    SetCookieTool,
    ClearCookiesTool,
    GetLocalStorageTool,
    SetLocalStorageTool,
    ClearStorageTool,
)
from canopus.tools.network import (
    EnableNetworkMonitoringTool,
    GetNetworkLogTool,
    ClearNetworkLogTool,
    BlockResourcesTool,
)
from canopus.tools.device import (
    SetViewportTool,
    EmulateDeviceTool,
    SetGeolocationTool,
    SetTimezoneTool,
)

__all__ = [
    # Navigation
    "NavigateToTool",
    "GoBackTool",
    "GoForwardTool",
    "RefreshTool",
    "GetSnapshotTool",
    # Tab Management
    "ListTabsTool",
    "NewTabTool",
    "SwitchTabTool",
    "CloseTabTool",
    # Interaction
    "ClickTool",
    "TypeTextTool",
    "PressKeyTool",
    "ScrollTool",
    "HoverTool",
    "DoubleClickTool",
    "SelectOptionTool",
    # Extraction
    "ScreenshotTool",
    "ExtractTextTool",
    "EvaluateJsTool",
    "GetLinksTool",
    "GetPageInfoTool",
    # Forms
    "FillFormTool",
    # Analysis
    "WaitForElementTool",
    "FindElementsTool",
    # Files
    "UploadFileTool",
    "SetDownloadPathTool",
    "WaitForDownloadTool",
    # Storage
    "GetCookiesTool",
    "SetCookieTool",
    "ClearCookiesTool",
    "GetLocalStorageTool",
    "SetLocalStorageTool",
    "ClearStorageTool",
    # Network
    "EnableNetworkMonitoringTool",
    "GetNetworkLogTool",
    "ClearNetworkLogTool",
    "BlockResourcesTool",
    # Device
    "SetViewportTool",
    "EmulateDeviceTool",
    "SetGeolocationTool",
    "SetTimezoneTool",
]
