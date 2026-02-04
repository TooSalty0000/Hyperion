"""Browser session management for Canopus."""

import asyncio
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, List, Any

logger = logging.getLogger(__name__)


@dataclass
class PageSnapshot:
    """Snapshot of a page's structure and content."""
    url: str
    title: str
    content: str  # Page structure, interactive elements, and text content
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class BrowserSession:
    """
    Represents a single browser session with page management.

    Each session has its own browser context and page, allowing
    isolated browsing with separate cookies, storage, etc.
    """
    session_id: str
    channel_id: int
    browser_type: str
    headless: bool
    viewport_width: int
    viewport_height: int

    # Playwright objects (set after creation)
    _browser: Any = field(default=None, repr=False)
    _context: Any = field(default=None, repr=False)
    _page: Any = field(default=None, repr=False)
    _playwright: Any = field(default=None, repr=False)

    # Session state
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_activity: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    history: List[str] = field(default_factory=list)
    is_active: bool = True

    # Last snapshot for reference
    _last_snapshot: Optional[PageSnapshot] = field(default=None, repr=False)

    @classmethod
    async def create(
        cls,
        session_id: str,
        channel_id: int,
        browser_type: str = "chromium",
        headless: bool = True,
        viewport_width: int = 1280,
        viewport_height: int = 720,
    ) -> "BrowserSession":
        """Create and initialize a new browser session."""
        from playwright.async_api import async_playwright

        session = cls(
            session_id=session_id,
            channel_id=channel_id,
            browser_type=browser_type,
            headless=headless,
            viewport_width=viewport_width,
            viewport_height=viewport_height,
        )

        # Start playwright
        session._playwright = await async_playwright().start()

        # Launch browser based on type
        browser_launchers = {
            "chromium": session._playwright.chromium,
            "firefox": session._playwright.firefox,
            "webkit": session._playwright.webkit,
        }

        launcher = browser_launchers.get(browser_type, session._playwright.chromium)

        # Anti-detection launch args (chromium only)
        launch_args = []
        if browser_type == "chromium":
            launch_args = [
                "--disable-blink-features=AutomationControlled",
                "--no-first-run",
                "--no-default-browser-check",
                "--disable-infobars",
                "--disable-dev-shm-usage",
            ]

        session._browser = await launcher.launch(
            headless=headless,
            args=launch_args if launch_args else None,
        )

        # Create context with viewport and modern user agent
        session._context = await session._browser.new_context(
            viewport={"width": viewport_width, "height": viewport_height},
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/131.0.0.0 Safari/537.36"
            ),
            ignore_https_errors=True,
        )

        # Override navigator.webdriver on every page load
        await session._context.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
        """)

        # Create initial page
        session._page = await session._context.new_page()

        logger.info(f"Browser session {session_id} created for channel {channel_id}")
        return session

    @property
    def page(self) -> Any:
        """Get the current page."""
        return self._page

    @property
    def context(self) -> Any:
        """Get the browser context."""
        return self._context

    @property
    def current_url(self) -> str:
        """Get the current page URL."""
        if self._page:
            return self._page.url
        return ""

    def _update_activity(self):
        """Update last activity timestamp."""
        self.last_activity = datetime.now(timezone.utc)

    async def navigate(self, url: str, wait_until: str = "load") -> Dict[str, Any]:
        """
        Navigate to a URL.

        Args:
            url: URL to navigate to
            wait_until: When to consider navigation complete
                       ("load", "domcontentloaded", "networkidle")

        Returns:
            Dict with navigation result
        """
        if not self._page:
            raise RuntimeError("Browser session not initialized")

        self._update_activity()

        try:
            response = await self._page.goto(url, wait_until=wait_until)
            self.history.append(url)

            return {
                "success": True,
                "url": self._page.url,
                "title": await self._page.title(),
                "status": response.status if response else None,
            }
        except Exception as e:
            logger.error(f"Navigation error: {e}")
            return {
                "success": False,
                "error": str(e),
                "url": url,
            }

    async def go_back(self) -> Dict[str, Any]:
        """Navigate back in history."""
        if not self._page:
            raise RuntimeError("Browser session not initialized")

        self._update_activity()

        try:
            await self._page.go_back()
            return {
                "success": True,
                "url": self._page.url,
                "title": await self._page.title(),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def get_snapshot(self) -> PageSnapshot:
        """
        Get a snapshot of the current page structure.

        This provides a text representation of the page structure
        that can be used by the LLM to understand page content.
        Uses JavaScript to extract interactive elements and content.
        """
        if not self._page:
            raise RuntimeError("Browser session not initialized")

        self._update_activity()

        content = ""

        # Extract interactive elements and page structure via JavaScript
        try:
            content = await self._page.evaluate("""() => {
                const elements = [];

                // Get all interactive elements
                const interactiveSelectors = [
                    'a[href]', 'button', 'input', 'select', 'textarea',
                    '[role="button"]', '[role="link"]', '[role="textbox"]',
                    '[role="checkbox"]', '[role="radio"]', '[role="combobox"]',
                    '[onclick]', '[tabindex]'
                ];

                const interactive = document.querySelectorAll(interactiveSelectors.join(','));
                interactive.forEach((el, idx) => {
                    const tag = el.tagName.toLowerCase();
                    const role = el.getAttribute('role') || tag;
                    const text = (el.innerText || el.value || el.placeholder || el.getAttribute('aria-label') || '').trim().slice(0, 80);
                    const href = el.getAttribute('href');
                    const type = el.getAttribute('type');
                    const id = el.id;
                    const testId = el.getAttribute('data-testid');
                    const name = el.getAttribute('name');

                    // Build a usable CSS selector
                    let selector = '';
                    if (id) {
                        selector = `#${id}`;
                    } else if (testId) {
                        selector = `[data-testid="${testId}"]`;
                    } else if (name && (tag === 'input' || tag === 'select' || tag === 'textarea')) {
                        selector = `${tag}[name="${name}"]`;
                    } else if (text && text.length < 30) {
                        // Use text selector for short, unique text
                        selector = `text="${text.replace(/"/g, '\\\\"')}"`;
                    }

                    let desc = `[${idx}]`;
                    if (selector) desc += ` selector="${selector}"`;
                    desc += ` [${role}]`;
                    if (text) desc += ` "${text.slice(0, 50)}"`;
                    if (href) desc += ` -> ${href.slice(0, 40)}`;
                    if (type) desc += ` (${type})`;

                    elements.push(desc);
                });

                // Get page headings for structure
                const headings = [];
                document.querySelectorAll('h1, h2, h3').forEach(h => {
                    const level = h.tagName.toLowerCase();
                    const text = h.innerText.trim().slice(0, 100);
                    if (text) headings.push(`${level}: ${text}`);
                });

                // Get main text content (limited)
                const mainText = (document.body.innerText || '').slice(0, 3000);

                let result = '';
                if (headings.length > 0) {
                    result += '## Page Structure\\n' + headings.join('\\n') + '\\n\\n';
                }
                if (elements.length > 0) {
                    result += '## Interactive Elements (use selector= value for click/type)\\n' + elements.slice(0, 50).join('\\n') + '\\n\\n';
                }
                result += '## Content Preview\\n' + mainText;

                return result;
            }""")
        except Exception as e:
            logger.warning(f"Page snapshot extraction failed: {e}")
            # Fallback to simple text content
            try:
                content = await self._page.evaluate("document.body.innerText")
                content = content[:5000] if content else ""
            except Exception:
                content = "[Unable to extract page content]"

        snapshot = PageSnapshot(
            url=self._page.url,
            title=await self._page.title(),
            content=content,
        )
        self._last_snapshot = snapshot
        return snapshot

    async def click(self, selector: str) -> Dict[str, Any]:
        """Click an element by selector."""
        if not self._page:
            raise RuntimeError("Browser session not initialized")

        self._update_activity()

        try:
            await self._page.click(selector, timeout=5000)
            return {"success": True, "selector": selector}
        except Exception as e:
            return {"success": False, "error": str(e), "selector": selector}

    async def click_by_index(self, index: int) -> Dict[str, Any]:
        """Click an interactive element by its index from the snapshot."""
        if not self._page:
            raise RuntimeError("Browser session not initialized")

        self._update_activity()

        try:
            # Get all interactive elements in same order as snapshot
            result = await self._page.evaluate("""(targetIndex) => {
                const interactiveSelectors = [
                    'a[href]', 'button', 'input', 'select', 'textarea',
                    '[role="button"]', '[role="link"]', '[role="textbox"]',
                    '[role="checkbox"]', '[role="radio"]', '[role="combobox"]',
                    '[onclick]', '[tabindex]'
                ];

                const elements = document.querySelectorAll(interactiveSelectors.join(','));
                if (targetIndex < 0 || targetIndex >= elements.length) {
                    return { success: false, error: `Index ${targetIndex} out of range (0-${elements.length - 1})` };
                }

                const el = elements[targetIndex];
                const rect = el.getBoundingClientRect();

                // Check if element is visible and clickable
                if (rect.width === 0 || rect.height === 0) {
                    return { success: false, error: 'Element has no visible size' };
                }

                // Return element info for clicking
                return {
                    success: true,
                    tag: el.tagName.toLowerCase(),
                    text: (el.innerText || el.value || '').trim().slice(0, 50),
                    x: rect.x + rect.width / 2,
                    y: rect.y + rect.height / 2
                };
            }""", index)

            if not result.get("success"):
                return result

            # Click at the center of the element
            await self._page.mouse.click(result["x"], result["y"])

            return {
                "success": True,
                "index": index,
                "tag": result.get("tag"),
                "text": result.get("text"),
            }
        except Exception as e:
            return {"success": False, "error": str(e), "index": index}

    async def type_text(self, selector: str, text: str, clear: bool = False) -> Dict[str, Any]:
        """Type text into an element."""
        if not self._page:
            raise RuntimeError("Browser session not initialized")

        self._update_activity()

        try:
            if clear:
                await self._page.fill(selector, text)
            else:
                await self._page.type(selector, text)
            return {"success": True, "selector": selector, "text": text}
        except Exception as e:
            return {"success": False, "error": str(e), "selector": selector}

    async def press_key(self, key: str) -> Dict[str, Any]:
        """Press a keyboard key."""
        if not self._page:
            raise RuntimeError("Browser session not initialized")

        self._update_activity()

        try:
            await self._page.keyboard.press(key)
            return {"success": True, "key": key}
        except Exception as e:
            return {"success": False, "error": str(e), "key": key}

    async def scroll(self, direction: str = "down", amount: int = 500) -> Dict[str, Any]:
        """Scroll the page."""
        if not self._page:
            raise RuntimeError("Browser session not initialized")

        self._update_activity()

        try:
            delta = amount if direction == "down" else -amount
            await self._page.mouse.wheel(0, delta)
            return {"success": True, "direction": direction, "amount": amount}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def screenshot(
        self,
        path: Optional[str] = None,
        full_page: bool = False,
        selector: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Take a screenshot of the page or element."""
        if not self._page:
            raise RuntimeError("Browser session not initialized")

        self._update_activity()

        try:
            if selector:
                element = await self._page.query_selector(selector)
                if element:
                    data = await element.screenshot(path=path)
                else:
                    return {"success": False, "error": f"Element not found: {selector}"}
            else:
                data = await self._page.screenshot(path=path, full_page=full_page)

            return {
                "success": True,
                "path": path,
                "bytes": data if not path else None,
                "full_page": full_page,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def evaluate(self, expression: str) -> Dict[str, Any]:
        """Evaluate JavaScript expression on the page."""
        if not self._page:
            raise RuntimeError("Browser session not initialized")

        self._update_activity()

        try:
            result = await self._page.evaluate(expression)
            return {"success": True, "result": result}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def wait_for_selector(
        self,
        selector: str,
        timeout: int = 10000,
        state: str = "visible",
    ) -> Dict[str, Any]:
        """Wait for an element to appear."""
        if not self._page:
            raise RuntimeError("Browser session not initialized")

        self._update_activity()

        try:
            await self._page.wait_for_selector(selector, timeout=timeout, state=state)
            return {"success": True, "selector": selector}
        except Exception as e:
            return {"success": False, "error": str(e), "selector": selector}

    async def get_console_logs(self) -> List[Dict[str, Any]]:
        """Get console messages (requires prior setup of listener)."""
        # This would need to be set up during page creation
        return []

    # =========================================================================
    # Tab Management
    # =========================================================================

    async def list_tabs(self) -> List[Dict[str, Any]]:
        """List all open tabs/pages in the browser context."""
        if not self._context:
            raise RuntimeError("Browser session not initialized")

        self._update_activity()

        tabs = []
        for i, page in enumerate(self._context.pages):
            is_current = page == self._page
            try:
                title = await page.title()
            except Exception:
                title = "(untitled)"

            tabs.append({
                "index": i,
                "url": page.url,
                "title": title,
                "is_current": is_current,
            })

        return tabs

    async def new_tab(self, url: Optional[str] = None) -> Dict[str, Any]:
        """Open a new tab and optionally navigate to a URL."""
        if not self._context:
            raise RuntimeError("Browser session not initialized")

        self._update_activity()

        try:
            new_page = await self._context.new_page()
            self._page = new_page  # Switch to new tab

            if url:
                response = await new_page.goto(url, wait_until="load")
                self.history.append(url)
                return {
                    "success": True,
                    "url": new_page.url,
                    "title": await new_page.title(),
                    "status": response.status if response else None,
                    "tab_count": len(self._context.pages),
                }

            return {
                "success": True,
                "url": "about:blank",
                "title": "",
                "tab_count": len(self._context.pages),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def switch_tab(self, index: int) -> Dict[str, Any]:
        """Switch to a tab by index."""
        if not self._context:
            raise RuntimeError("Browser session not initialized")

        self._update_activity()

        pages = self._context.pages
        if index < 0 or index >= len(pages):
            return {
                "success": False,
                "error": f"Tab index {index} out of range (0-{len(pages)-1})",
            }

        try:
            self._page = pages[index]
            await self._page.bring_to_front()
            return {
                "success": True,
                "index": index,
                "url": self._page.url,
                "title": await self._page.title(),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def close_tab(self, index: Optional[int] = None) -> Dict[str, Any]:
        """Close a tab by index. If no index, closes current tab."""
        if not self._context:
            raise RuntimeError("Browser session not initialized")

        self._update_activity()

        pages = self._context.pages
        if len(pages) <= 1:
            return {"success": False, "error": "Cannot close the last tab"}

        try:
            if index is not None:
                if index < 0 or index >= len(pages):
                    return {
                        "success": False,
                        "error": f"Tab index {index} out of range",
                    }
                page_to_close = pages[index]
            else:
                page_to_close = self._page
                index = pages.index(self._page)

            await page_to_close.close()

            # Switch to another tab if we closed the current one
            if page_to_close == self._page:
                remaining = self._context.pages
                self._page = remaining[min(index, len(remaining) - 1)]

            return {
                "success": True,
                "closed_index": index,
                "remaining_tabs": len(self._context.pages),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    # =========================================================================
    # Additional Navigation
    # =========================================================================

    async def refresh(self) -> Dict[str, Any]:
        """Refresh the current page."""
        if not self._page:
            raise RuntimeError("Browser session not initialized")

        self._update_activity()

        try:
            response = await self._page.reload(wait_until="load")
            return {
                "success": True,
                "url": self._page.url,
                "title": await self._page.title(),
                "status": response.status if response else None,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def go_forward(self) -> Dict[str, Any]:
        """Navigate forward in history."""
        if not self._page:
            raise RuntimeError("Browser session not initialized")

        self._update_activity()

        try:
            await self._page.go_forward()
            return {
                "success": True,
                "url": self._page.url,
                "title": await self._page.title(),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    # =========================================================================
    # Additional Interaction
    # =========================================================================

    async def hover(self, selector: str) -> Dict[str, Any]:
        """Hover over an element."""
        if not self._page:
            raise RuntimeError("Browser session not initialized")

        self._update_activity()

        try:
            await self._page.hover(selector)
            return {"success": True, "selector": selector}
        except Exception as e:
            return {"success": False, "error": str(e), "selector": selector}

    async def double_click(self, selector: str) -> Dict[str, Any]:
        """Double-click an element."""
        if not self._page:
            raise RuntimeError("Browser session not initialized")

        self._update_activity()

        try:
            await self._page.dblclick(selector)
            return {"success": True, "selector": selector}
        except Exception as e:
            return {"success": False, "error": str(e), "selector": selector}

    async def select_option(self, selector: str, value: str) -> Dict[str, Any]:
        """Select an option from a dropdown."""
        if not self._page:
            raise RuntimeError("Browser session not initialized")

        self._update_activity()

        try:
            await self._page.select_option(selector, value)
            return {"success": True, "selector": selector, "value": value}
        except Exception as e:
            return {"success": False, "error": str(e), "selector": selector}

    # =========================================================================
    # Additional Extraction
    # =========================================================================

    async def get_links(self, selector: Optional[str] = None) -> Dict[str, Any]:
        """Extract all links from the page or a specific element."""
        if not self._page:
            raise RuntimeError("Browser session not initialized")

        self._update_activity()

        try:
            if selector:
                js_code = f'''
                    Array.from(document.querySelector("{selector}")?.querySelectorAll("a") || [])
                        .map(a => ({{ href: a.href, text: a.innerText.trim().substring(0, 100) }}))
                        .filter(l => l.href)
                '''
            else:
                js_code = '''
                    Array.from(document.querySelectorAll("a"))
                        .map(a => ({ href: a.href, text: a.innerText.trim().substring(0, 100) }))
                        .filter(l => l.href)
                '''

            links = await self._page.evaluate(js_code)
            return {"success": True, "links": links, "count": len(links)}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def get_page_info(self) -> Dict[str, Any]:
        """Get comprehensive information about the current page."""
        if not self._page:
            raise RuntimeError("Browser session not initialized")

        self._update_activity()

        try:
            info = await self._page.evaluate('''
                () => ({
                    url: window.location.href,
                    title: document.title,
                    description: document.querySelector('meta[name="description"]')?.content || "",
                    keywords: document.querySelector('meta[name="keywords"]')?.content || "",
                    author: document.querySelector('meta[name="author"]')?.content || "",
                    canonical: document.querySelector('link[rel="canonical"]')?.href || "",
                    language: document.documentElement.lang || "",
                    viewport: {
                        width: window.innerWidth,
                        height: window.innerHeight,
                        scrollX: window.scrollX,
                        scrollY: window.scrollY,
                        scrollHeight: document.body.scrollHeight,
                        scrollWidth: document.body.scrollWidth,
                    },
                    counts: {
                        links: document.querySelectorAll("a").length,
                        images: document.querySelectorAll("img").length,
                        forms: document.querySelectorAll("form").length,
                        inputs: document.querySelectorAll("input, textarea, select").length,
                        buttons: document.querySelectorAll("button, input[type='submit']").length,
                    },
                })
            ''')
            return {"success": True, **info}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def find_elements(
        self,
        selector: str,
        limit: int = 20,
    ) -> Dict[str, Any]:
        """Find elements matching a selector and return info about them."""
        if not self._page:
            raise RuntimeError("Browser session not initialized")

        self._update_activity()

        try:
            elements = await self._page.evaluate(f'''
                (limit) => {{
                    const els = Array.from(document.querySelectorAll("{selector}")).slice(0, limit);
                    return els.map((el, i) => ({{
                        index: i,
                        tag: el.tagName.toLowerCase(),
                        id: el.id || null,
                        className: el.className || null,
                        text: el.innerText?.trim().substring(0, 100) || null,
                        href: el.href || null,
                        src: el.src || null,
                        type: el.type || null,
                        name: el.name || null,
                        value: el.value?.substring(0, 50) || null,
                        placeholder: el.placeholder || null,
                        isVisible: el.offsetParent !== null,
                    }}));
                }}
            ''', limit)

            return {
                "success": True,
                "elements": elements,
                "count": len(elements),
                "selector": selector,
            }
        except Exception as e:
            return {"success": False, "error": str(e), "selector": selector}

    async def get_attribute(self, selector: str, attribute: str) -> Dict[str, Any]:
        """Get an attribute value from an element."""
        if not self._page:
            raise RuntimeError("Browser session not initialized")

        self._update_activity()

        try:
            value = await self._page.get_attribute(selector, attribute)
            return {"success": True, "selector": selector, "attribute": attribute, "value": value}
        except Exception as e:
            return {"success": False, "error": str(e)}

    # =========================================================================
    # File Upload/Download
    # =========================================================================

    async def upload_file(self, selector: str, file_path: str) -> Dict[str, Any]:
        """Upload a file to a file input element."""
        if not self._page:
            raise RuntimeError("Browser session not initialized")

        self._update_activity()

        try:
            # Verify file exists
            path = Path(file_path)
            if not path.exists():
                return {"success": False, "error": f"File not found: {file_path}"}

            await self._page.set_input_files(selector, file_path)
            return {
                "success": True,
                "selector": selector,
                "file": path.name,
                "size": path.stat().st_size,
            }
        except Exception as e:
            return {"success": False, "error": str(e), "selector": selector}

    async def set_download_path(self, download_path: str) -> Dict[str, Any]:
        """Set the download directory for the session."""
        if not self._context:
            raise RuntimeError("Browser session not initialized")

        self._update_activity()

        try:
            path = Path(download_path)
            path.mkdir(parents=True, exist_ok=True)
            # Store for reference
            self._download_path = path
            return {"success": True, "download_path": str(path)}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def wait_for_download(self, timeout: int = 30000) -> Dict[str, Any]:
        """Wait for a download to complete and return file info."""
        if not self._page:
            raise RuntimeError("Browser session not initialized")

        self._update_activity()

        try:
            async with self._page.expect_download(timeout=timeout) as download_info:
                download = await download_info.value
                # Save to download path if set
                save_path = None
                if hasattr(self, '_download_path') and self._download_path:
                    save_path = self._download_path / download.suggested_filename
                    await download.save_as(save_path)

                return {
                    "success": True,
                    "filename": download.suggested_filename,
                    "url": download.url,
                    "saved_path": str(save_path) if save_path else None,
                }
        except Exception as e:
            return {"success": False, "error": str(e)}

    # =========================================================================
    # Cookie and Storage Management
    # =========================================================================

    async def get_cookies(self, urls: Optional[List[str]] = None) -> Dict[str, Any]:
        """Get cookies from the browser context."""
        if not self._context:
            raise RuntimeError("Browser session not initialized")

        self._update_activity()

        try:
            if urls:
                cookies = await self._context.cookies(urls)
            else:
                cookies = await self._context.cookies()

            return {
                "success": True,
                "cookies": [
                    {
                        "name": c["name"],
                        "value": c["value"][:50] + "..." if len(c["value"]) > 50 else c["value"],
                        "domain": c.get("domain", ""),
                        "path": c.get("path", "/"),
                        "expires": c.get("expires"),
                        "httpOnly": c.get("httpOnly", False),
                        "secure": c.get("secure", False),
                    }
                    for c in cookies
                ],
                "count": len(cookies),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def set_cookie(
        self,
        name: str,
        value: str,
        domain: Optional[str] = None,
        path: str = "/",
        expires: Optional[int] = None,
        http_only: bool = False,
        secure: bool = False,
    ) -> Dict[str, Any]:
        """Set a cookie in the browser context."""
        if not self._context:
            raise RuntimeError("Browser session not initialized")

        self._update_activity()

        try:
            cookie = {
                "name": name,
                "value": value,
                "path": path,
                "httpOnly": http_only,
                "secure": secure,
            }

            # Domain is required - use current page domain if not specified
            if domain:
                cookie["domain"] = domain
            elif self._page:
                from urllib.parse import urlparse
                parsed = urlparse(self._page.url)
                cookie["domain"] = parsed.netloc
                cookie["url"] = self._page.url
            else:
                return {"success": False, "error": "Domain required when no page is loaded"}

            if expires:
                cookie["expires"] = expires

            await self._context.add_cookies([cookie])
            return {"success": True, "cookie": name, "domain": cookie.get("domain")}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def clear_cookies(self) -> Dict[str, Any]:
        """Clear all cookies from the browser context."""
        if not self._context:
            raise RuntimeError("Browser session not initialized")

        self._update_activity()

        try:
            await self._context.clear_cookies()
            return {"success": True, "message": "All cookies cleared"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def get_local_storage(self) -> Dict[str, Any]:
        """Get all localStorage items from the current page."""
        if not self._page:
            raise RuntimeError("Browser session not initialized")

        self._update_activity()

        try:
            storage = await self._page.evaluate('''
                () => {
                    const items = {};
                    for (let i = 0; i < localStorage.length; i++) {
                        const key = localStorage.key(i);
                        items[key] = localStorage.getItem(key);
                    }
                    return items;
                }
            ''')
            return {"success": True, "storage": storage, "count": len(storage)}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def set_local_storage(self, key: str, value: str) -> Dict[str, Any]:
        """Set a localStorage item on the current page."""
        if not self._page:
            raise RuntimeError("Browser session not initialized")

        self._update_activity()

        try:
            await self._page.evaluate(
                '([key, value]) => localStorage.setItem(key, value)',
                [key, value]
            )
            return {"success": True, "key": key}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def clear_storage(self, storage_type: str = "all") -> Dict[str, Any]:
        """Clear browser storage (localStorage, sessionStorage, or both)."""
        if not self._page:
            raise RuntimeError("Browser session not initialized")

        self._update_activity()

        try:
            if storage_type in ["local", "all"]:
                await self._page.evaluate('localStorage.clear()')
            if storage_type in ["session", "all"]:
                await self._page.evaluate('sessionStorage.clear()')

            return {"success": True, "cleared": storage_type}
        except Exception as e:
            return {"success": False, "error": str(e)}

    # =========================================================================
    # Network Interception and Monitoring
    # =========================================================================

    async def enable_request_interception(self) -> Dict[str, Any]:
        """Enable request/response monitoring."""
        if not self._page:
            raise RuntimeError("Browser session not initialized")

        self._update_activity()

        try:
            # Initialize request log
            if not hasattr(self, '_network_log'):
                self._network_log = []

            async def on_request(request):
                self._network_log.append({
                    "type": "request",
                    "url": request.url,
                    "method": request.method,
                    "resource_type": request.resource_type,
                    "headers": dict(request.headers),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })

            async def on_response(response):
                self._network_log.append({
                    "type": "response",
                    "url": response.url,
                    "status": response.status,
                    "headers": dict(response.headers),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })

            self._page.on("request", on_request)
            self._page.on("response", on_response)
            self._network_monitoring = True

            return {"success": True, "message": "Network monitoring enabled"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def get_network_log(
        self,
        filter_type: Optional[str] = None,
        limit: int = 50,
    ) -> Dict[str, Any]:
        """Get captured network requests/responses."""
        if not self._page:
            raise RuntimeError("Browser session not initialized")

        self._update_activity()

        try:
            if not hasattr(self, '_network_log'):
                return {"success": True, "log": [], "message": "No network monitoring active"}

            log = self._network_log
            if filter_type:
                log = [e for e in log if e.get("type") == filter_type]

            # Return most recent entries
            log = log[-limit:]

            return {
                "success": True,
                "log": log,
                "count": len(log),
                "total": len(self._network_log),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def clear_network_log(self) -> Dict[str, Any]:
        """Clear the captured network log."""
        self._update_activity()

        try:
            if hasattr(self, '_network_log'):
                self._network_log = []
            return {"success": True, "message": "Network log cleared"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def block_resources(self, resource_types: List[str]) -> Dict[str, Any]:
        """Block specific resource types (images, stylesheets, fonts, etc.)."""
        if not self._page:
            raise RuntimeError("Browser session not initialized")

        self._update_activity()

        try:
            async def route_handler(route):
                if route.request.resource_type in resource_types:
                    await route.abort()
                else:
                    await route.continue_()

            await self._page.route("**/*", route_handler)
            return {"success": True, "blocked_types": resource_types}
        except Exception as e:
            return {"success": False, "error": str(e)}

    # =========================================================================
    # Mobile Viewport and Device Emulation
    # =========================================================================

    async def set_viewport(
        self,
        width: int,
        height: int,
        device_scale_factor: float = 1.0,
        is_mobile: bool = False,
        has_touch: bool = False,
    ) -> Dict[str, Any]:
        """Set viewport size and device characteristics."""
        if not self._page:
            raise RuntimeError("Browser session not initialized")

        self._update_activity()

        try:
            await self._page.set_viewport_size({"width": width, "height": height})

            # Update stored dimensions
            self.viewport_width = width
            self.viewport_height = height

            return {
                "success": True,
                "viewport": {
                    "width": width,
                    "height": height,
                    "deviceScaleFactor": device_scale_factor,
                    "isMobile": is_mobile,
                    "hasTouch": has_touch,
                },
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def emulate_device(self, device_name: str) -> Dict[str, Any]:
        """Emulate a specific device (iPhone, Pixel, iPad, etc.)."""
        if not self._context or not self._playwright:
            raise RuntimeError("Browser session not initialized")

        self._update_activity()

        # Common device presets
        devices = {
            "iphone_12": {
                "viewport": {"width": 390, "height": 844},
                "user_agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1",
                "device_scale_factor": 3,
                "is_mobile": True,
                "has_touch": True,
            },
            "iphone_14": {
                "viewport": {"width": 393, "height": 852},
                "user_agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0 Mobile/15E148 Safari/604.1",
                "device_scale_factor": 3,
                "is_mobile": True,
                "has_touch": True,
            },
            "pixel_5": {
                "viewport": {"width": 393, "height": 851},
                "user_agent": "Mozilla/5.0 (Linux; Android 11; Pixel 5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.91 Mobile Safari/537.36",
                "device_scale_factor": 2.75,
                "is_mobile": True,
                "has_touch": True,
            },
            "ipad": {
                "viewport": {"width": 768, "height": 1024},
                "user_agent": "Mozilla/5.0 (iPad; CPU OS 14_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1",
                "device_scale_factor": 2,
                "is_mobile": True,
                "has_touch": True,
            },
            "ipad_pro": {
                "viewport": {"width": 1024, "height": 1366},
                "user_agent": "Mozilla/5.0 (iPad; CPU OS 14_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1",
                "device_scale_factor": 2,
                "is_mobile": True,
                "has_touch": True,
            },
            "desktop_hd": {
                "viewport": {"width": 1920, "height": 1080},
                "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "device_scale_factor": 1,
                "is_mobile": False,
                "has_touch": False,
            },
            "desktop_4k": {
                "viewport": {"width": 3840, "height": 2160},
                "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "device_scale_factor": 1,
                "is_mobile": False,
                "has_touch": False,
            },
        }

        device_key = device_name.lower().replace(" ", "_").replace("-", "_")
        device = devices.get(device_key)

        if not device:
            return {
                "success": False,
                "error": f"Unknown device: {device_name}",
                "available_devices": list(devices.keys()),
            }

        try:
            # Set viewport
            await self._page.set_viewport_size(device["viewport"])

            # We can't change user agent on existing context, but we can note it
            self.viewport_width = device["viewport"]["width"]
            self.viewport_height = device["viewport"]["height"]

            return {
                "success": True,
                "device": device_name,
                "viewport": device["viewport"],
                "is_mobile": device["is_mobile"],
                "note": "User agent cannot be changed on existing session. Create new session for full device emulation.",
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def set_geolocation(
        self,
        latitude: float,
        longitude: float,
        accuracy: float = 100,
    ) -> Dict[str, Any]:
        """Set geolocation for the browser context."""
        if not self._context:
            raise RuntimeError("Browser session not initialized")

        self._update_activity()

        try:
            await self._context.set_geolocation({
                "latitude": latitude,
                "longitude": longitude,
                "accuracy": accuracy,
            })
            await self._context.grant_permissions(["geolocation"])

            return {
                "success": True,
                "geolocation": {
                    "latitude": latitude,
                    "longitude": longitude,
                    "accuracy": accuracy,
                },
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def set_timezone(self, timezone_id: str) -> Dict[str, Any]:
        """Set timezone for the browser context."""
        if not self._context:
            raise RuntimeError("Browser session not initialized")

        self._update_activity()

        try:
            # Timezone is set at context creation, but we can emulate via JS
            await self._page.evaluate(f'''
                () => {{
                    Object.defineProperty(Intl, 'DateTimeFormat', {{
                        value: new Proxy(Intl.DateTimeFormat, {{
                            construct: (target, args) => {{
                                args[1] = {{ ...args[1], timeZone: "{timezone_id}" }};
                                return new target(...args);
                            }}
                        }})
                    }});
                }}
            ''')
            return {"success": True, "timezone": timezone_id}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def close(self):
        """Close the browser session and clean up resources."""
        self.is_active = False

        try:
            if self._page:
                await self._page.close()
            if self._context:
                await self._context.close()
            if self._browser:
                await self._browser.close()
            if self._playwright:
                await self._playwright.stop()

            logger.info(f"Browser session {self.session_id} closed")
        except Exception as e:
            logger.error(f"Error closing browser session: {e}")


class BrowserSessionManager:
    """
    Manages browser sessions across Discord channels.

    Provides session lifecycle management, lookup, and cleanup.
    """

    def __init__(
        self,
        browser_type: str = "chromium",
        headless: bool = True,
        viewport_width: int = 1280,
        viewport_height: int = 720,
        max_sessions_per_channel: int = 1,
        session_idle_timeout: int = 1800,  # 30 minutes
        screenshot_dir: str = "./screenshots",
    ):
        self.browser_type = browser_type
        self.headless = headless
        self.viewport_width = viewport_width
        self.viewport_height = viewport_height
        self.max_sessions_per_channel = max_sessions_per_channel
        self.session_idle_timeout = session_idle_timeout
        self.screenshot_dir = Path(screenshot_dir)

        # Sessions indexed by channel_id
        self._sessions: Dict[int, BrowserSession] = {}
        self._session_counter = 0
        self._lock = asyncio.Lock()

        # Ensure screenshot directory exists
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)

    async def get_or_create(self, channel_id: int) -> BrowserSession:
        """
        Get existing session for channel or create a new one.

        Args:
            channel_id: Discord channel ID

        Returns:
            BrowserSession instance
        """
        async with self._lock:
            # Check for existing active session
            if channel_id in self._sessions:
                session = self._sessions[channel_id]
                if session.is_active:
                    return session
                else:
                    # Clean up dead session
                    del self._sessions[channel_id]

            # Create new session
            self._session_counter += 1
            session_id = f"canopus-{channel_id}-{self._session_counter}"

            session = await BrowserSession.create(
                session_id=session_id,
                channel_id=channel_id,
                browser_type=self.browser_type,
                headless=self.headless,
                viewport_width=self.viewport_width,
                viewport_height=self.viewport_height,
            )

            self._sessions[channel_id] = session
            logger.info(f"Created browser session {session_id} for channel {channel_id}")

            return session

    async def get(self, channel_id: int) -> Optional[BrowserSession]:
        """Get existing session for channel if it exists."""
        session = self._sessions.get(channel_id)
        if session and session.is_active:
            return session
        return None

    async def close(self, channel_id: int) -> bool:
        """
        Close and remove session for channel.

        Returns:
            True if session was closed, False if no session existed
        """
        async with self._lock:
            session = self._sessions.pop(channel_id, None)
            if session:
                await session.close()
                return True
            return False

    async def close_all(self):
        """Close all browser sessions."""
        async with self._lock:
            for session in self._sessions.values():
                await session.close()
            self._sessions.clear()
            logger.info("All browser sessions closed")

    async def cleanup_idle_sessions(self):
        """Close sessions that have been idle too long."""
        now = datetime.now(timezone.utc)
        to_close = []

        for channel_id, session in self._sessions.items():
            idle_seconds = (now - session.last_activity).total_seconds()
            if idle_seconds > self.session_idle_timeout:
                to_close.append(channel_id)

        for channel_id in to_close:
            await self.close(channel_id)
            logger.info(f"Closed idle session for channel {channel_id}")

    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all active sessions."""
        return [
            {
                "session_id": s.session_id,
                "channel_id": s.channel_id,
                "current_url": s.current_url,
                "created_at": s.created_at.isoformat(),
                "last_activity": s.last_activity.isoformat(),
                "is_active": s.is_active,
            }
            for s in self._sessions.values()
            if s.is_active
        ]

    def get_screenshot_path(self, filename: str) -> Path:
        """Get full path for a screenshot file."""
        return self.screenshot_dir / filename
