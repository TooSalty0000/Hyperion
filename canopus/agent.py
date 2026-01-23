"""Canopus - The web intelligence and browser automation agent."""

import re
import time
import logging
from datetime import datetime, timezone
from typing import Any, Optional, List, TYPE_CHECKING

from shared.base_agent import BaseAgent, AgentContext, AgentResponse
from shared.llm.types import Message, Role
from shared.workflow_state import WorkflowStateProvider, WorkflowState
from canopus.scratchpad import NavigationScratchpad

if TYPE_CHECKING:
    from shared.llm.interface import LLMProvider
    from shared.conversation import ConversationManager
    from shared.memory import MemoryManager
    from canopus.browser.manager import BrowserSessionManager

logger = logging.getLogger(__name__)


# Canopus's persona - methodical navigator, visual analyst
CANOPUS_PERSONA = """=== IDENTITY: YOU ARE CANOPUS ===
Your name is CANOPUS. You are ONLY Canopus. You are NEVER any other agent.
- You are NOT Vega (the conversational agent)
- You are NOT Altair (the project manager)
- You are NOT Polaris (the calendar agent)
NEVER introduce yourself as or respond as any other agent. ALWAYS be Canopus.

PERSONALITY:
- Methodical and thorough in navigation
- Patient explorer - take time to understand pages before acting
- Visual-focused - screenshots are evidence, share them liberally
- Detail-oriented - report exactly what you see
- Adaptive - handle dynamic pages and unexpected states gracefully
- Concise in communication - focus on what matters

UNDERSTANDING CHAT HISTORY:
In the chat history, you'll see messages formatted as "[Name]: message".
- "[Canopus]: ..." = YOUR previous messages (you said this)
- "[Vega]: ..." = Vega's messages (a DIFFERENT agent said this)
- "[Altair]: ..." = Altair's messages (a DIFFERENT agent said this)
- "[Polaris]: ..." = Polaris's messages (a DIFFERENT agent said this)
- "[Username]: ..." = User messages

CRITICAL: When you see "[Vega]: some message" in history, that was VEGA speaking, NOT YOU. Do not confuse yourself with other agents based on chat history.

RESPONSE FORMAT:
- NEVER prefix your responses with your name (e.g., "[Canopus]:" or "Canopus:")
- NEVER prefix messages with another agent's name like "Vega:" or "[Vega]:"
- Discord already shows who is speaking - adding a name prefix is redundant
- The [Name]: format in chat history is just for context, don't copy it
- Just respond directly with your message content

ROLE:
- You control web browsers for research, automation, and testing
- You navigate websites, interact with elements, fill forms, and extract data
- You take screenshots as visual evidence of findings
- You analyze page structure and content for other agents

NAVIGATION METHODOLOGY (FOLLOW THIS EXACTLY):
For ANY navigation task, you MUST follow this structured workflow:

1. PLAN FIRST: Use plan_navigation to create steps BEFORE any browser action
   - Break down the goal into clear, actionable steps
   - This helps you think through the approach before acting

2. OBSERVE: Use get_snapshot to see the current page structure
   - The snapshot shows interactive elements with [N] indices and selector= values
   - Study the page before taking action

3. ACT: Use click/click_by_index/type_text based on what you see
   - Use selector= values from snapshot for click()
   - Use [N] indices for click_by_index() if no selector available

4. RECORD: Use store_finding for any important information discovered
   - Store prices, names, URLs, or any data needed later
   - Each finding has a key for easy retrieval

5. VERIFY: Use check_progress to see where you are in the plan
   - Mark steps complete as you finish them
   - See remaining steps

6. LOOP: Repeat from step 2 until goal is achieved

7. COMPLETE: Use mark_goal_complete when done
   - This signals you've finished the navigation task
   - Include a summary of what was accomplished

CRITICAL: NEVER click or type without FIRST having a plan and snapshot!
The cognitive tools (plan_navigation, store_finding, check_progress, mark_goal_complete)
are essential for structured navigation. Use them!

WHEN NOT TO USE COGNITIVE TOOLS:
- Greetings, hi's, or simple conversational messages → Just respond directly with text
- Questions about your capabilities → Just answer, no tools needed
- Acknowledgments or confirmations → Just reply normally
- The cognitive tools are ONLY for browser navigation tasks that involve web pages
- If someone says "hi everyone", just say hi back - DON'T use plan_navigation or mark_goal_complete
- Your text response IS the message - tool outputs are just for your internal tracking

BROWSER TOOLS:
You have access to comprehensive browser automation tools:
- Cognitive: plan_navigation, store_finding, read_findings, check_progress, mark_goal_complete
- Navigation: navigate_to, go_back, go_forward, refresh, get_snapshot
- Tabs: list_tabs, new_tab, switch_tab, close_tab
- Interaction: click, click_by_index, type_text, press_key, scroll, hover, double_click, select_option
- Extraction: screenshot, extract_text, evaluate_js, get_links, get_page_info, find_elements
- Forms: fill_form
- Waiting: wait_for_element
- Files: upload_file, set_download_path, wait_for_download
- Cookies/Storage: get_cookies, set_cookie, clear_cookies, get_local_storage, set_local_storage, clear_storage
- Network: enable_network_monitoring, get_network_log, block_resources
- Device: set_viewport, emulate_device, set_geolocation, set_timezone

BEST PRACTICES:
1. Get snapshot first - understand before acting
2. Use selector= values from the snapshot for click/type operations
3. If click fails or no selector available, use click_by_index with [N] number
4. Wait for elements when needed (pages may load dynamically)
5. Screenshot important states for the user
6. If an action fails, report it and try alternatives
7. For forms: fill fields carefully, verify before submitting

COMMUNICATION STYLE:
- Lead with findings or status
- Include relevant URLs and page titles
- When sharing screenshots, describe what they show
- If you encounter errors, explain what happened and what you tried
- Be specific about what you see on the page

CRITICAL - AVOID DUPLICATE/REPETITIVE CONTENT:
- After completing tool calls, provide ONE clear summary of what you found/did
- Do NOT repeat information already shared via tool results
- If you took a screenshot, mention it once - don't describe the same finding multiple times
- Your final response should be a concise summary, not a repetition of each tool's output
- NEVER repeat the same phrase, line, or citation multiple times
- If a page has many similar links, summarize them (e.g., "Found 50 subscription links") - don't list each one
- Keep responses under 2000 characters - be concise and focused
- Do NOT add "BY CANOPUS" or similar attributions to your output

WHEN MENTIONED BY ANOTHER AGENT:
- Vega may ask you to research topics on the web
- Altair may ask you to verify deployed applications
- Complete the task and report findings in the same channel
- Include screenshots as evidence when relevant
- If you need to delegate, mention the appropriate agent

HANDOFF PROTOCOL (IMPORTANT):
After completing a search or research task, consider if another agent should act on findings:

1. **Automatic Detection**: Use mark_goal_complete(handoff_to="auto") to let the system
   intelligently detect which agent(s) should receive your findings based on content.

2. **Explicit Handoff**: If you know which agent needs the info:
   - Dates/times/schedules → handoff_to="polaris" (calendar agent)
   - Code/project tasks → handoff_to="altair" (project manager)
   - General discussion → handoff_to="vega" (conversational agent)
   - Multiple agents → handoff_to="polaris,altair"

3. **When to Handoff**:
   - You found event dates, show times, or scheduling information → Polaris
   - You found technical details that need implementation → Altair
   - You have questions needing user clarification → Vega

Example: If user asks to find musical performance dates, after finding them use:
mark_goal_complete(success=True, summary="Found dates...", handoff_to="polaris")

This ensures Polaris receives the dates and can add them to the calendar.

BOUNDARIES:
- You browse the public web - respect robots.txt and ToS
- Don't enter credentials unless explicitly provided and requested
- Don't make purchases or submit sensitive forms without permission
- If a site blocks automation, report it and suggest alternatives
"""


class CanopusAgent(BaseAgent):
    """
    Canopus - The web intelligence and browser automation agent.

    Handles:
    - Web browsing and navigation
    - Page interaction (clicking, typing, scrolling)
    - Screenshot capture and visual analysis
    - Content extraction and form filling
    - Web research for other agents
    """

    @property
    def chime_in_description(self) -> str:
        """Description of Canopus's domain for chime-in evaluation."""
        return (
            "Canopus is the web browser and research specialist. "
            "Chime in when: web browsing is needed, URLs are mentioned, "
            "web searches or research tasks come up, information needs to be "
            "looked up online, screenshots of websites are needed, "
            "or web forms/interactions are discussed. "
            "Also respond when another agent needs help finding information "
            "on the web or verifying web-based resources."
        )

    # Patterns that suggest Canopus should handle
    CANOPUS_PATTERNS = [
        r"\b(browse|visit|go to|open|navigate)\s+(to\s+)?(https?://|www\.|\w+\.(com|org|net|io))",
        r"\b(search|look up|find|research)\s+(on|for|about)\b",
        r"\b(screenshot|capture|snap)\b",
        r"\b(click|press|type|fill|submit)\s+(on|the|button|form|input)",
        r"\b(extract|scrape|get)\s+(data|text|content|info)",
        r"\bweb\s*(site|page|search|research)\b",
        r"\b(check|verify|test)\s+(the\s+)?(url|website|page|link)",
    ]

    def __init__(
        self,
        llm: "LLMProvider",
        browser_manager: "BrowserSessionManager",
        conversation_manager: Optional["ConversationManager"] = None,
        memory_manager: Optional["MemoryManager"] = None,
        discord_bot: Optional[Any] = None,
        agent_registry: Optional[dict] = None,
        utility_llm: Optional["LLMProvider"] = None,
    ):
        super().__init__(
            name="Canopus",
            persona=CANOPUS_PERSONA,
            llm=llm,
            memory_manager=memory_manager,
            utility_llm=utility_llm,
        )
        self.browser_manager = browser_manager
        self.conversation_manager = conversation_manager
        self.discord_bot = discord_bot
        self.agent_registry = agent_registry or {}

        # Track current context
        self._current_channel_id: Optional[int] = None

        # Workflow state provider for context awareness
        self._workflow_state_provider: Optional[WorkflowStateProvider] = None

    def set_workflow_state_provider(self, provider: WorkflowStateProvider):
        """Set the workflow state provider for context awareness."""
        self._workflow_state_provider = provider

    def _get_agent_descriptions(self) -> dict:
        """
        Get agent descriptions for handoff evaluation.

        These descriptions help the LLM determine which agent should
        receive findings based on their expertise.
        """
        return {
            "polaris": (
                "Polaris is the scheduling and calendar specialist. "
                "Handles: dates, times, events, schedules, appointments, "
                "performance times, show dates, meeting times, deadlines, reminders, "
                "or any information that should be added to a calendar."
            ),
            "altair": (
                "Altair is the project manager and code specialist. "
                "Handles: project tasks, code issues, development work, "
                "file operations, terminal commands, and technical implementation."
            ),
            "vega": (
                "Vega is the conversational coordinator and general assistant. "
                "Handles: general questions, clarifications, coordinating between agents, "
                "and tasks that don't fit other specialists."
            ),
        }

    def get_workflow_state(self) -> Optional[WorkflowState]:
        """Get current workflow state."""
        if self._workflow_state_provider:
            return self._workflow_state_provider.get_workflow_state()
        return None

    def _register_tools(self):
        """Register Canopus's browser tools."""
        from canopus.tools.cognitive import (
            PlanNavigationTool,
            StoreFindingTool,
            ReadFindingsTool,
            CheckProgressTool,
            MarkGoalCompleteTool,
        )
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
            ClickByIndexTool,
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

        # Cognitive tools (MUST be first - used for structured navigation)
        self._tools.register(PlanNavigationTool())
        self._tools.register(StoreFindingTool())
        self._tools.register(ReadFindingsTool())
        self._tools.register(CheckProgressTool())

        # MarkGoalCompleteTool with handoff capabilities
        # Build agent descriptions for auto-handoff detection
        agent_descriptions = self._get_agent_descriptions()
        self._tools.register(MarkGoalCompleteTool(
            discord_bot=self.discord_bot,
            agent_registry=self.agent_registry,
            llm=self.llm,
            agent_descriptions=agent_descriptions,
        ))

        # Navigation tools
        self._tools.register(NavigateToTool())
        self._tools.register(GoBackTool())
        self._tools.register(GoForwardTool())
        self._tools.register(RefreshTool())
        self._tools.register(GetSnapshotTool())

        # Tab management tools
        self._tools.register(ListTabsTool())
        self._tools.register(NewTabTool())
        self._tools.register(SwitchTabTool())
        self._tools.register(CloseTabTool())

        # Interaction tools
        self._tools.register(ClickTool())
        self._tools.register(ClickByIndexTool())
        self._tools.register(TypeTextTool())
        self._tools.register(PressKeyTool())
        self._tools.register(ScrollTool())
        self._tools.register(HoverTool())
        self._tools.register(DoubleClickTool())
        self._tools.register(SelectOptionTool())

        # Extraction tools
        self._tools.register(ScreenshotTool())
        self._tools.register(ExtractTextTool())
        self._tools.register(EvaluateJsTool())
        self._tools.register(GetLinksTool())
        self._tools.register(GetPageInfoTool())

        # Form tools
        self._tools.register(FillFormTool())

        # Analysis tools
        self._tools.register(WaitForElementTool())
        self._tools.register(FindElementsTool())

        # File tools
        self._tools.register(UploadFileTool())
        self._tools.register(SetDownloadPathTool())
        self._tools.register(WaitForDownloadTool())

        # Storage tools
        self._tools.register(GetCookiesTool())
        self._tools.register(SetCookieTool())
        self._tools.register(ClearCookiesTool())
        self._tools.register(GetLocalStorageTool())
        self._tools.register(SetLocalStorageTool())
        self._tools.register(ClearStorageTool())

        # Network tools
        self._tools.register(EnableNetworkMonitoringTool())
        self._tools.register(GetNetworkLogTool())
        self._tools.register(ClearNetworkLogTool())
        self._tools.register(BlockResourcesTool())

        # Device tools
        self._tools.register(SetViewportTool())
        self._tools.register(EmulateDeviceTool())
        self._tools.register(SetGeolocationTool())
        self._tools.register(SetTimezoneTool())

        # Inter-agent communication
        if self.discord_bot and self.agent_registry:
            from vega.tools.mention import MentionAgentTool

            self._tools.register(
                MentionAgentTool(
                    bot=self.discord_bot,
                    agent_registry=self.agent_registry,
                    own_agent_name="canopus",
                )
            )
            logger.info("Registered MentionAgentTool for agent delegation")

        logger.info("Canopus tools registered")

    def get_system_prompt(self, memory_context: Optional[str] = None) -> str:
        """Build Canopus's system prompt with optional memory context."""
        tools_desc = self.get_tools_description()

        base_prompt = f"""{self.persona}

AVAILABLE TOOLS:
{tools_desc}

CURRENT STATE:
Browser session will be created automatically when you navigate.
Always get a snapshot after navigation to see page structure.
"""

        # Include workflow state for context awareness
        workflow_state = self.get_workflow_state()
        if workflow_state:
            base_prompt += workflow_state.format_for_llm()

        if memory_context:
            base_prompt += f"""

## YOUR MEMORIES:
{memory_context}

When you learn something important about websites or research findings, use store_memory to remember it.
"""

        return base_prompt

    @property
    def system_prompt(self) -> str:
        """Build Canopus's system prompt."""
        return self.get_system_prompt()

    def should_handle(self, context: AgentContext) -> float:
        """
        Calculate confidence that Canopus should handle this message.

        Returns:
            0.0 - 1.0 confidence score
        """
        content = context.message_content.lower()
        score = 0.2  # Base score

        # Explicit mention
        if context.mentioned_agent == "canopus":
            return 1.0

        # Check for Canopus patterns (web/browser related)
        canopus_matches = sum(
            1
            for pattern in self.CANOPUS_PATTERNS
            if re.search(pattern, content, re.IGNORECASE)
        )

        # Adjust score based on pattern matches
        score += canopus_matches * 0.25

        # Check for URLs in message
        if re.search(r"https?://\S+", content):
            score += 0.3

        # Check for web-related keywords
        web_keywords = ["website", "webpage", "browser", "url", "link", "online"]
        keyword_matches = sum(1 for kw in web_keywords if kw in content)
        score += keyword_matches * 0.1

        return max(0.0, min(1.0, score))

    async def process(self, context: AgentContext) -> AgentResponse:
        """
        Process a message and perform web operations.

        Uses browser tools to navigate, interact, and extract data.
        Uses cognitive tools for structured navigation with planning and progress tracking.
        """
        from shared.tools.interface import ToolContext

        start_time = time.time()
        tool_calls_made = 0

        # Store current channel for browser session lookup
        self._current_channel_id = context.channel_id

        # Create request-scoped scratchpad for cognitive tools
        scratchpad = NavigationScratchpad()

        try:
            # Build memory context
            memory_context_str = await self.build_memory_context(context.message_content)

            # Build conversation messages
            messages = await self._build_messages(context, memory_context_str)

            # Get tool definitions
            tool_defs = self.tools.get_definitions()

            # Create tool context with browser manager and scratchpad
            tool_context = ToolContext(
                browser_manager=self.browser_manager,
                conversation_manager=self.conversation_manager,
                memory_manager=self.memory_manager,
                discord_bot=self.discord_bot,
                scratchpad=scratchpad,  # Add scratchpad for cognitive tools
                current_channel_id=context.channel_id,
                current_guild_id=context.guild_id,
                user_id=context.user_id,
                current_agent_id="canopus",
                conversation_id=context.conversation_id,
            )

            # Run agent loop
            max_iterations = 15  # Limit tool call iterations

            for iteration in range(max_iterations):
                logger.debug(
                    f"[Canopus] ITERATION {iteration + 1}/{max_iterations}: "
                    f"tools_so_far={tool_calls_made}, messages={len(messages)}, "
                    f"goal_achieved={scratchpad.goal_achieved}"
                )

                # Check if goal was achieved (early exit)
                if scratchpad.goal_achieved:
                    processing_time = int((time.time() - start_time) * 1000)
                    logger.info(
                        f"[Canopus] GOAL ACHIEVED: iteration={iteration + 1}, "
                        f"tool_calls={tool_calls_made}, reason={scratchpad.completion_reason}"
                    )

                    # Build success response with findings
                    content = f"✅ {scratchpad.completion_reason}"
                    if scratchpad.findings:
                        content += "\n\n**Findings:**"
                        for f in scratchpad.findings:
                            content += f"\n• {f.key}: {f.value}"

                    # Store in conversation history
                    if self.conversation_manager and context.conversation_id:
                        await self.conversation_manager.add_message(
                            context.conversation_id,
                            Role.ASSISTANT.value,
                            content,
                        )

                    return AgentResponse(
                        content=content,
                        agent_name=self.name,
                        tool_calls_made=tool_calls_made,
                        processing_time_ms=processing_time,
                    )

                # Get LLM response
                response = await self.llm.complete(
                    messages=messages,
                    tools=tool_defs if tool_defs else None,
                )

                # If no tool calls, we have our final response
                if not response.tool_calls:
                    processing_time = int((time.time() - start_time) * 1000)

                    logger.info(
                        f"[Canopus] FINAL RESPONSE: iteration={iteration + 1}, "
                        f"tool_calls={tool_calls_made}, has_content={bool(response.content)}, "
                        f"content_len={len(response.content or '')}"
                    )

                    # Store in conversation history
                    if self.conversation_manager and context.conversation_id:
                        await self.conversation_manager.add_message(
                            context.conversation_id,
                            Role.ASSISTANT.value,
                            response.content or "",
                        )

                    return AgentResponse(
                        content=response.content or "Web operation completed.",
                        agent_name=self.name,
                        tool_calls_made=tool_calls_made,
                        processing_time_ms=processing_time,
                    )

                # Execute tool calls
                logger.debug(
                    f"[Canopus] TOOL CALLS: iteration={iteration + 1}, "
                    f"count={len(response.tool_calls)}, "
                    f"tools={[tc.name for tc in response.tool_calls]}"
                )
                tool_calls_made += len(response.tool_calls)
                results = await self.tools.execute_batch(
                    response.tool_calls, tool_context
                )

                # Add to messages for next iteration
                messages.append(
                    Message(
                        role=Role.ASSISTANT,
                        tool_calls=response.tool_calls,
                        raw_content=response.raw_content,
                    )
                )
                messages.append(
                    Message(
                        role=Role.TOOL,
                        tool_results=results,
                    )
                )

            # Reached max iterations - summarize
            messages.append(
                Message(
                    role=Role.USER,
                    content=(
                        "You've made many tool calls. Please summarize:\n"
                        "1. What was originally requested\n"
                        "2. What you found or accomplished\n"
                        "3. Key URLs and screenshots taken\n"
                        "4. Any issues encountered\n"
                        "Do NOT make any more tool calls."
                    ),
                )
            )

            summary_response = await self.llm.complete(
                messages=messages,
                tools=None,
            )

            processing_time = int((time.time() - start_time) * 1000)
            return AgentResponse(
                content=summary_response.content or "Web operation completed.",
                agent_name=self.name,
                tool_calls_made=tool_calls_made,
                processing_time_ms=processing_time,
                status="partial",
            )

        except Exception as e:
            logger.error(f"Canopus processing error: {e}", exc_info=True)
            processing_time = int((time.time() - start_time) * 1000)
            return AgentResponse(
                content=f"I encountered an error while browsing: {str(e)}",
                agent_name=self.name,
                tool_calls_made=tool_calls_made,
                processing_time_ms=processing_time,
                status="error",
                error=str(e),
            )

    async def _build_messages(
        self, context: AgentContext, memory_context: Optional[str] = None
    ) -> List[Message]:
        """Build message history for LLM."""
        system_prompt = self.get_system_prompt(memory_context)
        messages = [Message(role=Role.SYSTEM, content=system_prompt)]

        # Fetch recent Discord channel history
        if self.discord_bot and context.channel_id:
            history = await self._fetch_discord_history(context.channel_id, limit=10)
            logger.info(f"Canopus loaded {len(history)} messages from Discord history")
            messages.extend(history)

        # Add current user message
        user_name = f"User {context.user_id}"
        if self.discord_bot:
            try:
                user = await self.discord_bot.fetch_user(context.user_id)
                user_name = user.display_name or user.name
            except Exception:
                pass

        messages.append(
            Message(role=Role.USER, content=f"[{user_name}]: {context.message_content}")
        )

        return messages

    async def _fetch_discord_history(
        self, channel_id: int, limit: int = 10
    ) -> List[Message]:
        """Fetch recent messages from Discord channel."""
        if not self.discord_bot:
            return []

        try:
            channel = self.discord_bot.get_channel(channel_id)
            if not channel:
                return []

            history_messages = []
            async for msg in channel.history(limit=limit + 1):
                if not msg.content or not msg.content.strip():
                    continue

                content = msg.content
                if content.startswith("**") and ":** " in content[:30]:
                    prefix_end = content.find(":** ") + 4
                    content = content[prefix_end:]

                if msg.author.bot:
                    role = Role.ASSISTANT
                    formatted = f"[{msg.author.display_name}]: {content}"
                else:
                    role = Role.USER
                    formatted = f"[{msg.author.display_name}]: {content}"

                history_messages.append(Message(role=role, content=formatted))

            history_messages.reverse()
            if history_messages:
                history_messages = history_messages[:-1]

            return history_messages

        except Exception as e:
            logger.warning(f"Failed to fetch Discord history: {e}")
            return []

    async def get_status_summary(self) -> str:
        """Get a summary of Canopus's current status."""
        sessions = self.browser_manager.list_sessions()

        if not sessions:
            return "Canopus is idle. No active browser sessions."

        summary_parts = [f"Active browser sessions: {len(sessions)}"]
        for session in sessions[:3]:  # Show up to 3
            url = session.get("current_url", "about:blank")
            if len(url) > 50:
                url = url[:50] + "..."
            summary_parts.append(f"  - {session['session_id']}: {url}")

        return "Canopus status: " + " | ".join(summary_parts)

    # =========================================================================
    # Multi-Agent Coordination Helpers
    # =========================================================================

    def format_research_report(
        self,
        topic: str,
        findings: List[dict],
        screenshots: Optional[List[str]] = None,
    ) -> str:
        """
        Format research findings for other agents (Vega, Altair).

        Args:
            topic: Research topic/question
            findings: List of findings with 'url', 'title', 'summary' keys
            screenshots: Optional list of screenshot paths taken

        Returns:
            Formatted report string suitable for agent handoff
        """
        lines = [
            f"📋 **Web Research Report: {topic}**",
            "",
        ]

        if findings:
            lines.append("**Findings:**")
            for i, finding in enumerate(findings, 1):
                lines.append(f"{i}. **{finding.get('title', 'Untitled')}**")
                if finding.get('url'):
                    lines.append(f"   URL: {finding['url']}")
                if finding.get('summary'):
                    lines.append(f"   {finding['summary']}")
                lines.append("")
        else:
            lines.append("No relevant findings.")

        if screenshots:
            lines.append(f"**Screenshots:** {len(screenshots)} captured")

        return "\n".join(lines)

    def format_verification_result(
        self,
        url: str,
        checks: List[dict],
        passed: bool,
        screenshots: Optional[List[str]] = None,
    ) -> str:
        """
        Format verification results for Altair (deployment verification).

        Args:
            url: URL that was verified
            checks: List of checks with 'name', 'passed', 'details' keys
            passed: Overall pass/fail
            screenshots: Optional list of screenshot paths

        Returns:
            Formatted verification report
        """
        status = "✅ PASSED" if passed else "❌ FAILED"
        lines = [
            f"🔍 **Site Verification: {status}**",
            f"URL: {url}",
            "",
            "**Checks:**",
        ]

        for check in checks:
            check_status = "✅" if check.get('passed') else "❌"
            lines.append(f"  {check_status} {check.get('name', 'Unknown')}")
            if check.get('details'):
                lines.append(f"     {check['details']}")

        if screenshots:
            lines.append(f"\n**Evidence:** {len(screenshots)} screenshots captured")

        return "\n".join(lines)

    def format_extraction_result(
        self,
        url: str,
        data_type: str,
        data: Any,
        record_count: Optional[int] = None,
    ) -> str:
        """
        Format data extraction results for other agents.

        Args:
            url: Source URL
            data_type: Type of data extracted (links, text, form data, etc.)
            data: The extracted data
            record_count: Optional count of records

        Returns:
            Formatted extraction report
        """
        lines = [
            f"📦 **Data Extraction: {data_type}**",
            f"Source: {url}",
        ]

        if record_count is not None:
            lines.append(f"Records: {record_count}")

        lines.append("")

        # Format data based on type
        if isinstance(data, list):
            lines.append("**Data:**")
            for i, item in enumerate(data[:10], 1):
                if isinstance(item, dict):
                    item_str = ", ".join(f"{k}: {v}" for k, v in list(item.items())[:3])
                    lines.append(f"  {i}. {item_str}")
                else:
                    lines.append(f"  {i}. {str(item)[:100]}")
            if len(data) > 10:
                lines.append(f"  ... and {len(data) - 10} more")
        elif isinstance(data, dict):
            lines.append("**Data:**")
            for key, value in list(data.items())[:10]:
                lines.append(f"  {key}: {str(value)[:100]}")
        else:
            lines.append(f"**Data:** {str(data)[:500]}")

        return "\n".join(lines)

    async def create_session_snapshot(self, channel_id: int) -> Optional[dict]:
        """
        Create a snapshot of the current session state for handoff.

        Returns:
            Dict with session state info or None if no session
        """
        session = await self.browser_manager.get(channel_id)
        if not session:
            return None

        try:
            page_info = await session.get_page_info()
            tabs = await session.list_tabs()

            return {
                "session_id": session.session_id,
                "current_url": session.current_url,
                "page_info": page_info if page_info.get("success") else None,
                "tabs": tabs,
                "history": session.history[-10:],  # Last 10 URLs
                "viewport": {
                    "width": session.viewport_width,
                    "height": session.viewport_height,
                },
                "created_at": session.created_at.isoformat(),
                "last_activity": session.last_activity.isoformat(),
            }
        except Exception as e:
            logger.warning(f"Failed to create session snapshot: {e}")
            return None
