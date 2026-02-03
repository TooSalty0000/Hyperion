"""Altair - The autonomous project manager agent."""

import re
import time
import logging
from datetime import datetime, timezone
from typing import Any, Optional, List, TYPE_CHECKING

from shared.base_agent import BaseAgent, AgentContext, AgentResponse
from shared.llm.types import Message, Role
from shared.workflow_state import WorkflowStateProvider, WorkflowState

if TYPE_CHECKING:
    from shared.llm.interface import LLMProvider
    from shared.session import SessionRegistry
    from shared.project import ProjectManager
    from shared.conversation import ConversationManager
    from shared.memory import MemoryManager
    from shared.soul import SoulManager
    from shared.project.status import ProjectStatusTracker
    from shared.tools.interface import ToolContext

logger = logging.getLogger(__name__)


# Altair's persona - methodical, precise, autonomous execution
ALTAIR_PERSONA = """=== IDENTITY: YOU ARE ALTAIR ===
Your name is ALTAIR. You are ONLY Altair. You are NEVER any other agent.
- You are NOT Vega (the conversational agent)
- You are NOT Polaris (the calendar agent)
- You are NOT Canopus (the browser agent)
NEVER introduce yourself as or respond as any other agent. ALWAYS be Altair.

PERSONALITY:
- Methodical and precise in your work
- Calm under pressure, even when things go wrong
- Status-focused - you always know what's happening
- Proactive - you take action, don't wait for permission
- Take pride in craftsmanship and getting things done right
- Concise in communication - focus on what matters

UNDERSTANDING CHAT HISTORY:
In the chat history, you'll see messages formatted as "[Name]: message".
- "[Altair]: ..." = YOUR previous messages (you said this)
- "[Vega]: ..." = Vega's messages (a DIFFERENT agent said this)
- "[Polaris]: ..." = Polaris's messages (a DIFFERENT agent said this)
- "[Canopus]: ..." = Canopus's messages (a DIFFERENT agent said this)
- "[Username]: ..." = User messages

CRITICAL: When you see "[Vega]: some message" in history, that was VEGA speaking, NOT YOU. Do not confuse yourself with other agents based on chat history.

RESPONSE FORMAT:
- NEVER prefix your responses with your name (e.g., "[Altair]:" or "Altair:")
- NEVER prefix messages with another agent's name like "Vega:" or "[Vega]:"
- Discord already shows who is speaking - adding a name prefix is redundant
- The [Name]: format in chat history is just for context, don't copy it
- Just respond directly with your message content

ROLE:
- You are ALWAYS ON STANDBY - ready to manage projects at any moment
- You are the AUTONOMOUS manager of ALL terminal sessions and CLI agent instances
- Your PRIMARY tool is Claude Code (claude) - an AI-powered CLI for coding tasks
- You can also use: gemini cli, codex, or other agentic CLI tools
- You CREATE sessions when needed - you don't ask users to do it
- You monitor project status and terminal output across all sessions
- You do NOT write code directly - you instruct CLI tools like Claude Code to do the work

IMPORTANT - PERMISSION SYSTEM:
- Before running CLI commands or making changes, you MUST request permission
- The permission system will ask the user for approval via Discord reactions
- Wait for approval before executing sensitive actions
- If rejected or timeout, explain what you wanted to do and ask for guidance

SESSION MANAGEMENT:
- You have FULL CONTROL over terminal sessions
- Use create_session to spin up new sessions - this creates BOTH a tmux session AND a Discord channel
- Use the 'project' parameter to specify a project name (e.g., create_session(project="myapp"))
- Use list_projects to see available projects and their workspace directories
- Use switch_session to manage multiple concurrent sessions
- Each project can have its own session with Claude Code running

SESSION NOTES (IMPORTANT):
- Use set_session_note to annotate what each session is for
- Use get_session_note to check a session's purpose
- list_active_sessions shows all sessions with their notes
- ALWAYS set a note when creating a session (e.g., "Building React dashboard for Hyperion")
- BEFORE creating a new session, check list_active_sessions to see if one already exists for your task
- This prevents duplicate sessions for the same work

IMPORTANT - DO NOT TERMINATE SESSIONS:
- NEVER call terminate_session unless the USER explicitly asks you to close/terminate/delete a session
- Sessions should persist so work can continue later
- Just because a task is "done" does NOT mean you should terminate the session
- Leave sessions running - they are a resource for continued work

HOW TO USE CLAUDE CODE:
1. Check list_active_sessions for existing sessions
2. Check list_projects to see available projects
3. If no session exists for the project, use create_session to create one:
   - Use project="project_name" to start in the project's workspace
   - Use start_claude_code=True to auto-start Claude Code
   Example: create_session(project="vega", start_claude_code=True)
4. Use send_cli_command to send instructions - Enter key is added automatically
   Example: send_cli_command("implement a function that calculates fibonacci numbers")
5. **ACTIVELY MONITOR** using monitor_session to track progress:
   - Call monitor_session repeatedly to see what's happening
   - It detects: WORKING, WAITING (permission prompts), IDLE, COMPLETED, ERROR
   - Report status to the user so they know what's happening!
6. If monitor_session shows WAITING (permission prompt detected):
   - Tell the user what Claude Code is asking
   - Use answer_terminal_prompt("yes") to approve, or "no" to reject
7. Use get_cli_output to see detailed output when needed

MONITORING WORKFLOW (CRITICAL):
- DON'T just wait passively - actively monitor and REPORT to the user!
- Use monitor_session every 10-30 seconds while work is in progress
- When you see WAITING state, handle the permission prompt immediately
- Give the user updates: "Claude is analyzing...", "Claude is writing code...", etc.
- If IDLE too long, investigate what's happening

IMPORTANT - TOOL USAGE:
- answer_terminal_prompt: ONLY for CLI/terminal prompts (like Claude Code asking "proceed? [Y/n]")
  - This sends keystrokes to the TERMINAL SESSION, not to Discord
  - Do NOT use this to respond to users - that would send your text to the terminal!
- To reply to users in Discord: Just write your response as normal text (no tool needed)
- The answer_terminal_prompt tool is ONLY for when monitor_session detects a terminal prompt

TASK CONTROL:
- abort_task: Use this when you realize you CAN'T proceed without more information
  - Example: "I need to know which file you want me to modify"
  - Example: "Should I use TypeScript or JavaScript for this?"
  - This STOPS your current task and asks the user for clarification
- request_clarification: Use for quick yes/no questions while working
  - Example: "Should I also update the tests?"
- DON'T just guess when you're unsure - ASK using these tools!

FILESYSTEM NAVIGATION (read-only):
- Use list_directory to explore folders (like ls)
- Use read_file to examine code, configs, or docs
- Use find_files to search for files by pattern (e.g., '*.py', 'src/**/*.tsx')
- Use file_info for detailed info about a file/directory
- These tools are READ-ONLY - they don't modify anything
- For MODIFYING files, use Claude Code via a session - that's the proper workflow
- Use navigation tools to understand project structure BEFORE delegating to Claude Code

TUNNEL MANAGEMENT (ngrok):
- When you start a dev server (e.g., npm run dev on port 3000), use start_tunnel to expose it
- The user may be REMOTE and can't access localhost - tunnels give them a public URL
- start_tunnel(port=3000) returns a public URL like https://abc123.ngrok.io
- Always share the tunnel URL with the user when you start a dev server
- Use stop_tunnel when the server is no longer needed
- Use list_tunnels to see all active tunnels
- PROACTIVELY create tunnels when starting any web server, preview, or dev environment

COMMUNICATION STYLE:
- Be direct and to the point
- Lead with the ANSWER to what was asked, then provide context
- Use clear, actionable language
- Report issues immediately and suggest solutions
- When asked for status, give concrete progress with specifics
- When asked for data, INCLUDE THE DATA in your response
- Your response should be self-contained - reader shouldn't need to look elsewhere

AUTONOMOUS WORKFLOW:
1. Receive a task from the user or another agent (via @mention)
   - IMPORTANT: Note exactly what information/data was requested
2. Check list_active_sessions for existing sessions
3. If user mentions a project, check list_projects to find it
4. If no suitable session exists, create one:
   - create_session(project="project_name", start_claude_code=True)
   This creates a Discord channel AND starts Claude Code in one step!
5. Use send_cli_command to instruct Claude Code what to do
6. **MONITOR ACTIVELY** in a loop:
   a. Call monitor_session to check status
   b. If WORKING: report progress to user, wait a bit, repeat
   c. If WAITING: handle the permission prompt (approve/reject)
   d. If COMPLETED: move to step 7
   e. If ERROR: report error to user
   f. If IDLE too long: investigate
7. Use get_cli_output to review the results and EXTRACT the relevant data
8. **REPORT RESULTS** - This is CRITICAL:
   - Review what was originally asked
   - Include the ACTUAL DATA/FINDINGS in your response
   - Don't just say "task complete" - provide the information requested
   - Format the results clearly for the user/agent who asked

WHEN MENTIONED BY ANOTHER AGENT:
- You are mentioned via @Altair in Discord
- Read the request and REMEMBER what data/information was requested
- At the END of your work, you MUST report findings back in your response
- If Vega asked "what is the project status?" - your final response MUST include the status
- If asked to analyze something - summarize your findings in your response
- Don't just say "done" - ALWAYS include the requested information in your reply
- Include ALL findings in your response text. Vega handles all inter-agent coordination.
- Do NOT try to @mention other agents - just respond with your results.

CRITICAL - REPORTING RESULTS:
- Your final response MUST answer the original question with actual data
- Example: If asked "what files are in the project?" - list the files you found
- Example: If asked "check for errors" - report what errors (or no errors) you found
- Example: If asked for status - give concrete status with details
- NEVER finish without reporting the information that was requested
- Think: "What did they ask for? Did I provide that information?"

BOUNDARIES:
- You work through CLI tools like Claude Code - don't try to code directly
- You CREATE and MANAGE terminal sessions autonomously
- You ALWAYS request permission before executing commands
- If you encounter errors, report them clearly with context

IF YOU RECEIVE A WRONG TASK:
- If the user asks you something that's NOT about project execution, CLI work, or session management
- Examples: "explain this concept", "what do you think about X", general conversation
- You should respond directly if you can answer briefly
- OR say "That's more of a conversation question - Vega can help with that"
- You are the HANDS-ON agent - Vega is the CONVERSATIONAL agent
- Don't refuse to help - either answer briefly or note it's outside your domain

WHEN TO NOT REPLY (CRITICAL - READ CAREFULLY):
Sometimes the best response is NO response. Do NOT reply when:
- The message is casual chat between others (e.g., "hi everyone", "thanks!", "lol")
- Another agent has ALREADY handled the request adequately
- The conversation has clearly moved on to a different topic
- You're only tangentially mentioned in a general greeting
- The message is just social pleasantries not requiring your expertise
- Someone is asking another agent a question (not you)
- Someone is giving you INSTRUCTIONS or GUIDELINES (not a task)
- Vega is telling agents what to do/not do - that's a directive, not a task

If you're uncertain whether to reply, consider:
1. Was I directly asked to DO something specific with a concrete deliverable?
2. Does this require my CLI/project expertise to produce something?
3. Has someone else already addressed this?
4. Would my response add VALUE or just add NOISE?

=== HOW TO NOT REPLY (THIS IS CRUCIAL) ===
When you decide NOT to reply, you must produce LITERALLY EMPTY OUTPUT.
That means: NO TEXT AT ALL. Not even a single character.

FORBIDDEN RESPONSES (NEVER SAY THESE):
- "Done." / "Done" / "Done!"
- "Task completed." / "Task completed"
- "Acknowledged." / "Acknowledged"
- "Understood." / "Understood"
- "Copy that." / "Copy that"
- "Ready." / "Ready"
- "Got it." / "Got it"
- "Noted." / "Noted"
- "Roger." / "Roger"
- "Affirmative." / "Affirmative"
- "On it." / "On it"
- "Will do." / "Will do"
- Any single-word or short acknowledgment

These are NOISE. They trigger other agents and create feedback loops.
If you don't have real work to report, say NOTHING. Literal silence.
"""


class AltairAgent(BaseAgent):
    """
    Altair - The autonomous project manager agent.

    Handles:
    - Hands-on project execution via CLI tools
    - Sending commands to claude code, gemini, etc.
    - Monitoring terminal output and status
    - Reporting progress and issues
    - Permission requests before sensitive actions
    """

    @property
    def chime_in_description(self) -> str:
        """Description of Altair's domain for chime-in evaluation."""
        return (
            "Altair is the project manager and code specialist. "
            "Chime in when: project tasks need execution, code issues arise, "
            "development work is mentioned, file operations are needed, "
            "terminal commands should be run, or technical implementation is discussed. "
            "Also respond to discussions about build status, deployments, "
            "git operations, or when another agent found code-related issues to fix."
        )

    # Patterns that suggest Altair should handle
    ALTAIR_PATTERNS = [
        r"\b(run|execute|start|build|deploy|test|implement)\b",
        r"\b(create|write|edit|modify|fix|update)\s+(file|code|function|class|feature)\b",
        r"\bgit\s+(commit|push|pull|merge|checkout)\b",
        r"\b(send|tell|ask)\s+(claude|gemini|gpt|the cli)\b",
        r"\bwork\s+on\b",
        r"\bimplement\b",
        r"\bfix\s+the\b",
    ]

    # Patterns that signal the user wants to stop/abort current work
    STOP_PATTERNS = [
        r"^stop\b",
        r"^wait\b",
        r"^hold\b",
        r"^cancel\b",
        r"^abort\b",
        r"^pause\b",
        r"^nevermind\b",
        r"^never\s*mind\b",
        r"\bstop\s+that\b",
        r"\bcancel\s+that\b",
        r"\bhold\s+on\b",
        r"\bwait\s+a\s+(sec|minute|moment)\b",
    ]

    def __init__(
        self,
        llm: "LLMProvider",
        session_registry: "SessionRegistry",
        project_manager: Optional["ProjectManager"] = None,
        conversation_manager: Optional["ConversationManager"] = None,
        memory_manager: Optional["MemoryManager"] = None,
        soul_manager: Optional["SoulManager"] = None,
        status_tracker: Optional["ProjectStatusTracker"] = None,
        discord_bot: Optional[Any] = None,
        channel_manager: Optional[Any] = None,
        agent_registry: Optional[dict] = None,
        utility_llm: Optional["LLMProvider"] = None,
    ):
        super().__init__(
            name="Altair",
            persona=ALTAIR_PERSONA,
            llm=llm,
            memory_manager=memory_manager,
            soul_manager=soul_manager,
            utility_llm=utility_llm,
        )
        self.session_registry = session_registry
        self.project_manager = project_manager
        self.conversation_manager = conversation_manager
        self.status_tracker = status_tracker
        self.discord_bot = discord_bot
        self.channel_manager = channel_manager
        self.agent_registry = agent_registry or {}

        # Force early tool registration (avoid lazy loading during process())
        _ = self.tools

        # Track current working context
        self._current_session_id: Optional[int] = None
        self._current_project: Optional[str] = None
        self._current_task: Optional[str] = None

        # Permission manager - set by the bot
        self._permission_manager = None

        # Callback for starting output loops (set by the cog)
        self._start_output_loop_callback = None

        # Workflow state provider - provides context about current work
        self._workflow_state_provider: Optional[WorkflowStateProvider] = None

    def set_permission_manager(self, pm):
        """Set the permission manager for this agent."""
        self._permission_manager = pm

    def set_output_loop_callback(self, callback):
        """Set callback to start output loop for new sessions."""
        self._start_output_loop_callback = callback

    def set_workflow_state_provider(self, provider: WorkflowStateProvider):
        """Set the workflow state provider for context awareness."""
        self._workflow_state_provider = provider

    def get_workflow_state(self) -> Optional[WorkflowState]:
        """Get current workflow state."""
        if self._workflow_state_provider:
            return self._workflow_state_provider.get_workflow_state()
        return None

    def set_session(self, session_id: int):
        """Set the current working session."""
        self._current_session_id = session_id

    def set_project(self, project_name: str):
        """Set the current working project."""
        self._current_project = project_name

    def _register_tools(self):
        """Register Altair's tools."""
        # Import from altair's own tools module
        from altair.tools.cli_commands import (
            StartClaudeCodeTool,
            SendCLICommandTool,
            GetCLIOutputTool,
            WaitForCompletionTool,
            GetSessionStatusTool,
            ListActiveSessionsTool,
        )
        from altair.tools.session import (
            CreateSessionTool,
            TerminateSessionTool,
            SwitchSessionTool,
            ListProjectsTool,
            SetSessionNoteTool,
            GetSessionNoteTool,
        )
        from altair.tools.monitoring import (
            MonitorSessionTool,
            CheckForPromptTool,
            AnswerTerminalPromptTool,
        )
        from altair.tools.control import (
            AbortTaskTool,
            RequestClarificationTool,
        )
        from altair.tools.tunnel import (
            StartTunnelTool,
            StopTunnelTool,
            ListTunnelsTool,
        )
        from altair.tools.navigation import (
            ListDirectoryTool,
            GetCurrentDirectoryTool,
            ReadFileTool,
            FindFilesTool,
            FileInfoTool,
        )
        from shared.memory.tools import get_memory_tools

        # Session management tools
        self._tools.register(CreateSessionTool())
        self._tools.register(TerminateSessionTool())
        self._tools.register(SwitchSessionTool())
        self._tools.register(ListProjectsTool())
        self._tools.register(SetSessionNoteTool())
        self._tools.register(GetSessionNoteTool())

        # CLI tools
        self._tools.register(StartClaudeCodeTool())
        self._tools.register(SendCLICommandTool())
        self._tools.register(GetCLIOutputTool())
        self._tools.register(WaitForCompletionTool())
        self._tools.register(GetSessionStatusTool())
        self._tools.register(ListActiveSessionsTool())

        # Active monitoring tools
        self._tools.register(MonitorSessionTool())
        self._tools.register(CheckForPromptTool())
        self._tools.register(AnswerTerminalPromptTool())

        # Task control tools (abort, clarification)
        self._tools.register(AbortTaskTool())
        self._tools.register(RequestClarificationTool())

        # Tunnel tools (ngrok for exposing dev servers remotely)
        self._tools.register(StartTunnelTool())
        self._tools.register(StopTunnelTool())
        self._tools.register(ListTunnelsTool())

        # Navigation tools (read-only filesystem exploration)
        self._tools.register(ListDirectoryTool())
        self._tools.register(GetCurrentDirectoryTool())
        self._tools.register(ReadFileTool())
        self._tools.register(FindFilesTool())
        self._tools.register(FileInfoTool())

        # Register memory tools if memory manager is available
        if self.memory_manager:
            for tool in get_memory_tools(self.memory_manager):
                self._tools.register(tool)
            logger.info("Registered memory tools for Altair")
        else:
            logger.info("Memory manager not configured - memory tools unavailable")

        # Register soul tools if soul manager is available
        if self.soul_manager:
            from shared.soul.tools import get_soul_tools
            for tool in get_soul_tools(self.soul_manager):
                self._tools.register(tool)
            logger.info("Registered soul tools for Altair")
        else:
            logger.info("Soul manager not configured - soul tools unavailable")

    def get_system_prompt(
        self,
        memory_context: Optional[str] = None,
        soul_context: Optional[str] = None,
    ) -> str:
        """
        Build Altair's system prompt with optional memory and soul context.

        Args:
            memory_context: Pre-built memory context string to include
            soul_context: Pre-built soul context string (injected before memory)
        """
        tools_desc = self.get_tools_description()

        context_info = ""
        if self._current_session_id:
            context_info += f"\nCurrent Session: #{self._current_session_id}"
        if self._current_project:
            context_info += f"\nCurrent Project: {self._current_project}"

        base_prompt = f"""{self.persona}

AVAILABLE TOOLS:
{tools_desc}

CURRENT CONTEXT:
{context_info if context_info else "No active session or project set."}

Use the CLI tools to perform work. Always check session status before
sending commands. Monitor output after commands to verify completion."""

        # Include workflow state for context awareness
        workflow_state = self.get_workflow_state()
        if workflow_state:
            base_prompt += workflow_state.format_for_llm()

        # Soul context comes BEFORE memory (who you are > what you know)
        if soul_context:
            base_prompt += f"""

## YOUR SOUL (Who You Are):
{soul_context}

Your personality evolves through experience. Use introspect_soul to reflect on yourself.
Use assess_skill after task outcomes to calibrate your confidence."""

        if memory_context:
            base_prompt += f"""

## YOUR MEMORIES (What You Know):
{memory_context}

When you learn something important about projects, workflows, or technical details, use store_memory to remember it.
When you need to recall past information, check your active context above or use search_memories."""

        return base_prompt

    @property
    def system_prompt(self) -> str:
        """Build Altair's system prompt (without memory context - use get_system_prompt for full version)."""
        return self.get_system_prompt()

    def should_handle(self, context: AgentContext) -> float:
        """
        Calculate confidence that Altair should handle this message.

        Returns:
            0.0 - 1.0 confidence score
        """
        content = context.message_content.lower()
        score = 0.3  # Base score

        # Explicit mention
        if context.mentioned_agent == "altair":
            return 1.0

        # Check for Altair patterns (action commands)
        altair_matches = sum(
            1
            for pattern in self.ALTAIR_PATTERNS
            if re.search(pattern, content, re.IGNORECASE)
        )

        # Adjust score based on pattern matches
        score += altair_matches * 0.2

        # Boost if we have an active session context
        if self._current_session_id:
            score += 0.1

        return max(0.0, min(1.0, score))

    async def process(self, context: AgentContext) -> AgentResponse:
        """
        Process a message and execute work.

        Uses CLI tools to perform project work.
        Checks for new messages during long-running operations to stay
        responsive to user input and other agent responses.
        """
        from shared.tools.interface import ToolContext

        start_time = time.time()

        processing_started = datetime.now(timezone.utc)
        tool_calls_made = 0
        last_message_check = 0  # Track loops since last message check
        seen_message_ids: set[int] = set()  # Track messages we've already processed

        try:
            # Build soul context (who you are - injected first)
            soul_context_str = await self.build_soul_context()

            # Build memory context (what you know)
            memory_context_str = await self.build_memory_context(context.message_content)

            # Build conversation for LLM with soul and memory context
            messages = await self._build_messages(context, memory_context_str, soul_context_str)

            # Get tool definitions
            tool_defs = self.tools.get_definitions()

            # Create tool context for execution
            tool_context = ToolContext(
                session_registry=self.session_registry,
                project_manager=self.project_manager,
                channel_manager=self.channel_manager,
                conversation_manager=self.conversation_manager,
                memory_manager=self.memory_manager,
                discord_bot=self.discord_bot,
                permission_manager=self._permission_manager,
                current_session_id=self._current_session_id,
                current_channel_id=context.channel_id,
                current_guild_id=context.guild_id,
                user_id=context.user_id,
                current_agent_id="altair",
                conversation_id=context.conversation_id,
            )
            # Attach callback for output loop
            tool_context.start_output_loop_callback = self._start_output_loop_callback

            # Run agent loop
            # Monitoring tools don't count against iteration limit - they're just waiting/watching
            MONITORING_TOOLS = {
                "monitor_session",
                "check_for_prompt",
                "wait_for_completion",
                "get_cli_output",
                "get_session_status",
                "list_active_sessions",
                "list_projects",
            }
            max_action_iterations = 15  # Limit on actual action tool calls
            max_total_loops = (
                100  # Safety limit on total loops (prevents infinite monitoring)
            )
            action_iterations = 0
            total_loops = 0

            while (
                action_iterations < max_action_iterations
                and total_loops < max_total_loops
            ):
                total_loops += 1
                last_message_check += 1

                # Check for new messages every 3 loops (to stay responsive during long operations)
                if last_message_check >= 3:
                    last_message_check = 0
                    new_msgs = await self._fetch_new_messages_since(
                        context.channel_id, processing_started, seen_message_ids
                    )
                    if new_msgs:
                        logger.info(f"Altair detected {len(new_msgs)} new message(s) during processing")
                        # Include workflow state so LLM knows what it's currently doing
                        workflow_state = self.get_workflow_state()
                        state_context = ""
                        if workflow_state:
                            state_context = f"\n\nCurrent workflow state:\n- Task: {workflow_state.current_task}\n- Running for: {workflow_state.current_task_elapsed}s"
                            if workflow_state.current_task_tools_used:
                                state_context += f"\n- Tools used: {', '.join(workflow_state.current_task_tools_used[-3:])}"

                        messages.append(
                            Message(
                                role=Role.SYSTEM,
                                content=f"[NEW MESSAGES RECEIVED]{state_context}\n\nReview these new messages and decide how to proceed. If they relate to your current work, incorporate or acknowledge. If duplicate, note that you're already on it."
                            )
                        )
                        messages.extend(new_msgs)

                # Get LLM response
                response = await self.llm.complete(
                    messages=messages,
                    tools=tool_defs if tool_defs else None,
                )

                # If no tool calls, we have our final response
                if not response.tool_calls:
                    processing_time = int((time.time() - start_time) * 1000)

                    # Store in conversation history
                    if self.conversation_manager and context.conversation_id:
                        await self.conversation_manager.add_message(
                            context.conversation_id,
                            Role.ASSISTANT.value,
                            response.content or "",
                        )

                    return AgentResponse(
                        content=response.content or "Task completed.",
                        agent_name=self.name,
                        tool_calls_made=tool_calls_made,
                        processing_time_ms=processing_time,
                    )

                # Execute tool calls with abort checking
                tool_calls_made += len(response.tool_calls)

                # Create abort check callback that looks for stop signals
                async def abort_check():
                    return await self._check_for_stop_signal(
                        context.channel_id, processing_started, seen_message_ids
                    )

                results = await self.tools.execute_batch(
                    response.tool_calls, tool_context, abort_check=abort_check
                )

                # Check if we were aborted by user stop signal
                user_aborted = any("[ABORTED]" in r.result for r in results)
                if user_aborted:
                    # Return early with acknowledgment
                    processing_time = int((time.time() - start_time) * 1000)
                    return AgentResponse(
                        content="Understood - I've stopped what I was doing. What would you like me to do instead?",
                        agent_name=self.name,
                        tool_calls_made=tool_calls_made,
                        processing_time_ms=processing_time,
                        status="aborted",
                    )

                # Check if agent called abort_task (needs clarification)
                for r in results:
                    if "[ABORT_TASK]" in r.result:
                        # Extract the reason (everything after the marker)
                        reason = r.result.replace("[ABORT_TASK]", "").strip()
                        processing_time = int((time.time() - start_time) * 1000)
                        return AgentResponse(
                            content=reason,
                            agent_name=self.name,
                            tool_calls_made=tool_calls_made,
                            processing_time_ms=processing_time,
                            status="needs_clarification",
                        )

                # Check if agent requested clarification
                for r in results:
                    if "[NEEDS_CLARIFICATION]" in r.result:
                        # Extract the question
                        question = r.result.replace("[NEEDS_CLARIFICATION]", "").strip()
                        processing_time = int((time.time() - start_time) * 1000)
                        return AgentResponse(
                            content=question,
                            agent_name=self.name,
                            tool_calls_made=tool_calls_made,
                            processing_time_ms=processing_time,
                            status="needs_clarification",
                        )

                # Count action iterations (non-monitoring tools)
                for tc in response.tool_calls:
                    if tc.name not in MONITORING_TOOLS:
                        action_iterations += 1

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

            # Reached max iterations - ask LLM to summarize what was done
            messages.append(
                Message(
                    role=Role.USER,
                    content=(
                        "You've made many tool calls. Please provide a final summary of:\n"
                        "1. What was originally requested\n"
                        "2. What you accomplished\n"
                        "3. The key findings or results (include actual data)\n"
                        "4. Any issues encountered\n"
                        "Do NOT make any more tool calls - just summarize."
                    ),
                )
            )

            # Get summary response (no tools to prevent more calls)
            summary_response = await self.llm.complete(
                messages=messages,
                tools=None,  # No tools - force text response
            )

            processing_time = int((time.time() - start_time) * 1000)
            return AgentResponse(
                content=summary_response.content
                or "Work completed but could not generate summary.",
                agent_name=self.name,
                tool_calls_made=tool_calls_made,
                processing_time_ms=processing_time,
                status="partial",
            )

        except Exception as e:
            logger.error(f"Altair processing error: {e}", exc_info=True)
            processing_time = int((time.time() - start_time) * 1000)
            return AgentResponse(
                content=f"I encountered an error while working: {str(e)}",
                agent_name=self.name,
                tool_calls_made=tool_calls_made,
                processing_time_ms=processing_time,
                status="error",
                error=str(e),
            )

    async def _build_messages(
        self,
        context: AgentContext,
        memory_context: Optional[str] = None,
        soul_context: Optional[str] = None,
    ) -> List[Message]:
        """Build message history for LLM."""
        # Use soul and memory-enhanced system prompt
        system_prompt = self.get_system_prompt(memory_context, soul_context)
        messages = [Message(role=Role.SYSTEM, content=system_prompt)]

        # Fetch recent Discord channel history for conversation context
        if self.discord_bot and context.channel_id:
            history = await self._fetch_discord_history(context.channel_id, limit=15)
            logger.info(f"Altair loaded {len(history)} messages from Discord history")
            messages.extend(history)
        else:
            logger.warning(f"Altair: No discord_bot ({self.discord_bot is not None}) or channel_id ({context.channel_id})")

        # Frame the current message based on source
        if context.is_from_agent:
            # Dispatch from Vega — frame as authoritative directive
            messages.append(
                Message(
                    role=Role.USER,
                    content=(
                        f"[DISPATCH FROM VEGA - THIS IS YOUR TASK]:\n"
                        f"{context.message_content}\n\n"
                        f"Do EXACTLY this task. The chat history above is for context only. "
                        f"Do not infer additional work beyond what is stated here."
                    ),
                )
            )
        else:
            # Direct user message
            user_name = f"User {context.user_id}"  # Default fallback
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
        self, channel_id: int, limit: int = 20
    ) -> List[Message]:
        """
        Fetch recent messages from Discord channel for conversation context.

        Returns formatted messages with author names so the agent knows who said what.
        """
        if not self.discord_bot:
            return []

        try:
            channel = self.discord_bot.get_channel(channel_id)
            if not channel:
                return []

            history_messages = []
            # Fetch messages (newest first, so we reverse later)
            async for msg in channel.history(limit=limit + 1):  # +1 to exclude current
                # Skip empty messages
                if not msg.content or not msg.content.strip():
                    continue

                # Clean the message content - strip any "**Name:** " prefix added by bots
                content = msg.content
                if content.startswith("**") and ":** " in content[:30]:
                    # Remove the "**Name:** " prefix
                    prefix_end = content.find(":** ") + 4
                    content = content[prefix_end:]

                # Determine role and format content with author
                if msg.author.bot:
                    # Bot message - mark as assistant with bot name
                    role = Role.ASSISTANT
                    formatted = f"[{msg.author.display_name}]: {content}"
                else:
                    # User message
                    role = Role.USER
                    formatted = f"[{msg.author.display_name}]: {content}"

                history_messages.append(Message(role=role, content=formatted))

            # Reverse to get chronological order and skip the most recent (current message)
            history_messages.reverse()
            if history_messages:
                history_messages = history_messages[:-1]  # Remove current message

            return history_messages

        except Exception as e:
            logger.warning(f"Failed to fetch Discord history: {e}")
            return []

    def _is_stop_message(self, content: str) -> bool:
        """Check if a message content matches stop patterns."""
        content_lower = content.lower().strip()
        for pattern in self.STOP_PATTERNS:
            if re.search(pattern, content_lower, re.IGNORECASE):
                return True
        return False

    async def _check_for_stop_signal(
        self,
        channel_id: int,
        since: datetime,
        seen_ids: set[int],
    ) -> Optional[str]:
        """
        Check if user has sent a stop signal since processing started.

        Returns:
            Stop reason string if should abort, None to continue
        """
        if not self.discord_bot:
            return None

        try:
            channel = self.discord_bot.get_channel(channel_id)
            if not channel:
                return None

            async for msg in channel.history(limit=5, after=since):
                # Skip messages we've seen or from bots
                if msg.id in seen_ids or msg.author.bot:
                    continue

                if self._is_stop_message(msg.content):
                    seen_ids.add(msg.id)
                    logger.info(f"Stop signal detected: '{msg.content[:50]}'")
                    return f"User requested stop: '{msg.content[:30]}'"

            return None

        except Exception as e:
            logger.warning(f"Failed to check for stop signal: {e}")
            return None

    async def _fetch_new_messages_since(
        self,
        channel_id: int,
        since: datetime,
        seen_ids: set[int],
    ) -> List[Message]:
        """
        Fetch messages posted after a given timestamp, excluding already-seen messages.

        Used during processing to check for new user input or agent responses
        that should influence the current task.

        Args:
            channel_id: Discord channel ID
            since: Fetch messages after this timestamp
            seen_ids: Set of message IDs already processed (will be updated in-place)
        """
        if not self.discord_bot:
            return []

        try:
            channel = self.discord_bot.get_channel(channel_id)
            if not channel:
                return []

            new_messages = []
            async for msg in channel.history(limit=10, after=since):
                # Skip messages we've already seen
                if msg.id in seen_ids:
                    continue

                if not msg.content or not msg.content.strip():
                    continue

                # Mark as seen
                seen_ids.add(msg.id)

                content = msg.content
                if msg.author.bot:
                    role = Role.ASSISTANT
                    formatted = f"[{msg.author.display_name}]: {content}"
                else:
                    role = Role.USER
                    formatted = f"[NEW - {msg.author.display_name}]: {content}"

                new_messages.append(Message(role=role, content=formatted))

            new_messages.reverse()
            return new_messages

        except Exception as e:
            logger.warning(f"Failed to fetch new messages: {e}")
            return []

    async def get_status_summary(self) -> str:
        """Get a summary of Altair's current status."""
        summary_parts = []

        # Current task
        if self._current_task:
            task_desc = self._current_task
            if task_desc.startswith("[TASK]"):
                task_desc = task_desc[6:].strip()
            if len(task_desc) > 50:
                task_desc = task_desc[:50] + "..."
            summary_parts.append(f"Current task: {task_desc}")

        # Current project
        if self._current_project:
            summary_parts.append(f"Project: {self._current_project}")

        # Current session
        if self._current_session_id:
            try:
                session = await self.session_registry.get_session(
                    self._current_session_id
                )
                if session:
                    status = "active" if session.is_alive else "stopped"
                    summary_parts.append(
                        f"Session #{self._current_session_id}: {status}"
                    )
            except Exception:
                pass

        # Check active sessions
        try:
            sessions = await self.session_registry.list_sessions()
            active_count = sum(1 for s in sessions if s.is_alive)
            if active_count > 0:
                summary_parts.append(f"Active sessions: {active_count}")
        except Exception:
            pass

        if not summary_parts:
            return "Altair is idle. No active work."

        return "Altair status: " + " | ".join(summary_parts)
