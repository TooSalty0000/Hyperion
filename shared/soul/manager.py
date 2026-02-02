"""Redis-backed soul/personality storage for agents."""

import asyncio
import json
import uuid
import logging
from typing import Optional, List, Dict, Any, TYPE_CHECKING
from datetime import datetime, timedelta

from .models import (
    SoulTrait,
    SoulContext,
    TraitEvolution,
    TraitType,
    TraitStability,
    REINFORCEMENT_DELTA,
    CONTRADICTION_DELTA,
    DAILY_DECAY_RATE,
    MIN_STRENGTH_THRESHOLD,
    MAX_STRENGTH,
    MIN_STRENGTH,
    STABILITY_RESISTANCE,
)

if TYPE_CHECKING:
    from shared.llm.interface import LLMProvider

logger = logging.getLogger(__name__)

# Lazy import
_redis = None


def _import_redis():
    global _redis
    if _redis is None:
        try:
            import redis.asyncio as redis_async
            _redis = redis_async
        except ImportError:
            raise ImportError(
                "redis is required for SoulManager. "
                "Install it with: pip install redis"
            )
    return _redis


class SoulManager:
    """
    Redis-backed soul/personality storage for agents.

    Provides persistent personality traits that evolve over time through
    reinforcement, contradiction, and decay. Unlike memories (facts),
    soul traits represent values, tendencies, and self-identity.

    Key patterns:
    - hyperion:soul:{agent_id}:trait:{trait_id} - Trait data (JSON)
    - hyperion:soul:{agent_id}:type:{type} - Set of trait IDs by type
    - hyperion:soul:{agent_id}:stability:{level} - Set of trait IDs by stability
    - hyperion:soul:{agent_id}:by_strength - Sorted set: trait_id -> strength
    - hyperion:soul:{agent_id}:evolution:{id} - Evolution event history
    - hyperion:soul:{agent_id}:last_decay - Timestamp of last decay application
    """

    # Key patterns
    TRAIT_KEY = "hyperion:soul:{agent_id}:trait:{trait_id}"
    TYPE_SET = "hyperion:soul:{agent_id}:type:{trait_type}"
    STABILITY_SET = "hyperion:soul:{agent_id}:stability:{stability}"
    BY_STRENGTH = "hyperion:soul:{agent_id}:by_strength"
    EVOLUTION_KEY = "hyperion:soul:{agent_id}:evolution:{evolution_id}"
    LAST_DECAY_KEY = "hyperion:soul:{agent_id}:last_decay"
    ALL_TRAITS_SET = "hyperion:soul:{agent_id}:all_traits"

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        llm: Optional['LLMProvider'] = None,
    ):
        """
        Initialize soul manager.

        Args:
            redis_url: Redis connection URL
            llm: Optional LLM provider for reflection and contradiction resolution
        """
        self._redis_url = redis_url
        self._redis = None
        self.llm = llm

    async def _get_client(self):
        """Get or create Redis client."""
        if self._redis is None:
            redis_async = _import_redis()
            self._redis = await redis_async.from_url(
                self._redis_url,
                encoding="utf-8",
                decode_responses=True
            )
        return self._redis

    # =========================================================================
    # Core CRUD Operations
    # =========================================================================

    async def create_trait(
        self,
        agent_id: str,
        trait_type: TraitType,
        name: str,
        description: str,
        expression: str,
        strength: float = 0.5,
        stability: TraitStability = TraitStability.NOVICE,
        confidence: float = 0.5,
        context_tags: Optional[List[str]] = None,
    ) -> SoulTrait:
        """
        Create a new soul trait.

        Args:
            agent_id: Agent identifier ("vega", "altair", etc.)
            trait_type: Type of trait (core_value, behavioral, etc.)
            name: Short identifier (e.g., "genuine_helpfulness")
            description: What this trait means
            expression: How it manifests in behavior
            strength: Initial strength 0.0-1.0
            stability: Initial stability level
            confidence: Self-assessed confidence 0.0-1.0
            context_tags: Optional tags for when this trait applies

        Returns:
            The created SoulTrait
        """
        redis = await self._get_client()

        trait_id = str(uuid.uuid4())

        trait = SoulTrait(
            id=trait_id,
            agent_id=agent_id,
            trait_type=trait_type,
            name=name,
            description=description,
            expression=expression,
            strength=max(MIN_STRENGTH, min(MAX_STRENGTH, strength)),
            stability=stability,
            confidence=max(MIN_STRENGTH, min(MAX_STRENGTH, confidence)),
            context_tags=context_tags or [],
        )

        # Store trait data
        trait_key = self.TRAIT_KEY.format(agent_id=agent_id, trait_id=trait_id)
        await redis.set(trait_key, json.dumps(trait.to_dict()))

        # Add to type set
        type_key = self.TYPE_SET.format(agent_id=agent_id, trait_type=trait_type.value)
        await redis.sadd(type_key, trait_id)

        # Add to stability set
        stability_key = self.STABILITY_SET.format(agent_id=agent_id, stability=stability.value)
        await redis.sadd(stability_key, trait_id)

        # Add to strength sorted set
        strength_key = self.BY_STRENGTH.format(agent_id=agent_id)
        await redis.zadd(strength_key, {trait_id: strength})

        # Add to all traits set
        all_key = self.ALL_TRAITS_SET.format(agent_id=agent_id)
        await redis.sadd(all_key, trait_id)

        logger.info(f"Created soul trait '{name}' ({trait_type.value}) for {agent_id}")
        return trait

    async def get_trait(self, agent_id: str, trait_id: str) -> Optional[SoulTrait]:
        """
        Retrieve a specific trait by ID.

        Also updates last_activated timestamp.
        """
        redis = await self._get_client()

        trait_key = self.TRAIT_KEY.format(agent_id=agent_id, trait_id=trait_id)
        data = await redis.get(trait_key)

        if not data:
            return None

        trait = SoulTrait.from_dict(json.loads(data))

        # Update last_activated
        trait.last_activated = datetime.utcnow()
        await redis.set(trait_key, json.dumps(trait.to_dict()))

        return trait

    async def get_trait_by_name(self, agent_id: str, name: str) -> Optional[SoulTrait]:
        """Get a trait by its name."""
        traits = await self.get_all_traits(agent_id)
        for trait in traits:
            if trait.name == name:
                return trait
        return None

    async def get_trait_readonly(self, agent_id: str, trait_id: str) -> Optional[SoulTrait]:
        """
        Retrieve a trait by ID without updating last_activated.

        Use this for read-only context building to avoid unnecessary writes.
        """
        redis = await self._get_client()

        trait_key = self.TRAIT_KEY.format(agent_id=agent_id, trait_id=trait_id)
        data = await redis.get(trait_key)

        if not data:
            return None

        return SoulTrait.from_dict(json.loads(data))

    async def get_traits_batch(
        self,
        agent_id: str,
        trait_ids: List[str],
    ) -> List[SoulTrait]:
        """
        Fetch multiple traits in a single mget call.

        This is a read-only operation that doesn't update last_activated.
        Use for context building to avoid N+1 Redis calls.

        Args:
            agent_id: Agent identifier
            trait_ids: List of trait IDs to fetch

        Returns:
            List of SoulTrait objects (excludes any not found)
        """
        if not trait_ids:
            return []

        redis = await self._get_client()
        keys = [
            self.TRAIT_KEY.format(agent_id=agent_id, trait_id=tid)
            for tid in trait_ids
        ]

        results = await redis.mget(keys)

        traits = []
        for data in results:
            if data:
                traits.append(SoulTrait.from_dict(json.loads(data)))

        return traits

    async def delete_trait(self, agent_id: str, trait_id: str) -> bool:
        """
        Remove a trait entirely.

        Returns:
            True if deleted, False if not found
        """
        redis = await self._get_client()

        trait_key = self.TRAIT_KEY.format(agent_id=agent_id, trait_id=trait_id)
        data = await redis.get(trait_key)

        if not data:
            return False

        trait = SoulTrait.from_dict(json.loads(data))

        # Remove from type set
        type_key = self.TYPE_SET.format(agent_id=agent_id, trait_type=trait.trait_type.value)
        await redis.srem(type_key, trait_id)

        # Remove from stability set
        stability_key = self.STABILITY_SET.format(agent_id=agent_id, stability=trait.stability.value)
        await redis.srem(stability_key, trait_id)

        # Remove from strength sorted set
        strength_key = self.BY_STRENGTH.format(agent_id=agent_id)
        await redis.zrem(strength_key, trait_id)

        # Remove from all traits set
        all_key = self.ALL_TRAITS_SET.format(agent_id=agent_id)
        await redis.srem(all_key, trait_id)

        # Delete the trait
        await redis.delete(trait_key)

        logger.info(f"Deleted soul trait '{trait.name}' for {agent_id}")
        return True

    # =========================================================================
    # Evolution Operations
    # =========================================================================

    async def reinforce_trait(
        self,
        agent_id: str,
        trait_id: str,
        trigger: str,
        evidence: Optional[str] = None,
    ) -> Optional[SoulTrait]:
        """
        Strengthen a trait through positive evidence.

        Increases strength by REINFORCEMENT_DELTA * stability_resistance.
        May advance stability level if threshold reached.

        Args:
            agent_id: Agent identifier
            trait_id: Trait to reinforce
            trigger: What triggered this reinforcement
            evidence: Supporting evidence

        Returns:
            Updated trait or None if not found
        """
        redis = await self._get_client()

        trait = await self.get_trait(agent_id, trait_id)
        if not trait:
            return None

        # Calculate reinforcement with stability resistance
        resistance = trait.get_stability_resistance()
        delta = REINFORCEMENT_DELTA * resistance

        # Record evolution event
        strength_before = trait.strength
        stability_before = trait.stability

        # Apply reinforcement
        trait.strength = min(MAX_STRENGTH, trait.strength + delta)
        trait.reinforcement_count += 1
        trait.last_activated = datetime.utcnow()

        # Check for stability advancement
        if trait.should_advance_stability():
            old_stability = trait.stability
            trait.stability = trait.get_next_stability()
            logger.info(
                f"Trait '{trait.name}' advanced from {old_stability.value} to {trait.stability.value}"
            )

            # Update stability sets
            old_key = self.STABILITY_SET.format(agent_id=agent_id, stability=old_stability.value)
            new_key = self.STABILITY_SET.format(agent_id=agent_id, stability=trait.stability.value)
            await redis.srem(old_key, trait_id)
            await redis.sadd(new_key, trait_id)

        # Save updated trait
        trait_key = self.TRAIT_KEY.format(agent_id=agent_id, trait_id=trait_id)
        await redis.set(trait_key, json.dumps(trait.to_dict()))

        # Update strength sorted set
        strength_key = self.BY_STRENGTH.format(agent_id=agent_id)
        await redis.zadd(strength_key, {trait_id: trait.strength})

        # Record evolution event
        await self._record_evolution(
            agent_id=agent_id,
            trait_id=trait_id,
            event_type="reinforce",
            trigger=trigger,
            evidence=evidence,
            strength_before=strength_before,
            strength_after=trait.strength,
            stability_before=stability_before.value,
            stability_after=trait.stability.value,
        )

        logger.debug(
            f"Reinforced trait '{trait.name}' for {agent_id}: "
            f"{strength_before:.2f} → {trait.strength:.2f}"
        )
        return trait

    async def contradict_trait(
        self,
        agent_id: str,
        trait_id: str,
        trigger: str,
        evidence: Optional[str] = None,
    ) -> Optional[SoulTrait]:
        """
        Weaken a trait through contradictory evidence.

        Decreases strength by CONTRADICTION_DELTA * stability_resistance.
        High-stability traits resist contradiction more.

        Args:
            agent_id: Agent identifier
            trait_id: Trait to contradict
            trigger: What triggered this contradiction
            evidence: Contradictory evidence

        Returns:
            Updated trait or None if not found
        """
        redis = await self._get_client()

        trait = await self.get_trait(agent_id, trait_id)
        if not trait:
            return None

        # Calculate contradiction with stability resistance
        resistance = trait.get_stability_resistance()
        delta = CONTRADICTION_DELTA * resistance

        # Record evolution event
        strength_before = trait.strength

        # Apply contradiction
        trait.strength = max(MIN_STRENGTH, trait.strength - delta)
        trait.contradiction_count += 1
        trait.last_activated = datetime.utcnow()

        # Save updated trait
        trait_key = self.TRAIT_KEY.format(agent_id=agent_id, trait_id=trait_id)
        await redis.set(trait_key, json.dumps(trait.to_dict()))

        # Update strength sorted set
        strength_key = self.BY_STRENGTH.format(agent_id=agent_id)
        await redis.zadd(strength_key, {trait_id: trait.strength})

        # Record evolution event
        await self._record_evolution(
            agent_id=agent_id,
            trait_id=trait_id,
            event_type="contradict",
            trigger=trigger,
            evidence=evidence,
            strength_before=strength_before,
            strength_after=trait.strength,
            stability_before=trait.stability.value,
            stability_after=trait.stability.value,
        )

        logger.debug(
            f"Contradicted trait '{trait.name}' for {agent_id}: "
            f"{strength_before:.2f} → {trait.strength:.2f}"
        )
        return trait

    async def apply_decay(self, agent_id: str) -> int:
        """
        Apply time-based decay to inactive traits.

        Called periodically (e.g., daily). Traits that haven't been activated
        recently lose strength. CORE stability traits are exempt from decay.

        Returns:
            Number of traits affected by decay
        """
        redis = await self._get_client()

        # Check last decay time
        decay_key = self.LAST_DECAY_KEY.format(agent_id=agent_id)
        last_decay_str = await redis.get(decay_key)

        if last_decay_str:
            last_decay = datetime.fromisoformat(last_decay_str)
            days_since_decay = (datetime.utcnow() - last_decay).days
            if days_since_decay < 1:
                return 0  # Already decayed today
        else:
            days_since_decay = 1  # First decay

        # Get all traits
        traits = await self.get_all_traits(agent_id)
        decayed_count = 0

        for trait in traits:
            # Skip CORE stability traits
            if trait.stability == TraitStability.CORE:
                continue

            # Calculate days since last activation
            days_inactive = (datetime.utcnow() - trait.last_activated).days

            if days_inactive < 1:
                continue  # Recently active

            # Apply decay based on inactivity
            decay_amount = DAILY_DECAY_RATE * days_inactive * trait.get_stability_resistance()
            new_strength = max(MIN_STRENGTH, trait.strength - decay_amount)

            if new_strength != trait.strength:
                # Record evolution
                await self._record_evolution(
                    agent_id=agent_id,
                    trait_id=trait.id,
                    event_type="decay",
                    trigger=f"Inactive for {days_inactive} days",
                    strength_before=trait.strength,
                    strength_after=new_strength,
                    stability_before=trait.stability.value,
                    stability_after=trait.stability.value,
                )

                trait.strength = new_strength

                # Save updated trait
                trait_key = self.TRAIT_KEY.format(agent_id=agent_id, trait_id=trait.id)
                await redis.set(trait_key, json.dumps(trait.to_dict()))

                # Update strength sorted set
                strength_key = self.BY_STRENGTH.format(agent_id=agent_id)
                await redis.zadd(strength_key, {trait.id: trait.strength})

                decayed_count += 1

        # Update last decay timestamp
        await redis.set(decay_key, datetime.utcnow().isoformat())

        if decayed_count > 0:
            logger.info(f"Applied decay to {decayed_count} traits for {agent_id}")

        return decayed_count

    async def resolve_contradiction(
        self,
        agent_id: str,
        trait_id: str,
        new_evidence: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Use LLM to resolve a trait contradiction.

        When a trait receives significant contradictory evidence, this method
        uses the LLM to determine how the trait should evolve.

        Args:
            agent_id: Agent identifier
            trait_id: Trait in conflict
            new_evidence: New contradictory evidence

        Returns:
            Resolution result or None if no LLM available
        """
        if not self.llm:
            logger.warning("No LLM available for contradiction resolution")
            return None

        trait = await self.get_trait(agent_id, trait_id)
        if not trait:
            return None

        # Build resolution prompt
        from shared.llm.types import Message, Role

        prompt = f"""You are analyzing a personality trait that has received contradictory evidence.

TRAIT: {trait.name}
DESCRIPTION: {trait.description}
EXPRESSION: {trait.expression}
CURRENT STRENGTH: {trait.strength:.2f}
REINFORCEMENT COUNT: {trait.reinforcement_count}
CONTRADICTION COUNT: {trait.contradiction_count}

NEW CONTRADICTORY EVIDENCE:
{new_evidence}

Analyze this situation and respond with JSON:
{{
    "action": "modify" | "replace" | "weaken" | "accept",
    "reasoning": "explanation of your decision",
    "new_expression": "if modify/replace, the new expression",
    "new_description": "if modify/replace, the new description",
    "strength_adjustment": float between -0.3 and 0.1
}}

Guidelines:
- "modify": Update the trait's expression while keeping its core meaning
- "replace": Fundamentally change what this trait represents
- "weaken": Accept the contradiction by significantly reducing strength
- "accept": Acknowledge contradiction but maintain the trait (minor strength reduction)
"""

        try:
            messages = [Message(role=Role.USER, content=prompt)]
            response = await self.llm.complete(messages=messages)

            # Parse JSON response
            import re
            json_match = re.search(r'\{[^}]+\}', response.content, re.DOTALL)
            if not json_match:
                logger.warning("Could not parse resolution response")
                return None

            resolution = json.loads(json_match.group())

            # Apply resolution
            strength_before = trait.strength
            action = resolution.get("action", "accept")

            if action == "modify" or action == "replace":
                if resolution.get("new_expression"):
                    trait.expression = resolution["new_expression"]
                if resolution.get("new_description"):
                    trait.description = resolution["new_description"]

            # Apply strength adjustment
            adjustment = resolution.get("strength_adjustment", -0.05)
            trait.strength = max(MIN_STRENGTH, min(MAX_STRENGTH, trait.strength + adjustment))

            # Save updated trait
            redis = await self._get_client()
            trait_key = self.TRAIT_KEY.format(agent_id=agent_id, trait_id=trait_id)
            await redis.set(trait_key, json.dumps(trait.to_dict()))

            # Update strength sorted set
            strength_key = self.BY_STRENGTH.format(agent_id=agent_id)
            await redis.zadd(strength_key, {trait_id: trait.strength})

            # Record evolution
            await self._record_evolution(
                agent_id=agent_id,
                trait_id=trait_id,
                event_type="resolve",
                trigger=f"Contradiction resolution: {action}",
                evidence=new_evidence,
                strength_before=strength_before,
                strength_after=trait.strength,
                stability_before=trait.stability.value,
                stability_after=trait.stability.value,
            )

            logger.info(f"Resolved contradiction for trait '{trait.name}': {action}")
            return resolution

        except Exception as e:
            logger.error(f"Error resolving contradiction: {e}")
            return None

    # =========================================================================
    # Retrieval Operations
    # =========================================================================

    async def get_traits_by_type(
        self,
        agent_id: str,
        trait_type: TraitType,
        min_strength: float = 0.0,
    ) -> List[SoulTrait]:
        """
        Get all traits of a specific type.

        Uses batch fetch to avoid N+1 Redis calls.
        """
        redis = await self._get_client()

        type_key = self.TYPE_SET.format(agent_id=agent_id, trait_type=trait_type.value)
        trait_ids = await redis.smembers(type_key)

        if not trait_ids:
            return []

        # Batch fetch all traits in one mget call
        traits = await self.get_traits_batch(agent_id, list(trait_ids))

        # Filter by min_strength
        filtered = [t for t in traits if t.strength >= min_strength]

        return sorted(filtered, key=lambda t: t.strength, reverse=True)

    async def get_strongest_traits(
        self,
        agent_id: str,
        limit: int = 10,
        min_strength: float = MIN_STRENGTH_THRESHOLD,
    ) -> List[SoulTrait]:
        """
        Get the strongest traits for an agent.

        Uses batch fetch to avoid N+1 Redis calls.
        """
        redis = await self._get_client()

        strength_key = self.BY_STRENGTH.format(agent_id=agent_id)
        # Get top traits by strength (descending)
        trait_ids = await redis.zrevrange(strength_key, 0, limit - 1)

        if not trait_ids:
            return []

        # Batch fetch all traits in one mget call
        traits = await self.get_traits_batch(agent_id, list(trait_ids))

        # Filter by min_strength and maintain order from sorted set
        # Create a map for O(1) lookup
        trait_map = {t.id: t for t in traits}
        result = []
        for tid in trait_ids:
            if tid in trait_map and trait_map[tid].strength >= min_strength:
                result.append(trait_map[tid])

        return result

    async def get_all_traits(self, agent_id: str) -> List[SoulTrait]:
        """
        Get all traits for an agent.

        Uses batch fetch to avoid N+1 Redis calls.
        """
        redis = await self._get_client()

        all_key = self.ALL_TRAITS_SET.format(agent_id=agent_id)
        trait_ids = await redis.smembers(all_key)

        if not trait_ids:
            return []

        # Batch fetch all traits in one mget call
        return await self.get_traits_batch(agent_id, list(trait_ids))

    async def build_soul_context(
        self,
        agent_id: str,
        context_tags: Optional[List[str]] = None,
        min_strength: float = MIN_STRENGTH_THRESHOLD,
    ) -> SoulContext:
        """
        Build complete soul context for agent prompt injection.

        Uses parallel fetches for all trait types to minimize Redis round-trips.

        Args:
            agent_id: Agent identifier
            context_tags: Optional tags to filter relevant traits
            min_strength: Minimum strength threshold for inclusion

        Returns:
            SoulContext ready for prompt formatting
        """
        # Fetch all trait types in parallel
        core_values, behavioral, preferences, skills, relationships = await asyncio.gather(
            self.get_traits_by_type(agent_id, TraitType.CORE_VALUE, min_strength),
            self.get_traits_by_type(agent_id, TraitType.BEHAVIORAL, min_strength),
            self.get_traits_by_type(agent_id, TraitType.PREFERENCE, min_strength),
            self.get_traits_by_type(agent_id, TraitType.SKILL, min_strength),
            self.get_traits_by_type(agent_id, TraitType.RELATIONSHIP, min_strength),
        )

        # Filter by context tags if provided
        if context_tags:
            def matches_tags(trait: SoulTrait) -> bool:
                if not trait.context_tags:
                    return True  # No tags = applies everywhere
                return any(tag in trait.context_tags for tag in context_tags)

            core_values = [t for t in core_values if matches_tags(t)]
            behavioral = [t for t in behavioral if matches_tags(t)]
            preferences = [t for t in preferences if matches_tags(t)]
            skills = [t for t in skills if matches_tags(t)]
            relationships = [t for t in relationships if matches_tags(t)]

        return SoulContext(
            core_values=core_values,
            behavioral_traits=behavioral,
            learned_preferences=preferences,
            skill_confidences=skills,
            relationships=relationships,
        )

    # =========================================================================
    # Reflection Operations
    # =========================================================================

    async def trigger_reflection(
        self,
        agent_id: str,
        recent_interactions: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Periodic reflection to consolidate soul.

        Analyzes recent interactions to:
        - Identify behavioral patterns that should become traits
        - Detect skill improvements/degradation
        - Recognize relationship dynamics
        - Propose new traits

        Args:
            agent_id: Agent identifier
            recent_interactions: Summary of recent interactions

        Returns:
            Reflection result with proposed changes
        """
        if not self.llm:
            logger.warning("No LLM available for reflection")
            return None

        # Get current soul state
        soul_context = await self.build_soul_context(agent_id)
        current_traits = "\n".join(
            f"- {t.name}: {t.expression} (strength: {t.strength:.2f})"
            for t in soul_context.get_all_traits()
        ) or "No traits yet."

        # Build reflection prompt
        from shared.llm.types import Message, Role

        prompt = f"""You are reflecting on an AI agent's personality development.

AGENT: {agent_id}

CURRENT SOUL TRAITS:
{current_traits}

RECENT INTERACTIONS:
{recent_interactions}

Analyze the interactions and identify any patterns that suggest:
1. New traits that should be created (behavioral patterns, preferences)
2. Existing traits that should be reinforced (consistent behavior)
3. Traits that might need contradiction (inconsistent behavior)
4. Skill confidence changes (successes/failures)

Respond with JSON:
{{
    "new_traits": [
        {{
            "trait_type": "behavioral" | "preference" | "skill",
            "name": "short_identifier",
            "description": "what this trait means",
            "expression": "how it manifests",
            "initial_strength": 0.3-0.5,
            "reasoning": "why this trait should be created"
        }}
    ],
    "reinforce": [
        {{
            "trait_name": "existing_trait_name",
            "evidence": "what supports this"
        }}
    ],
    "contradict": [
        {{
            "trait_name": "existing_trait_name",
            "evidence": "what contradicts this"
        }}
    ],
    "insights": "overall observations about personality development"
}}

Be conservative - only propose changes with clear evidence from interactions.
"""

        try:
            messages = [Message(role=Role.USER, content=prompt)]
            response = await self.llm.complete(messages=messages)

            # Parse JSON response
            import re
            json_match = re.search(r'\{[\s\S]+\}', response.content)
            if not json_match:
                logger.warning("Could not parse reflection response")
                return None

            reflection = json.loads(json_match.group())

            # Apply proposed changes
            results = {
                "created": [],
                "reinforced": [],
                "contradicted": [],
                "insights": reflection.get("insights", ""),
            }

            # Create new traits
            for new_trait in reflection.get("new_traits", []):
                try:
                    trait_type = TraitType(new_trait.get("trait_type", "behavioral"))
                    created = await self.create_trait(
                        agent_id=agent_id,
                        trait_type=trait_type,
                        name=new_trait["name"],
                        description=new_trait["description"],
                        expression=new_trait["expression"],
                        strength=new_trait.get("initial_strength", 0.4),
                    )
                    results["created"].append(created.name)
                except Exception as e:
                    logger.warning(f"Failed to create trait: {e}")

            # Reinforce existing traits
            for reinforce in reflection.get("reinforce", []):
                trait = await self.get_trait_by_name(agent_id, reinforce["trait_name"])
                if trait:
                    await self.reinforce_trait(
                        agent_id=agent_id,
                        trait_id=trait.id,
                        trigger="Periodic reflection",
                        evidence=reinforce.get("evidence"),
                    )
                    results["reinforced"].append(trait.name)

            # Contradict existing traits
            for contradict in reflection.get("contradict", []):
                trait = await self.get_trait_by_name(agent_id, contradict["trait_name"])
                if trait:
                    await self.contradict_trait(
                        agent_id=agent_id,
                        trait_id=trait.id,
                        trigger="Periodic reflection",
                        evidence=contradict.get("evidence"),
                    )
                    results["contradicted"].append(trait.name)

            logger.info(
                f"Reflection complete for {agent_id}: "
                f"created={len(results['created'])}, "
                f"reinforced={len(results['reinforced'])}, "
                f"contradicted={len(results['contradicted'])}"
            )
            return results

        except Exception as e:
            logger.error(f"Error during reflection: {e}")
            return None

    async def introspect(
        self,
        agent_id: str,
        question: str,
    ) -> Optional[str]:
        """
        Self-awareness query about personality.

        Allows the agent to reflect on why they behave certain ways.

        Args:
            agent_id: Agent identifier
            question: Question about self (e.g., "Why do I prioritize X?")

        Returns:
            Introspective response
        """
        if not self.llm:
            return "I cannot introspect without an LLM configured."

        # Get current soul state
        soul_context = await self.build_soul_context(agent_id)

        if not soul_context.has_traits():
            return "My soul is still forming. I haven't developed enough traits to introspect meaningfully yet."

        # Build introspection prompt
        from shared.llm.types import Message, Role

        current_traits = "\n".join(
            f"- {t.name} ({t.trait_type.value}): {t.expression} [strength: {t.strength:.2f}]"
            for t in soul_context.get_all_traits()
        )

        prompt = f"""You are introspecting on your own personality as an AI agent named {agent_id}.

YOUR SOUL TRAITS:
{current_traits}

INTROSPECTION QUESTION:
{question}

Reflect deeply on this question in first person. Consider:
- Which traits are relevant to this question?
- How did these traits develop?
- How do they influence your behavior?

Respond thoughtfully and authentically, as if genuinely exploring your own nature.
Keep the response focused and under 200 words.
"""

        try:
            messages = [Message(role=Role.USER, content=prompt)]
            response = await self.llm.complete(messages=messages)
            return response.content

        except Exception as e:
            logger.error(f"Error during introspection: {e}")
            return f"I encountered an error while introspecting: {e}"

    # =========================================================================
    # Helper Methods
    # =========================================================================

    async def _record_evolution(
        self,
        agent_id: str,
        trait_id: str,
        event_type: str,
        trigger: str,
        evidence: Optional[str] = None,
        strength_before: float = 0.0,
        strength_after: float = 0.0,
        stability_before: Optional[str] = None,
        stability_after: Optional[str] = None,
    ):
        """Record an evolution event for a trait."""
        redis = await self._get_client()

        evolution_id = str(uuid.uuid4())

        evolution = TraitEvolution(
            id=evolution_id,
            trait_id=trait_id,
            agent_id=agent_id,
            event_type=event_type,
            trigger=trigger,
            evidence=evidence,
            strength_before=strength_before,
            strength_after=strength_after,
            stability_before=stability_before,
            stability_after=stability_after,
        )

        # Store evolution event
        evolution_key = self.EVOLUTION_KEY.format(
            agent_id=agent_id, evolution_id=evolution_id
        )
        await redis.set(evolution_key, json.dumps(evolution.to_dict()))

        # Add to trait's evolution history
        trait = await self.get_trait(agent_id, trait_id)
        if trait:
            trait.evolution_history.append(evolution_id)
            # Keep only last 50 evolution events per trait
            if len(trait.evolution_history) > 50:
                trait.evolution_history = trait.evolution_history[-50:]

            trait_key = self.TRAIT_KEY.format(agent_id=agent_id, trait_id=trait_id)
            await redis.set(trait_key, json.dumps(trait.to_dict()))

    async def ensure_defaults_initialized(
        self,
        agent_id: str,
        default_traits: List[Dict[str, Any]],
    ) -> int:
        """
        Ensure default traits are initialized for an agent.

        Only creates traits if the agent has no existing traits.

        Args:
            agent_id: Agent identifier
            default_traits: List of default trait configurations

        Returns:
            Number of traits created
        """
        existing = await self.get_all_traits(agent_id)
        if existing:
            return 0  # Already has traits

        created = 0
        for trait_config in default_traits:
            try:
                await self.create_trait(
                    agent_id=agent_id,
                    trait_type=TraitType(trait_config["trait_type"]),
                    name=trait_config["name"],
                    description=trait_config["description"],
                    expression=trait_config["expression"],
                    strength=trait_config.get("strength", 0.5),
                    stability=TraitStability(trait_config.get("stability", "novice")),
                )
                created += 1
            except Exception as e:
                logger.warning(f"Failed to create default trait: {e}")

        if created > 0:
            logger.info(f"Initialized {created} default traits for {agent_id}")

        return created
