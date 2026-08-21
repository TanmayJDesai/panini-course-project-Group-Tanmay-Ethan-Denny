"""PANINI connected-DAG Reasoning Inference Chain Retrieval."""

from __future__ import annotations

import itertools
import math
import re
from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence


PLACEHOLDER_PATTERN = re.compile(r"<ENTITY_Q(\d+)>")


@dataclass(frozen=True)
class Candidate:
    qa_uid: str
    answer_names: tuple[str, ...]
    score: float
    question: str = ""
    answer_ids: tuple[str, ...] = ()
    answer_role_states: tuple[str, ...] = ()
    document_id: str = ""
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ChainState:
    steps: tuple[Candidate, ...]
    answers_by_step: Mapping[int, str | tuple[str, ...]]
    score: float
    last_hop_score: float = 0.0

    @property
    def current_answers(self) -> tuple[str, ...]:
        return self.steps[-1].answer_names if self.steps else ()


@dataclass(frozen=True)
class RICRResult:
    components: tuple[tuple[int, ...], ...]
    chains: tuple[ChainState, ...]
    evidence: tuple[Candidate, ...]
    issued_queries: tuple[str, ...]
    fallback: bool = False
    trace: Mapping[str, object] = field(default_factory=dict)


def normalize_entity_name(value: str) -> str:
    return " ".join(value.casefold().split())


def geometric_mean(scores: Sequence[float], epsilon: float = 1e-12) -> float:
    if not scores:
        return 0.0
    safe = [max(float(score), epsilon) for score in scores]
    return float(math.exp(sum(math.log(score) for score in safe) / len(safe)))


def panini_chain_score(steps: Sequence[Candidate]) -> float:
    normalized = [
        max(1e-6, min(1.0, 0.5 * (float(step.score) + 1.0)))
        for step in steps
    ]
    return geometric_mean(normalized, epsilon=1e-6)


def harmonic_mean(scores: Sequence[float]) -> float:
    valid = [float(score) for score in scores if float(score) > 1e-6]
    return len(valid) / sum(1.0 / score for score in valid) if valid else 1e-6


def instantiate_question(
    template: str,
    answers_by_step: Mapping[int, str | tuple[str, ...]],
) -> str:
    def replace(match: re.Match[str]) -> str:
        step = int(match.group(1))
        if step not in answers_by_step:
            raise KeyError(f"Question references unresolved Q{step}: {template}")
        value = answers_by_step[step]
        return ", ".join(value) if isinstance(value, tuple) else value

    return PLACEHOLDER_PATTERN.sub(replace, template)


def _deterministic_topological_order(
    nodes: set[int], edges: set[tuple[int, int]]
) -> list[int]:
    parents = {node: set() for node in nodes}
    children = {node: set() for node in nodes}
    for parent, child in edges:
        if parent in nodes and child in nodes:
            parents[child].add(parent)
            children[parent].add(child)
    ready = sorted(node for node in nodes if not parents[node])
    order: list[int] = []
    while ready:
        node = ready.pop(0)
        order.append(node)
        for child in sorted(children[node]):
            parents[child].discard(node)
            if not parents[child] and child not in order and child not in ready:
                ready.append(child)
                ready.sort()
    if len(order) != len(nodes):
        raise ValueError("Retrieval dependency graph contains a cycle")
    return order


def _dependency_maps(
    decomposed_questions: Sequence[Mapping[str, object]],
) -> tuple[set[int], dict[int, set[int]], dict[int, set[int]], set[tuple[int, int]]]:
    count = len(decomposed_questions)
    retrieval_nodes = {
        index
        for index, row in enumerate(decomposed_questions, start=1)
        if bool(row.get("requires_retrieval", True))
    }
    full_edges: set[tuple[int, int]] = set()
    retrieval_edges: set[tuple[int, int]] = set()
    for child, row in enumerate(decomposed_questions, start=1):
        question = str(row.get("question", ""))
        for raw_parent in PLACEHOLDER_PATTERN.findall(question):
            parent = int(raw_parent)
            if parent < 1 or parent > count:
                raise ValueError(f"Q{child} references missing Q{parent}")
            if parent >= child:
                raise ValueError(f"Q{child} references non-earlier Q{parent}")
            full_edges.add((parent, child))
            if parent in retrieval_nodes and child in retrieval_nodes:
                retrieval_edges.add((parent, child))

    # Validate the complete dependency graph, including deterministic nodes.
    _deterministic_topological_order(set(range(1, count + 1)), full_edges)

    parents = {node: set() for node in retrieval_nodes}
    children = {node: set() for node in retrieval_nodes}
    for parent, child in retrieval_edges:
        parents[child].add(parent)
        children[parent].add(child)
    return retrieval_nodes, parents, children, retrieval_edges


def identify_retrieval_components(
    decomposed_questions: Sequence[Mapping[str, object]],
) -> list[list[int]]:
    """Return non-singleton retrieval components in deterministic topological order."""

    retrieval_nodes, parents, children, retrieval_edges = _dependency_maps(
        decomposed_questions
    )
    unseen = set(retrieval_nodes)
    components: list[list[int]] = []
    while unseen:
        start = min(unseen)
        stack = [start]
        component: set[int] = set()
        while stack:
            node = stack.pop()
            if node in component:
                continue
            component.add(node)
            stack.extend(sorted((parents[node] | children[node]) - component, reverse=True))
        unseen -= component
        if len(component) > 1:
            components.append(
                _deterministic_topological_order(component, retrieval_edges)
            )
    components.sort(key=lambda component: (component[0], tuple(component)))
    return components


def _candidate_summary(candidate: Candidate) -> dict[str, object]:
    return {
        "qa_uid": candidate.qa_uid,
        "answer_names": list(candidate.answer_names),
        "answer_ids": list(candidate.answer_ids),
        "score": float(candidate.score),
        "question": candidate.question,
        "document_id": candidate.document_id,
    }


def _state_summary(state: ChainState) -> dict[str, object]:
    return {
        "qa_ids": [candidate.qa_uid for candidate in state.steps],
        "answers_by_step": {str(key): value for key, value in state.answers_by_step.items()},
        "score": float(state.score),
        "last_hop_score": float(state.last_hop_score),
    }


def _state_sort_key(state: ChainState):
    return (
        -float(state.score),
        -float(state.last_hop_score),
        tuple(candidate.qa_uid for candidate in state.steps),
        tuple(
            (int(key), str(value))
            for key, value in sorted(state.answers_by_step.items())
        ),
    )


def _deduplicate_candidates(
    candidates: Sequence[Candidate], limit: int
) -> list[Candidate]:
    best: dict[str, Candidate] = {}
    for candidate in candidates:
        if not isinstance(candidate, Candidate):
            raise TypeError("retrieve_and_score must return Candidate objects")
        uid = str(candidate.qa_uid)
        incumbent = best.get(uid)
        if incumbent is None or (
            float(candidate.score), tuple(candidate.answer_names), uid
        ) > (
            float(incumbent.score), tuple(incumbent.answer_names), uid
        ):
            best[uid] = candidate
    ordered = sorted(
        best.values(), key=lambda candidate: (-float(candidate.score), candidate.qa_uid)
    )
    return ordered[: max(0, limit)]


def _append_step(
    state: ChainState,
    node: int,
    candidate: Candidate,
    answer_value: str | tuple[str, ...],
) -> ChainState:
    answers = dict(state.answers_by_step)
    answers[node] = answer_value
    steps = state.steps + (candidate,)
    return ChainState(
        steps=steps,
        answers_by_step=answers,
        score=panini_chain_score(steps),
        last_hop_score=float(candidate.score),
    )


def _answer_identity(candidate: Candidate, answer_index: int, answer_name: str) -> str:
    raw_id = (
        str(candidate.answer_ids[answer_index])
        if answer_index < len(candidate.answer_ids)
        else ""
    )
    if raw_id:
        if "::" in raw_id:
            return f"id::{raw_id}"
        gsw_file = str(candidate.metadata.get("gsw_file", ""))
        if candidate.document_id and gsw_file:
            return f"id::{candidate.document_id}::{gsw_file}::{raw_id}"
        if candidate.document_id:
            return f"id::{candidate.document_id}::{raw_id}"
        # Never treat a raw local ID such as e1 as global.
        return f"id::{candidate.qa_uid}::{raw_id}"
    return f"name::{normalize_entity_name(answer_name)}"


def _expand_intermediate(
    base_states: Sequence[ChainState],
    node: int,
    candidates: Sequence[Candidate],
    beam_width: int,
    unique_intermediate_entities: bool,
) -> tuple[list[ChainState], list[ChainState]]:
    expansions: list[tuple[str, ChainState]] = []
    for state in base_states:
        for candidate in candidates:
            seen_within_candidate: set[str] = set()
            for answer_index, answer_name in enumerate(candidate.answer_names):
                answer = str(answer_name).strip()
                if not answer:
                    continue
                identity = _answer_identity(candidate, answer_index, answer)
                if identity in seen_within_candidate:
                    continue
                seen_within_candidate.add(identity)
                expansions.append(
                    (identity, _append_step(state, node, candidate, answer))
                )

    if unique_intermediate_entities:
        best_by_entity: dict[str, ChainState] = {}
        for identity, state in expansions:
            incumbent = best_by_entity.get(identity)
            if incumbent is None or _state_sort_key(state) < _state_sort_key(incumbent):
                best_by_entity[identity] = state
        all_states = list(best_by_entity.values())
    else:
        all_states = [state for _, state in expansions]

    ordered = sorted(all_states, key=_state_sort_key)
    return ordered[:beam_width], ordered[beam_width:]


def _expand_final(
    base_states: Sequence[ChainState],
    node: int,
    candidates: Sequence[Candidate],
    beam_width: int,
) -> tuple[list[ChainState], list[ChainState]]:
    states: list[ChainState] = []
    for state in base_states:
        for candidate in candidates:
            if len(candidate.answer_names) == 1:
                answer_value: str | tuple[str, ...] = candidate.answer_names[0]
            else:
                answer_value = tuple(candidate.answer_names)
            states.append(_append_step(state, node, candidate, answer_value))
    ordered = sorted(states, key=_state_sort_key)
    return ordered[:beam_width], ordered[beam_width:]


def _compatible_merge(states: Sequence[ChainState]) -> ChainState | None:
    answers: dict[int, str | tuple[str, ...]] = {}
    steps: list[Candidate] = []
    seen_qa: set[str] = set()
    for state in states:
        for step, value in state.answers_by_step.items():
            if step in answers and answers[step] != value:
                return None
            answers[step] = value
        for candidate in state.steps:
            if candidate.qa_uid not in seen_qa:
                seen_qa.add(candidate.qa_uid)
                steps.append(candidate)
    return ChainState(
        steps=tuple(steps),
        answers_by_step=answers,
        score=panini_chain_score(steps) if steps else 1.0,
        last_hop_score=float(steps[-1].score) if steps else 1.0,
    )


def _combine_parent_beams(
    parent_nodes: Sequence[int],
    beams: Mapping[int, Sequence[ChainState]],
    beam_width: int,
    threshold: float,
) -> tuple[list[ChainState], list[dict[str, object]]]:
    combinations: list[tuple[float, tuple[ChainState, ...], ChainState]] = []
    for states in itertools.product(*(beams[parent] for parent in parent_nodes)):
        merged = _compatible_merge(states)
        if merged is None:
            continue
        harmonic = harmonic_mean([state.score for state in states])
        combinations.append((harmonic, tuple(states), merged))
    combinations.sort(
        key=lambda item: (
            -item[0],
            tuple(tuple(candidate.qa_uid for candidate in state.steps) for state in item[1]),
        )
    )
    examined = combinations[:beam_width]
    selected = [item for item in examined if item[0] >= threshold]
    if not selected and examined:
        selected = [examined[0]]
    selected_ids = {id(item) for item in selected}
    trace = [
        {
            "parent_nodes": list(parent_nodes),
            "parent_qa_ids": [
                [candidate.qa_uid for candidate in state.steps] for state in states
            ],
            "harmonic_score": float(harmonic),
            "examined": index < beam_width,
            "selected": any(item is selected_item for selected_item in selected),
        }
        for index, item in enumerate(combinations)
        for harmonic, states, _ in [item]
    ]
    return [item[2] for item in selected], trace


def _evidence_union(chains: Sequence[ChainState]) -> tuple[Candidate, ...]:
    evidence: list[Candidate] = []
    seen: set[str] = set()
    for chain in chains:
        for candidate in chain.steps:
            if candidate.qa_uid in seen:
                continue
            seen.add(candidate.qa_uid)
            evidence.append(candidate)
    return tuple(evidence)


def run_panini_ricr(
    decomposed_questions: Sequence[Mapping[str, object]],
    retrieve_and_score: Callable[[str, int], Sequence[Candidate]],
    *,
    original_question: str,
    beam_width: int = 5,
    candidates_per_hop: int = 15,
    multi_dependency_threshold: float = 0.3,
    unique_intermediate_entities: bool = True,
) -> RICRResult:
    """Execute connected-DAG PANINI RICR with deterministic beam pruning."""

    if beam_width <= 0:
        raise ValueError("beam_width must be positive")
    if candidates_per_hop <= 0:
        raise ValueError("candidates_per_hop must be positive")
    if multi_dependency_threshold < 0:
        raise ValueError("multi_dependency_threshold must be non-negative")

    components = identify_retrieval_components(decomposed_questions)
    retrieval_nodes, parents, children, _ = _dependency_maps(decomposed_questions)
    issued_queries: list[str] = []
    retrieval_cache: dict[str, list[Candidate]] = {}

    def retrieve(query: str) -> list[Candidate]:
        if query not in retrieval_cache:
            issued_queries.append(query)
            retrieval_cache[query] = _deduplicate_candidates(
                tuple(retrieve_and_score(query, candidates_per_hop)),
                candidates_per_hop,
            )
        return retrieval_cache[query]

    if not components:
        candidates = retrieve(original_question)
        base = ChainState(steps=(), answers_by_step={}, score=1.0, last_hop_score=1.0)
        retained, pruned = _expand_final([base], 1, candidates, beam_width)
        return RICRResult(
            components=(),
            chains=tuple(retained),
            evidence=_evidence_union(retained),
            issued_queries=tuple(issued_queries),
            fallback=True,
            trace={
                "fallback": True,
                "query": original_question,
                "candidates": [_candidate_summary(candidate) for candidate in candidates],
                "retained_states": [_state_summary(state) for state in retained],
                "pruned_states": [_state_summary(state) for state in pruned],
            },
        )

    final_chains: list[ChainState] = []
    component_traces: list[dict[str, object]] = []
    for component in components:
        component_set = set(component)
        component_parents = {
            node: sorted(parents[node] & component_set) for node in component
        }
        component_children = {
            node: sorted(children[node] & component_set) for node in component
        }
        beams: dict[int, list[ChainState]] = {}
        node_traces: list[dict[str, object]] = []

        for node in component:
            node_parents = component_parents[node]
            parent_trace: list[dict[str, object]] = []
            if not node_parents:
                base_states = [
                    ChainState(
                        steps=(), answers_by_step={}, score=1.0, last_hop_score=1.0
                    )
                ]
            elif len(node_parents) == 1:
                base_states = list(beams.get(node_parents[0], ()))
            else:
                base_states, parent_trace = _combine_parent_beams(
                    node_parents,
                    beams,
                    beam_width,
                    multi_dependency_threshold,
                )

            query_groups: dict[str, list[ChainState]] = {}
            query_errors: list[str] = []
            for state in base_states:
                try:
                    query = instantiate_question(
                        str(decomposed_questions[node - 1].get("question", "")),
                        state.answers_by_step,
                    )
                except KeyError as error:
                    query_errors.append(str(error))
                    continue
                query_groups.setdefault(query, []).append(state)

            is_final = not component_children[node]
            retained_states: list[ChainState] = []
            pruned_states: list[ChainState] = []
            query_trace: list[dict[str, object]] = []
            for query in sorted(query_groups):
                try:
                    candidates = retrieve(query)
                except KeyError as error:
                    if (
                        error.__class__.__name__
                        != "MissingQueryEmbeddingError"
                    ):
                        raise

                    query_errors.append(
                        f"{query}: {error}"
                    )
                    continue

                if is_final:
                    retained, pruned = _expand_final(
                        query_groups[query], node, candidates, beam_width
                    )
                else:
                    retained, pruned = _expand_intermediate(
                        query_groups[query],
                        node,
                        candidates,
                        beam_width,
                        unique_intermediate_entities,
                    )
                retained_states.extend(retained)
                pruned_states.extend(pruned)
                query_trace.append(
                    {
                        "query": query,
                        "base_states": [
                            _state_summary(state) for state in query_groups[query]
                        ],
                        "candidates": [
                            _candidate_summary(candidate) for candidate in candidates
                        ],
                    }
                )

            # Queries produced from different parent states are pooled before the
            # node-level global beam is selected.
            ordered = sorted(retained_states + pruned_states, key=_state_sort_key)
            beams[node] = ordered[:beam_width]
            globally_pruned = ordered[beam_width:]
            node_traces.append(
                {
                    "node": node,
                    "template": str(
                        decomposed_questions[node - 1].get("question", "")
                    ),
                    "parents": node_parents,
                    "children": component_children[node],
                    "final": is_final,
                    "parent_combinations": parent_trace,
                    "query_errors": query_errors,
                    "queries": query_trace,
                    "retained_states": [
                        _state_summary(state) for state in beams[node]
                    ],
                    "pruned_states": [
                        _state_summary(state) for state in globally_pruned
                    ],
                }
            )

        sink_nodes = [node for node in component if not component_children[node]]
        component_final = sorted(
            [state for node in sink_nodes for state in beams.get(node, ())],
            key=_state_sort_key,
        )[:beam_width]
        final_chains.extend(component_final)
        component_traces.append(
            {
                "nodes": list(component),
                "parents": {
                    str(node): component_parents[node] for node in component
                },
                "children": {
                    str(node): component_children[node] for node in component
                },
                "node_traces": node_traces,
                "final_qa_ids": [
                    [candidate.qa_uid for candidate in state.steps]
                    for state in component_final
                ],
            }
        )

    return RICRResult(
        components=tuple(tuple(component) for component in components),
        chains=tuple(final_chains),
        evidence=_evidence_union(final_chains),
        issued_queries=tuple(issued_queries),
        fallback=False,
        trace={
            "fallback": False,
            "components": component_traces,
            "evidence_qa_ids": [
                candidate.qa_uid for candidate in _evidence_union(final_chains)
            ],
        },
    )


def run_linear_ricr(
    decomposed_questions: Sequence[Mapping[str, object]],
    retrieve_and_score: Callable[[str, int], Sequence[Candidate]],
    *,
    beam_width: int = 5,
    candidates_per_hop: int = 15,
) -> list[ChainState]:
    first = next(
        (
            str(row["question"])
            for row in decomposed_questions
            if row.get("requires_retrieval", True)
        ),
        "",
    )
    return list(
        run_panini_ricr(
            decomposed_questions,
            retrieve_and_score,
            original_question=first,
            beam_width=beam_width,
            candidates_per_hop=candidates_per_hop,
        ).chains
    )
