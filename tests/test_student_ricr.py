from panini_course.ricr import Candidate, run_panini_ricr


def qa(uid, answers, score, *, ids=(), document_id='', gsw_file=''):
    return Candidate(
        qa_uid=uid,
        answer_names=tuple(answers),
        score=score,
        question=uid,
        answer_ids=tuple(ids),
        document_id=document_id,
        metadata={'gsw_file': gsw_file},
    )


def test_harmonic_threshold_keeps_best_tuple_when_every_tuple_is_weak():
    plan = [
        {'question': 'Root A', 'requires_retrieval': True},
        {'question': 'Root B', 'requires_retrieval': True},
        {'question': 'Join <ENTITY_Q1> and <ENTITY_Q2>', 'requires_retrieval': True},
    ]
    table = {
        'Root A': [qa('a', ['Ada'], -0.9, ids=['doc-a::e1'])],
        'Root B': [qa('b', ['Grace'], -0.8, ids=['doc-b::e1'])],
        'Join Ada and Grace': [qa('c', ['answer'], 0.8)],
    }
    result = run_panini_ricr(
        plan,
        lambda query, top_k: table[query][:top_k],
        original_question='Join them',
        beam_width=2,
        multi_dependency_threshold=0.3,
    )
    assert 'Join Ada and Grace' in result.issued_queries
    assert result.chains


def test_raw_local_answer_ids_from_different_documents_do_not_collapse():
    plan = [
        {'question': 'Find person', 'requires_retrieval': True},
        {'question': 'Finish <ENTITY_Q1>', 'requires_retrieval': True},
    ]
    table = {
        'Find person': [
            qa('a', ['Ada'], 0.9, ids=['e1'], document_id='doc-a', gsw_file='a.json'),
            qa('b', ['Grace'], 0.8, ids=['e1'], document_id='doc-b', gsw_file='b.json'),
        ],
        'Finish Ada': [qa('fa', ['1843'], 0.9)],
        'Finish Grace': [qa('fg', ['1952'], 0.8)],
    }
    result = run_panini_ricr(
        plan,
        lambda query, top_k: table[query][:top_k],
        original_question='Finish',
        beam_width=2,
    )
    assert {chain.steps[0].qa_uid for chain in result.chains} == {'a', 'b'}


def test_final_hop_preserves_two_qa_records_with_the_same_answer():
    plan = [
        {'question': 'Find person', 'requires_retrieval': True},
        {'question': 'Finish <ENTITY_Q1>', 'requires_retrieval': True},
    ]
    table = {
        'Find person': [qa('root', ['Ada'], 0.9, ids=['doc::ada'])],
        'Finish Ada': [
            qa('final-a', ['1843'], 0.9),
            qa('final-b', ['1843'], 0.8),
        ],
    }
    result = run_panini_ricr(
        plan,
        lambda query, top_k: table[query][:top_k],
        original_question='Finish',
        beam_width=2,
    )
    assert {chain.steps[-1].qa_uid for chain in result.chains} == {'final-a', 'final-b'}


def test_trace_contains_dependencies_queries_pruning_and_evidence_ids():
    plan = [
        {'question': 'Root', 'requires_retrieval': True},
        {'question': 'Finish <ENTITY_Q1>', 'requires_retrieval': True},
    ]
    table = {
        'Root': [
            qa('root-a', ['Ada'], 0.9, ids=['doc::ada']),
            qa('root-b', ['Grace'], 0.8, ids=['doc::grace']),
        ],
        'Finish Ada': [qa('answer-a', ['1843'], 0.9)],
        'Finish Grace': [qa('answer-b', ['1952'], 0.8)],
    }
    result = run_panini_ricr(
        plan,
        lambda query, top_k: table[query][:top_k],
        original_question='When?',
        beam_width=2,
    )
    assert result.trace['components'][0]['parents'] == {'1': [], '2': [1]}
    assert set(result.trace['evidence_qa_ids']) == {
        'root-a', 'root-b', 'answer-a', 'answer-b'
    }
