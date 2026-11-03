def test_agent_modules_import():
    from travel_agent import agent, guardrails, prompts

    assert agent is not None
    assert guardrails is not None
    assert prompts is not None