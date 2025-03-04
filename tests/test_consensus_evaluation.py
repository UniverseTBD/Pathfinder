# tests/test_consensus_evaluation.py

from unittest.mock import Mock, patch

import pytest

from src.consensus_evaluation import (
    OverallConsensusEvaluation,
    evaluate_overall_consensus,
)


@pytest.fixture
def mock_consensus_response():
    """
    Provide a mock OverallConsensusEvaluation response that
    the instructor-patched openai client might return.
    """
    return OverallConsensusEvaluation(
        rewritten_statement="The query as a statement",
        consensus="Moderate Agreement Between Abstracts and Query",
        explanation="Explanation about the consensus in up to six sentences.",
        relevance_score=0.75,
    )


@patch("src.consensus_evaluation.get_instructor_client")
def test_evaluate_overall_consensus(mock_get_instructor_client, mock_consensus_response):
    """
    Test that evaluate_overall_consensus returns a valid OverallConsensusEvaluation
    object, given a list of abstracts, by mocking out the instructor client.
    """
    # Mock the instructor client instance
    mock_instructor_client = Mock()
    mock_get_instructor_client.return_value = mock_instructor_client
    
    # Mock the .chat.completions.create(...) method to return our mock response
    mock_instructor_client.chat.completions.create.return_value = mock_consensus_response

    query = "What is the mass of typical dark matter halos?"
    abstracts = [
        "Abstract about dark matter and galaxy formation...",
        "Another abstract focusing on halo mass distributions...",
    ]

    # Call the function under test
    result = evaluate_overall_consensus(query, abstracts)

    # Verify we invoked the instructor client with the correct structure
    assert (
        mock_instructor_client.chat.completions.create.called
    ), "Expected the instructor client to be called with .chat.completions.create(...)"

    # Check that we got a valid OverallConsensusEvaluation
    assert isinstance(
        result, OverallConsensusEvaluation
    ), f"Expected an OverallConsensusEvaluation, got {type(result)}"
    assert result.consensus == "Moderate Agreement Between Abstracts and Query"
    assert result.explanation.startswith("Explanation about the consensus")
    assert result.relevance_score == 0.75
