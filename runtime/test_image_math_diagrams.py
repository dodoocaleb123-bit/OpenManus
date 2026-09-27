from pathlib import Path

from app.api.routes import image_answer_needs_completion


def test_image_math_answer_requires_a_conclusion():
    half = "Based on the geometric solid, the dimensions are AG=2x and GF=4x. To"
    assert image_answer_needs_completion(half, None, math_problem=True)

    complete = "### Therefore, the correct answer is:\n$$\\boxed{x=4}$$"
    assert not image_answer_needs_completion(complete, "stop", math_problem=True)
    assert image_answer_needs_completion("A partial answer", "length", math_problem=True)


def test_image_math_detector_does_not_require_math_markers_for_description():
    assert not image_answer_needs_completion("The image shows a prism.", "stop", math_problem=False)


def test_web_client_contains_mermaid_renderer_and_image_budget_controls():
    html = Path("web/index.html").read_text(encoding="utf-8")
    assert "mermaid@11" in html
    assert "renderDiagrams" in html
    assert "mermaid-source" in html
    routes = Path("app/api/routes.py").read_text(encoding="utf-8")
    assert "PLATFORM_IMAGE_MAX_TOKENS" in routes
    assert "solve every visible subquestion" in routes
