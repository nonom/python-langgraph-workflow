import argparse
import json
import os
import sys
from typing import Literal, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field


class LogState(TypedDict, total=False):
    log: str
    type: str
    priority: str
    subject: str
    category: str
    classification_reason: str
    analysis: str
    probable_cause: str
    classification_confidence: float
    analysis_confidence: float
    recommended_next_steps: list[str]
    validation: str
    validation_passed: bool
    final_response: dict


class Classification(BaseModel):
    category: Literal[
        "database",
        "network",
        "authentication",
        "configuration",
        "performance",
        "unknown",
    ]
    confidence: float = Field(ge=0, le=100, description="Confidence percentage")
    reason: str = Field(description="Brief reason based on the log evidence")


class Analysis(BaseModel):
    probable_cause: str
    analysis: str
    confidence: float = Field(ge=0, le=100, description="Confidence percentage")
    recommended_next_steps: list[str]


class Validation(BaseModel):
    valid: bool = Field(
        description=(
            "True only when the diagnosis is supported by the log; false when it "
            "includes unsupported assumptions"
        )
    )
    feedback: str


class FinalResponse(BaseModel):
    category: str
    probable_cause: str
    confidence: float = Field(ge=0, le=100, description="Confidence percentage")
    recommended_next_steps: list[str]


MODEL = os.getenv("OLLAMA_MODEL", "granite3-dense:2b")
llm = ChatOllama(
    model=MODEL,
    temperature=0.1,
    top_k=50,
    repeat_penalty=1.05,
)


def classify_log(state: LogState) -> dict:
    classifier = llm.with_structured_output(Classification)
    result = classifier.invoke(
        [
            SystemMessage(
                "Classify the log using only its evidence. Choose the category, "
                "confidence, and give a brief reason. Use unknown when no domain "
                "is clear. If the message is malformed or mostly noise, use "
                "unknown. "
                "Confidence: 0-100."
            ),
            HumanMessage(state["log"]),
        ]
    )
    return {
        "category": result.category,
        "classification_confidence": result.confidence,
        "classification_reason": result.reason,
    }


def analyse_log(state: LogState) -> dict:
    analyst = llm.with_structured_output(Analysis)
    result = analyst.invoke(
        [
            SystemMessage(
                "Analyse the log. Give the probable cause and next steps supported "
                "by the log. Confidence: 0-100."
            ),
            HumanMessage(state["log"]),
        ]
    )
    return {
        "analysis": result.analysis,
        "probable_cause": result.probable_cause,
        "analysis_confidence": result.confidence,
        "recommended_next_steps": result.recommended_next_steps,
    }


def validate_analysis(state: LogState) -> dict:
    validator = llm.with_structured_output(Validation)
    result = validator.invoke(
        [
            SystemMessage(
                "Review the diagnosis against the log. Check that the cause and "
                "next steps are supported by the evidence."
            ),
            HumanMessage(
                content=json.dumps(
                    {
                        "log": state["log"],
                        "category": state.get("category"),
                        "analysis": state.get("analysis"),
                        "probable_cause": state.get("probable_cause"),
                        "confidence": state.get("analysis_confidence"),
                        "recommended_next_steps": state.get("recommended_next_steps"),
                    }
                )
            ),
        ]
    )
    return {
        "validation": result.feedback,
        "validation_passed": result.valid,
    }


def final_response(state: LogState) -> dict:
    result = FinalResponse(
        category=state["category"],
        probable_cause=state["probable_cause"],
        confidence=state["analysis_confidence"],
        recommended_next_steps=state["recommended_next_steps"],
    )
    return {"final_response": result.model_dump()}


def rejected_response(state: LogState) -> dict:
    return {
        "final_response": {
            "status": "rejected",
            "reason": state.get(
                "validation",
                state.get(
                    "classification_reason",
                    "The log could not be classified.",
                ),
            ),
        }
    }


def route_after_classification(state: LogState) -> str:
    return "rejected_response" if state["category"] == "unknown" else "analyse_log"


def route_after_validation(state: LogState) -> str:
    return "final_response" if state["validation_passed"] else "rejected_response"


def build_graph():
    graph = StateGraph(LogState)
    graph.add_node("classify_log", classify_log)
    graph.add_node("analyse_log", analyse_log)
    graph.add_node("validate_analysis", validate_analysis)
    graph.add_node("final_response", final_response)
    graph.add_node("rejected_response", rejected_response)
    graph.add_edge(START, "classify_log")
    graph.add_conditional_edges(
        "classify_log",
        route_after_classification,
        {
            "analyse_log": "analyse_log",
            "rejected_response": "rejected_response",
        },
    )
    graph.add_edge("analyse_log", "validate_analysis")
    graph.add_conditional_edges(
        "validate_analysis",
        route_after_validation,
        {
            "final_response": "final_response",
            "rejected_response": "rejected_response",
        },
    )
    graph.add_edge("final_response", END)
    graph.add_edge("rejected_response", END)
    return graph.compile()


def read_log(log_file: str | None, log_input: str | None) -> str:
    if log_input is not None:
        content = log_input
    elif log_file:
        with open(log_file, encoding="utf-8") as file:
            content = file.read()
    elif not sys.stdin.isatty():
        content = sys.stdin.read()
    else:
        raise SystemExit("Usage: python main.py example.log")

    lines = [line.strip() for line in content.splitlines() if line.strip()]
    if not lines:
        raise SystemExit("Expected one log line, received an empty input.")
    return lines[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diagnose an application log")
    parser.add_argument("log_file", nargs="?", help="Path to a log file")
    parser.add_argument("--input", dest="log_input", help="Log line to analyse")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show each node output")
    args = parser.parse_args()

    initial_state = {"log": read_log(args.log_file, args.log_input)}
    graph = build_graph()

    if args.verbose:
        green = "\033[32m" if sys.stdout.isatty() else ""
        reset = "\033[0m" if green else ""
        print(f"\n{green}Log input:{reset} {initial_state['log']}")
        node_models = {
            "classify_log": MODEL,
            "analyse_log": MODEL,
            "validate_analysis": MODEL,
            "final_response": "Python",
            "rejected_response": "Python",
        }
        for update in graph.stream(initial_state, stream_mode="updates"):
            for node, output in update.items():
                print(f"\n{green}[{node} · {node_models[node]}]{reset}")
                print(json.dumps(output, indent=2, ensure_ascii=False))
    else:
        result = graph.invoke(initial_state)
        print(json.dumps(result["final_response"], indent=2, ensure_ascii=False))
