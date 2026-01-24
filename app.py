#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import sys
from langchain_core.messages import HumanMessage
from graph.builder import build_graph
from utils.context import load_context
from llm_vllm import build_llm
from utils.misc import yesno_to_bool

print("======= 🚀 Loading the model. This may take a moment =======")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_data', type=str, 
                        default='/projects/f_wj183_1/work/xutao/2025_epi_LLM/simple_rag/data/',
                        help='Path to directory containing data.json and column.json')
    parser.add_argument('--path_to_output', type=str, 
                        default='/projects/f_wj183_1/work/xutao/2025_epi_LLM/simple_rag/llm_output',
                        help='Path for output files'),
    parser.add_argument('--db_path', type=str, 
                        default='/projects/f_wj183_1/work/xutao/2025_epi_LLM/RePORTAI_db/agent_memory.db',
                        help='Path for memory database')
    parser.add_argument('--model_name', type=str, default='meta-llama/Llama-3.1-8B-Instruct')
    parser.add_argument('--temperature', type=float, default=0.1)
    parser.add_argument('--top_p', type=float, default=0.9)
    return parser.parse_args()

def main():
    args = get_args()
    
    # Paths
    data_path = f'{args.path_to_data}/data.json'
    schema_path = f'{args.path_to_data}/column.json'

    # Load Data
    try:
        df, schema = load_context(data_path, schema_path)
    except FileNotFoundError as e:
        print(f"Warning: {e}. Proceeding without dataset/schema.")
        df, schema = None, None
    # Preprocessing
    # for col in ["is_smoke", "ever_had_active_tb", "is_hiv_positive", "has_diabetes"]:
    #     if col in df.columns:
    #         df[col] = df[col].map(yesno_to_bool)

    # Build LLM
    model_name = args.model_name
    db_path = args.db_path
    base_url = 'http://localhost:8000/v1'
    temperature = args.temperature
    top_p = args.top_p
    llm = build_llm(model_name, temperature, top_p, base_url)
    app = build_graph(llm, df, schema, db_path=db_path)
    thread_id = "chat-session-1"
    print("Biostats Code Agent (Streaming Mode)")
    print("Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        state = {
            "messages": [HumanMessage(content=user_input)],
            "generated_code": None,
            "output": None,
            "qa_response": None,
            "error": None,
            "next_action": None,
            "last_action": None,
            "planner_mode": "hybrid",
            "observations": [],
            "orchestrator": {
                "tool_results": [],
            },
            "agents": {
                "executor": {"run_status": "idle"},
                "human_review": {"before_run_decision": None, "final_decision": None},
            },
            "meta": {},
        }
        config = {"configurable": {"thread_id": thread_id}}

        final_state = None
        print("\nAgent:")
        for event in app.stream(state, config=config):
            node_name = list(event.keys())[0]
            print(f"\n--- NODE: {node_name} ---")
            # Some nodes emit None (e.g. checkpoint nodes)
            node_state = event[node_name]
            if not isinstance(node_state, dict):
                continue
            final_state = node_state
            # Checkpoint nodes
            if node_name.startswith("human_review"):
                code = node_state.get("generated_code")

                if code:
                    print(f"\n--- HUMAN CHECKPOINT {node_name} ---")
                    print("Generated code:\n")
                    print(code)
                    print("\nApprove the code? (y/n/edit)")

                    action = input("> ")

                    if action == "y":
                        continue   # resume execution
                    elif action == "edit":
                        edited = input("Paste updated code:\n")
                        node_state["generated_code"] = edited
                        # re-invoke starting from modified node state
                        final_state = app.invoke(node_state, config=config)
                        break
                    else:
                        print("Execution stopped.")
                        # return

        # Print final answer
        if final_state:
            if final_state.get("agents", {}).get("executor", {}).get("run_status") == "ok":
                print(final_state["output"])
            else:
                print("Error:", final_state.get("error"))

if __name__ == "__main__":
    main()
