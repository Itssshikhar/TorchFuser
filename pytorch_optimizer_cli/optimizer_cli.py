# optimizer_cli.py
import argparse
import os
import sys
from dotenv import load_dotenv

# Load environment variables from .env file, if it exists
load_dotenv()

# Add the directory containing optimizer_utils to the Python path
# This allows importing optimizer_utils when running optimizer_cli.py directly
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

try:
    from optimizer_utils import (
        profile_pytorch_function,
        analyze_pytorch_profile_data,
        extract_function_source,
        create_pytorch_llm_prompt,
        query_openai_llm,
        query_gemini_llm
    )
except ImportError as e:
    print(f"Error importing optimizer_utils: {e}")
    print("Please ensure optimizer_utils.py is in the same directory as optimizer_cli.py.")
    sys.exit(1)


def main_pytorch():
    parser = argparse.ArgumentParser(
        description="PyTorch to CUDA Converter & Optimizer CLI")
    parser.add_argument("--script_path", required=True,
                        help="Path to the Python script with PyTorch code.")
    parser.add_argument("--target_function", required=True,
                        help="Name of the function within the script to profile and convert to CUDA.")
    parser.add_argument("--output_dir", default="./opt_runs_pytorch",
                        help="Directory to store profiles and logs.")
    parser.add_argument("--llm_provider", default="openai",
                        choices=["openai", "google"], help="LLM provider.")
    parser.add_argument("--model_name", default=None,
                        help="LLM model name (e.g., gpt-4-turbo, gemini-1.5-pro-latest). Defaults based on provider.")
    parser.add_argument("--api_key_env_var", default=None,
                        help="Environment variable for LLM API key (e.g., OPENAI_API_KEY or GOOGLE_API_KEY). Defaults based on provider.")
    parser.add_argument("--no_cuda", action="store_false", dest="profile_cuda",
                        help="Disable CUDA profiling even if available.")
    parser.add_argument("--profile_memory", action="store_true",
                        help="Enable memory profiling with torch.profiler.")
    # TODO: Add argument parsing for script_function_args if needed in a future version
    # parser.add_argument("--script_args", type=str, help="JSON string of arguments for the target function.")

    args = parser.parse_args()

    # Determine default API key env var if not provided
    if args.api_key_env_var is None:
        if args.llm_provider == "openai":
            args.api_key_env_var = "OPENAI_API_KEY"
        elif args.llm_provider == "google":
            args.api_key_env_var = "GOOGLE_API_KEY"
        else:
            print(f"Error: Unknown LLM provider '{args.llm_provider}'. Cannot determine default API key variable.")
            sys.exit(1)

    # Determine default model name if not provided
    if args.model_name is None:
        if args.llm_provider == "openai":
            args.model_name = "gpt-4-turbo"  # Or another suitable default like gpt-4
            print(f"Using default OpenAI model: {args.model_name}")
        elif args.llm_provider == "google":
            args.model_name = "gemini-2.0-flash"  # Using gemini-2.0-flash as default
            print(f"Using default Google model: {args.model_name}")
        # No else needed as provider choice is constrained by argparse

    api_key = os.getenv(args.api_key_env_var)
    if not api_key:
        print(f"Error: API key not found in environment variable '{args.api_key_env_var}'")
        print("Please set the environment variable before running.")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    run_count = 0
    # Use the absolute path for consistency, especially with dynamic imports
    current_script_path = os.path.abspath(args.script_path)
    if not os.path.isfile(current_script_path):
        print(f"Error: Script file not found at '{current_script_path}'.")
        sys.exit(1)

    while True:
        run_count += 1
        print(f"\n--- Iteration {run_count} (PyTorch to CUDA Conversion) ---")

        # --- 1. Profile PyTorch function ---
        # For MVP, script_function_args is None. User must design target_function to be callable without args.
        # A future version could parse --script_args JSON string here.
        print(f"Using script: {current_script_path}")
        prof_obj, cpu_table, cuda_table, trace_file = profile_pytorch_function(
            current_script_path,
            args.target_function,
            args.output_dir,
            run_count,
            profile_cuda=args.profile_cuda,
            profile_memory=args.profile_memory,
            script_function_args=None  # Placeholder for future arg parsing
        )

        if not prof_obj:
            print("Profiling failed. Check logs above. Exiting iteration.")
            action = input("Options: (r)etry profiling, (e)xit? ").lower()
            if action == 'e':
                break
            else:
                continue  # Retry loop

        # --- 2. Analyze PyTorch Profile ---
        bottleneck_details, _ = analyze_pytorch_profile_data(
            prof_obj, cpu_table, cuda_table)

        # --- 3. Extract Source of the target_function ---
        source_extraction_details = {
            "file_path": current_script_path, "function_name": args.target_function}
        function_source = extract_function_source(source_extraction_details)

        if not function_source:
            print(f"Warning: Could not automatically extract source for '{args.target_function}'. LLM suggestion will lack direct code context.")
        else:
            print(f"\n--- Original Source for '{args.target_function}' ---")
            print("```python")
            print(function_source)
            print("```")

        # --- 4. LLM Interaction ---
        llm_suggestion = None
        if function_source:  # Only query LLM if we have source code
            llm_prompt = create_pytorch_llm_prompt(
                args.target_function, function_source, bottleneck_details)

            print("\n--- Sending to LLM for CUDA Implementation ---")
            # print("DEBUG: LLM Prompt:\n", llm_prompt) # Uncomment for debugging

            if args.llm_provider == "openai":
                llm_suggestion = query_openai_llm(
                    llm_prompt, api_key, args.model_name)
            elif args.llm_provider == "google":
                llm_suggestion = query_gemini_llm(
                    llm_prompt, api_key, args.model_name)
            else:
                print(f"Error: LLM provider '{args.llm_provider}' not implemented.")
                llm_suggestion = None

            if not llm_suggestion:
                print("Failed to get suggestion from LLM. This could be due to an API issue, incorrect API key, or unsupported provider.")
            else:
                print("\n--- LLM Suggested CUDA Implementation ---")
                print("```cuda")
                print(llm_suggestion)
                print("```")
                # Save the CUDA code to a file for easier access
                cuda_file_path = os.path.join(args.output_dir, f"{args.target_function}_cuda_implementation_{run_count}.cu")
                try:
                    with open(cuda_file_path, 'w') as f:
                        f.write(llm_suggestion)
                    print(f"\nCUDA implementation saved to: {cuda_file_path}")
                except Exception as e:
                    print(f"Warning: Could not save CUDA implementation to file: {e}")
                
                print("\nReminder: Please review the CUDA implementation and compile it using nvcc before testing.")
        else:
            print("Skipping LLM interaction as source code for the target function was not available.")

        # --- 5. Iterate ---
        while True:
            action = input("\nOptions: (r)eprofile current script, (e)xit? ").lower()
            if action in ['r', 'e']:
                break
            else:
                print("Invalid option.")

        if action == 'e':
            break
        elif action == 'r':
            # Check if the user might have modified the script
            # (We don't automatically apply changes in MVP)
            print(f"\nReprofiling '{current_script_path}'. You may want to compare the CUDA implementation with the original PyTorch code.")
            continue

    print("\nOptimizer session finished.")


if __name__ == "__main__":
    # Add basic check for presence of PyTorch
    try:
        import torch
    except ImportError:
        print("Error: PyTorch is not installed. Please install it (`pip install torch`) and required dependencies from requirements.txt")
        sys.exit(1)
    main_pytorch()
