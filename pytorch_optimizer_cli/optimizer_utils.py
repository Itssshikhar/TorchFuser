import torch
import torch.profiler
import importlib.util
import os
import inspect
import traceback
import openai
import google.generativeai as genai


def profile_pytorch_function(script_path, target_function_name, output_dir, run_id,
                             profile_cuda=True, profile_memory=False,
                             script_function_args=None):  # script_function_args could be a dict
    """Profiles the target PyTorch function using torch.profiler."""
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    trace_file_path = os.path.join(output_dir, f"pytorch_trace_{run_id}.json")
    tb_trace_dir = os.path.join(output_dir, f"tb_trace_{run_id}")

    try:
        # Dynamically import the user's script and target function
        module_name = os.path.splitext(os.path.basename(script_path))[
            0] + f"_run_{run_id}"  # Make module name unique per run
        spec = importlib.util.spec_from_file_location(module_name, script_path)
        if spec is None or spec.loader is None:
            print(f"Error: Could not create module spec for '{script_path}'.")
            return None, None, None, None
        user_module = importlib.util.module_from_spec(spec)
        # TODO: Handle potential sys.modules caching issues if the same script is reloaded multiple times.
        # A safer approach might involve copying the script to a temp location or using unique module names.
        spec.loader.exec_module(user_module)

        if not hasattr(user_module, target_function_name):
            print(f"Error: Function '{target_function_name}' not found in '{script_path}'.")
            return None, None, None, None

        target_func = getattr(user_module, target_function_name)

        activities = [torch.profiler.ProfilerActivity.CPU]
        use_cuda = profile_cuda and torch.cuda.is_available()
        if use_cuda:
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        else:
            if profile_cuda:
                print(
                    "Warning: CUDA profiling requested but CUDA is not available. Profiling CPU only.")
            profile_cuda = False  # Ensure flag reflects reality

        print(f"Profiling '{target_function_name}' from '{script_path}'...")
        print(f"Activities: {activities}, Profile Memory: {profile_memory}")

        # Ensure tensorboard directory exists for handler
        # os.makedirs(tb_trace_dir, exist_ok=True)

        with torch.profiler.profile(
            activities=activities,
            record_shapes=True,  # Useful for identifying tensor shape issues
            profile_memory=profile_memory,
            with_stack=True,  # Helps map to Python source, crucial for us
            # on_trace_ready=torch.profiler.tensorboard_trace_handler(tb_trace_dir) # Optional: For TensorBoard
        ) as prof:
            # Execute the target function
            if script_function_args:
                _ = target_func(**script_function_args)
            else:
                _ = target_func()  # User must ensure this is callable

        print(f"Profiling complete.")

        # Option 1: Export trace (can be useful for detailed analysis)
        try:
            prof.export_chrome_trace(trace_file_path)
            print(f"Profiler trace exported to: {trace_file_path}")
        except Exception as trace_e:
            print(f"Warning: Could not export Chrome trace: {trace_e}")
            trace_file_path = None

        # Option 2: Get key averages for immediate analysis
        key_avg_table_cpu = prof.key_averages(group_by_input_shape=True, group_by_stack_n=5).table(
            sort_by="self_cpu_time_total", row_limit=15)

        key_avg_table_cuda = None
        # Check if any CUDA events were actually recorded before trying to sort by CUDA time
        cuda_events_exist = any(
            event.device_type == torch.autograd.DeviceType.CUDA for event in prof.events())
        if use_cuda and cuda_events_exist:
            key_avg_table_cuda = prof.key_averages(group_by_input_shape=True, group_by_stack_n=5).table(
                sort_by="self_cuda_time_total", row_limit=15)

        return prof, key_avg_table_cpu, key_avg_table_cuda, trace_file_path

    except Exception as e:
        print(f"Error during PyTorch profiling of '{target_function_name}': {e}")
        traceback.print_exc()
        return None, None, None, None


def analyze_pytorch_profile_data(prof, key_avg_cpu_table_str, key_avg_cuda_table_str, top_n=3):
    """Analyzes the profiler output tables."""
    print("\n--- PyTorch Profiler CPU Key Averages (Top) ---")
    print(key_avg_cpu_table_str)
    if key_avg_cuda_table_str:
        print("\n--- PyTorch Profiler CUDA Key Averages (Top) ---")
        print(key_avg_cuda_table_str)

    # For the LLM, we need to identify a Python code segment.
    # The 'Stack' info in key_averages is key.
    # This MVP passes the summary tables and asks the LLM to analyze the whole function.
    # A more advanced version would parse the stack traces from prof.key_averages()
    # to find the specific lines associated with top operators.

    bottleneck_details = {
        "profiler_summary_cpu": key_avg_cpu_table_str,
        "profiler_summary_cuda": key_avg_cuda_table_str if key_avg_cuda_table_str else "N/A",
        "primary_focus": "See profiler tables for specific operator times. The LLM should analyze the provided function source in light of these tables."
    }
    # Placeholder for top event info string - could be enhanced by parsing tables/events
    top_event_info = "Could not automatically determine top operator for LLM. Please refer to tables above."

    return bottleneck_details, top_event_info


def extract_function_source(source_details):
    """Extracts the source code of a function from a file."""
    file_path = source_details.get("file_path")
    function_name = source_details.get("function_name")

    if not file_path or not function_name:
        print("Error: File path or function name missing for source extraction.")
        return None

    if not os.path.exists(file_path):
        print(f"Error: Source file not found: {file_path}")
        return None

    try:
        # Dynamically import to use inspect (safer than reading/parsing manually)
        # Use a unique module name to avoid conflicts if called multiple times
        module_name = f"source_extractor_{os.path.splitext(os.path.basename(file_path))[0]}_{function_name}"
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            print(f"Error: Could not create module spec for source extraction from '{file_path}'.")
            return None
        source_module = importlib.util.module_from_spec(spec)
        # This executes the user's script! Be careful.
        spec.loader.exec_module(source_module)

        target_func = getattr(source_module, function_name, None)
        if target_func and inspect.isfunction(target_func):
            source_code = inspect.getsource(target_func)
            return source_code
        else:
            # Fallback or alternative: try parsing the file if import fails or it's not a function
            print(f"Warning: Could not find function '{function_name}' via inspect in '{file_path}'. Trying manual parse (less reliable).")
            # Basic manual parsing (less reliable, might grab wrong block)
            with open(file_path, 'r') as f:
                lines = f.readlines()
            start_line = -1
            source_lines = []
            in_func = False
            indent_level = -1
            for i, line in enumerate(lines):
                stripped_line = line.strip()
                current_indent = len(line) - len(line.lstrip(' '))
                if stripped_line.startswith(f"def {function_name}("):
                    start_line = i
                    in_func = True
                    indent_level = current_indent
                    source_lines.append(line)
                elif in_func:
                    if current_indent > indent_level or not stripped_line:  # Part of function or empty line
                        source_lines.append(line)
                    elif current_indent <= indent_level and stripped_line:  # Dedented, end of function
                        break
            if source_lines:
                return "".join(source_lines)
            else:
                print(f"Error: Could not find definition for '{function_name}' in '{file_path}'.")
                return None

    except Exception as e:
        print(f"Error extracting source for '{function_name}' from '{file_path}': {e}")
        traceback.print_exc()
        return None


def create_pytorch_llm_prompt(target_function_name, function_source, bottleneck_details):
    """Constructs the prompt for the LLM to optimize the PyTorch function."""
    prompt = f"""You are an expert CUDA optimization engineer.
The Python function {target_function_name}, which contains PyTorch operations, was profiled.
Your goal is to analyze its source code in conjunction with the provided profiler summary and suggest a CUDA implementation that would be more efficient.

Original Function Code ({target_function_name}):
```python
{function_source}
```

Profiler Summary:
CPU Operator Times (Self):
```
{bottleneck_details['profiler_summary_cpu']}
```
CUDA Operator Times (Self, if applicable):
```
{bottleneck_details['profiler_summary_cuda']}
```

{bottleneck_details['primary_focus']}

Please analyze the function code and the profiler summary.
Instead of optimizing the PyTorch code, rewrite it as a CUDA implementation (.cu file) to maximize performance.

Focus on:
- Identifying bottlenecks in the original PyTorch code
- Creating proper CUDA kernel implementations for computationally intensive parts
- Optimizing memory access patterns for coalesced access
- Using shared memory when applicable to reduce global memory access
- Optimizing thread and block dimensions for your kernels
- Minimizing CPU-GPU data transfers
- Utilizing CUDA streams for parallelization when possible
- Implementing proper error handling for CUDA operations

Provide complete CUDA code that implements the same functionality as the original function. Include:
1. CUDA kernel definitions (.cu file)
2. Host functions to call the kernels
3. Memory management code (allocation, transfers, freeing)
4. Brief comments explaining your optimization strategy for each kernel

The output should be a complete, runnable CUDA implementation wrapped in triple backticks ```cuda ... ```.
Include comments in the code describing the key optimizations you've made and how they improve upon the original.
"""
    return prompt

# --- LLM Interaction Functions (Placeholders/Basic Implementations) ---


def query_openai_llm(prompt, api_key, model_name="gpt-4"):
    """Sends prompt to OpenAI API and returns the response."""
    print(f"Querying OpenAI model: {model_name}")
    try:
        if not api_key:
            print("Error: OpenAI API key is empty or invalid.")
            return None

        openai.api_key = api_key
        response = openai.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful programming assistant specialized in code optimization."},
                {"role": "user", "content": prompt}
            ]
        )
        # Add error handling for API response structure
        if not hasattr(response, 'choices') or len(response.choices) == 0:
            print("Error: Unexpected OpenAI API response format.")
            return None
            
        suggestion = response.choices[0].message.content.strip()

        # Extract code block if wrapped
        if suggestion.startswith("```cuda") and suggestion.endswith("```"):
            suggestion = suggestion[len("```cuda"): -len("```")].strip()
        elif suggestion.startswith("```python") and suggestion.endswith("```"):
            suggestion = suggestion[len("```python"): -len("```")].strip()
        elif suggestion.startswith("```") and suggestion.endswith("```"):
            suggestion = suggestion[len("```"): -len("```")].strip()

        return suggestion
    except Exception as e:
        print(f"Error querying OpenAI: {e}")
        traceback.print_exc()
        return None


def query_gemini_llm(prompt, api_key, model_name="gemini-2.0-flash"):
    """Sends prompt to Google Gemini API and returns the response."""
    print(f"Querying Google Gemini model: {model_name}")
    try:
        if not api_key:
            print("Error: Google Gemini API key is empty or invalid.")
            return None
            
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)

        # Add more robust error handling and check response structure
        if not hasattr(response, 'parts') or not response.parts:
            print("Error: Gemini response did not contain expected content.")
            return None
            
        suggestion = response.text.strip()
        # Extract code block if wrapped
        if suggestion.startswith("```cuda") and suggestion.endswith("```"):
            suggestion = suggestion[len("```cuda"): -len("```")].strip()
        elif suggestion.startswith("```python") and suggestion.endswith("```"):
            suggestion = suggestion[len("```python"): -len("```")].strip()
        elif suggestion.startswith("```") and suggestion.endswith("```"):
            suggestion = suggestion[len("```"): -len("```")].strip()
        return suggestion
    except Exception as e:
        print(f"Error querying Google Gemini: {e}")
        traceback.print_exc()
        return None
