#!/bin/sh
# ==============================================================================
# This script was entirely generated on June 9, 2026, by Google AI
# under the guidance of Mathieu Baudier.
# 
# This library is free software; you can redistribute it and/or modify it under 
# the terms of the GNU Lesser General Public License as published by the Free 
# Software Foundation; either version 2.1 of the License, or (at your option) 
# any later version.
#
# This library is distributed in the hope that it will be useful, but WITHOUT 
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS 
# FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more 
# details.
# ==============================================================================

# 1. Dynamically locate the directory where this shell script resides (sdk/)
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

# 2. Define standard test payload data structures
TEST_MESSAGES='[{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "Hello!"}]'
TEST_TOOLS='[{"type": "function", "function": {"name": "test_tool", "description": "A dummy testing tool", "parameters": {"type": "object", "properties": {"param1": {"type": "string", "description": "Required text parameter"}}, "required": ["param1"]}}}]'

echo "=== STARTING BULK JINJA TEMPLATE VALIDATION ==="

# 3. Target the templates directory (native/ is at the same level as sdk/)
for template_file in "$SCRIPT_DIR"/../native/tp/llama.cpp/models/templates/*.jinja; do
    filename=$(basename "$template_file")
    
    # Skip templates we do not target for compatibility
    if [ "$filename" = "fireworks-ai-llama-3-firefunction-v2.jinja" ]; then
        echo "⏭️  [SKIP] $filename (not targeted)"
        continue
    fi
    
    # 4. Execute the Python script using its location relative to sdk/ (java/examples/)
    python3 "$SCRIPT_DIR/java/examples/jjml-jinja.py" \
      add_generation_prompt=true \
      enable_thinking=true \
      messages="$TEST_MESSAGES" \
      tools="$TEST_TOOLS" \
      < "$template_file" > /dev/null 2> "$SCRIPT_DIR/error.log"
    
    exit_code=$?
    
    # 5. Assert results neutrally without descriptive exaggeration
    if [ $exit_code -eq 0 ]; then
        echo "✅ [PASS] $filename rendered without crashing."
    elif [ "$filename" = "google-gemma-2-2b-it.jinja" ] && grep -q "JinjaTemplateError: System role not supported" "$SCRIPT_DIR/error.log"; then
        echo "✅ [PASS] $filename raised expected system role exception."
    else
        echo "❌ [FAIL] $filename crashed during execution!"
        echo "--- Error Traceback for $filename ---"
        cat "$SCRIPT_DIR/error.log"
        echo "----------------------------------------"
    fi
done

# Clean up the temporary log file inside the script directory
rm -f "$SCRIPT_DIR/error.log"
echo "=== TEMPLATE VALIDATION COMPLETE ==="
