#!/bin/bash
#
# Run AlfWorld trajectory collection with LLM integration
# Usage: ./run_alfworld.sh [num_tasks] [steps] [batch_size]
#

# Configuration
API_ENDPOINT="${OAI_ENDPOINT:-}"
API_KEY="${OAI_KEY:-}"
NUM_TASKS="${1:-1}"
MAX_STEPS="${2:-10}"
BATCH_SIZE="${3:-1}"

# Set environment
export OAI_ENDPOINT="$API_ENDPOINT"
export OAI_KEY="$API_KEY"

# Display configuration
echo "========================================"
echo "AlfWorld Trajectory Collection"
echo "========================================"
echo "API Endpoint: ${API_ENDPOINT:0:30}..."
echo "Tasks: $NUM_TASKS"
echo "Steps: $MAX_STEPS"
echo "Batch: $BATCH_SIZE"
echo ""

# Run trajectory collection
cd "$(dirname "$0")/.." || exit 1
python test/alfworld_rollout.py --num_tasks "$NUM_TASKS" --steps "$MAX_STEPS" --batch "$BATCH_SIZE"

# Check for generated trajectories
if [ -d "trajectories" ]; then
    echo ""
    echo "Generated trajectories:"
    ls -lh trajectories/*.json 2>/dev/null | tail -5
fi