#!/bin/bash

# ============================================================================
# Script avvio Phase 1 Training con ottimizzazioni memoria CUDA
# ============================================================================
#
# Questo script lancia il training con:
# - Batch size ridotto: 32 → 16
# - Video frames ridotti: 150 → 100
# - Text max length ridotto: 512 → 256
# - Mixed Precision Training (AMP) abilitato
# - PYTORCH_CUDA_ALLOC_CONF per evitare frammentazione memoria
#
# Questo riduce il consumo di memoria GPU di ~40-50% rispetto alla config originale.
# ============================================================================

set -e  # Exit on any error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}"
echo "================================================================================"
echo "PHASE 1 TRAINING - Optimized for Memory (CUDA)"
echo "================================================================================"
echo -e "${NC}"

# Check if .venv exists
if [ ! -d ".venv" ]; then
    echo -e "${YELLOW}[!] Virtualenv not found. Creating...${NC}"
    python3 -m venv .venv
fi

# Activate virtual environment
echo -e "${BLUE}[1/3] Activating virtual environment...${NC}"
source .venv/bin/activate

# Install dependencies (optional)
# python -m pip install --quiet -r requirements.txt

# Set CUDA memory optimization
echo -e "${BLUE}[2/3] Setting CUDA memory optimization...${NC}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo -e "${GREEN}✓ PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True${NC}"

# Run training
echo -e "${BLUE}[3/3] Starting Phase 1 training...${NC}"
echo -e "${YELLOW}"
echo "Configuration applied:"
echo "  - Batch size: 16 (reduced from 32)"
echo "  - Video max frames: 100 (reduced from 150)"
echo "  - Text max length: 256 (reduced from 512)"
echo "  - Mixed Precision Training: ENABLED"
echo "  - CUDA memory optimization: ENABLED"
echo -e "${NC}"
echo "================================================================================"
echo ""

python train_phase1.py "$@"

echo ""
echo -e "${GREEN}✓ Training completed${NC}"
