#!/bin/bash
# Type checking helper script for CREST Demand Model

set -e

# Get script directory and project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Change to project root
cd "$PROJECT_ROOT"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "🔍 Running mypy type checker..."
echo ""

# Run mypy and capture output (use python if venv doesn't exist)
MYPY_CMD="python3 -m mypy"
if [ -f "venv/bin/mypy" ]; then
    MYPY_CMD="venv/bin/mypy"
fi

if $MYPY_CMD python/crest/ --config-file python/mypy.ini "$@" > /tmp/mypy_output.txt 2>&1; then
    echo -e "${GREEN}✅ No type errors found!${NC}"
    exit 0
else
    # Parse results
    ERROR_COUNT=$(tail -1 /tmp/mypy_output.txt | grep -oP '\d+(?= error)' || echo "0")
    FILE_COUNT=$(tail -1 /tmp/mypy_output.txt | grep -oP '\d+(?= file)' || echo "0")

    echo -e "${RED}❌ Found $ERROR_COUNT errors in $FILE_COUNT files${NC}"
    echo ""

    # Show summary by error type
    echo "📊 Error breakdown:"
    echo ""
    grep -oP '\[.*?\]' /tmp/mypy_output.txt | sort | uniq -c | sort -rn | head -10
    echo ""

    # Show files with most errors
    echo "📄 Files with most errors:"
    echo ""
    grep "error:" /tmp/mypy_output.txt | cut -d: -f1 | sort | uniq -c | sort -rn | head -10
    echo ""

    echo "💡 Tips:"
    echo "  • View full output: cat /tmp/mypy_output.txt"
    echo "  • Check specific file: $MYPY_CMD python/crest/core/building.py"
    echo "  • Ignore errors in file: # type: ignore (at end of line)"
    echo "  • See config: cat python/mypy.ini"
    echo ""

    exit 1
fi
