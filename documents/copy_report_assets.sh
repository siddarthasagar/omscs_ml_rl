#!/usr/bin/env bash
#
# copy_report_assets.sh
#
# Copy generated figures and LaTeX tables from artifacts/ to REPORT_RL/
# This ensures the LaTeX report uses the latest generated assets.
#
# Usage:
#   bash copy_report_assets.sh
#   bash copy_report_assets.sh --dry-run  # Preview changes without applying

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Anchor all paths to the repo root (one level up from this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# Directories
RESULTS_FIGURES="$REPO_ROOT/artifacts/figures"
RESULTS_TABLES="$REPO_ROOT/artifacts/tables"
REPORT_FIGURES="$REPO_ROOT/REPORT_RL/figures"
REPORT_TABLES="$REPO_ROOT/REPORT_RL/tables"

# Parse arguments
DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
    echo -e "${YELLOW}DRY RUN MODE - No changes will be made${NC}"
    echo ""
fi

# Function to print section headers
print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

# Function to print success
print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

# Function to print warning
print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

# Function to print error
print_error() {
    echo -e "${RED}✗${NC} $1"
}

# ============================================================================
# PRE-FLIGHT CHECKS
# ============================================================================

print_header "Pre-flight Checks"

# Check if source directories exist
MISSING_DIRS=()

if [[ ! -d "$RESULTS_FIGURES" ]]; then
    MISSING_DIRS+=("$RESULTS_FIGURES")
fi

if [[ ! -d "$RESULTS_TABLES" ]]; then
    MISSING_DIRS+=("$RESULTS_TABLES")
fi

if [[ ${#MISSING_DIRS[@]} -gt 0 ]]; then
    print_error "Required source directories not found:"
    for dir in "${MISSING_DIRS[@]}"; do
        echo "  - $dir"
    done
    echo ""
    echo -e "${YELLOW}You need to generate report assets first:${NC}"
    echo ""
    echo "  make phase2 phase3 phase4 phase5 phase6 phase7 phase8"
    echo ""
    exit 1
fi

# Check if source directories have files
FIGURE_COUNT=$(find "$RESULTS_FIGURES" -type f \( -name "*.png" -o -name "*.pdf" -o -name "*.jpg" \) 2>/dev/null | wc -l | tr -d ' ')
TABLE_COUNT=$(find "$RESULTS_TABLES" -maxdepth 1 -type f -name "*.tex" 2>/dev/null | wc -l | tr -d ' ')

if [[ "$FIGURE_COUNT" -eq 0 ]]; then
    print_error "No figures found in $RESULTS_FIGURES"
    echo ""
    echo "Generate figures first:"
    echo "  make phase2 phase3 phase4 phase5 phase6 phase7"
    echo ""
    exit 1
fi

if [[ "$TABLE_COUNT" -eq 0 ]]; then
    print_error "No LaTeX tables found in $RESULTS_TABLES"
    echo ""
    echo "Generate tables first:"
    echo "  make phase8"
    echo ""
    exit 1
fi

print_success "Source directories exist"
print_success "Found $FIGURE_COUNT figure(s)"
print_success "Found $TABLE_COUNT LaTeX table(s)"
echo ""

# Create REPORT_RL directories if they don't exist
if [[ "$DRY_RUN" == false ]]; then
    mkdir -p "$REPORT_FIGURES"
    mkdir -p "$REPORT_TABLES"
fi

# ============================================================================
# COPY FIGURES
# ============================================================================

print_header "Copying Figures"

echo "Copying $FIGURE_COUNT figure(s) from $RESULTS_FIGURES to $REPORT_FIGURES..."
echo ""

if [[ "$DRY_RUN" == false ]]; then
    # Remove old figures
    if [[ -d "$REPORT_FIGURES" ]]; then
        rm -rf "$REPORT_FIGURES"
        mkdir -p "$REPORT_FIGURES"
        print_success "Cleaned $REPORT_FIGURES"
    fi
    
    # Copy new figures (preserving phase subdirectory structure)
    cp -r "$RESULTS_FIGURES"/. "$REPORT_FIGURES/"

    # Verify copy
    COPIED_COUNT=$(find "$REPORT_FIGURES" -type f \( -name "*.png" -o -name "*.pdf" -o -name "*.jpg" \) | wc -l | tr -d ' ')
    print_success "Copied $COPIED_COUNT figure(s)"

    # List copied files
    echo ""
    echo "Copied figures:"
    find "$REPORT_FIGURES" -type f \( -name "*.png" -o -name "*.pdf" -o -name "*.jpg" \) | sed "s|$REPORT_FIGURES/||" | sort | sed 's/^/  - /'
else
    echo "Would remove: $REPORT_FIGURES/*"
    echo "Would copy (recursive):"
    find "$RESULTS_FIGURES" -type f \( -name "*.png" -o -name "*.pdf" -o -name "*.jpg" \) | sed "s|$RESULTS_FIGURES/||" | sort | sed 's/^/  - /'
fi

echo ""

# ============================================================================
# COPY LATEX TABLES
# ============================================================================

print_header "Copying LaTeX Tables"

echo "Copying $TABLE_COUNT LaTeX table(s) from $RESULTS_TABLES to $REPORT_TABLES..."
echo ""

if [[ "$DRY_RUN" == false ]]; then
    # Remove old tables
    if [[ -d "$REPORT_TABLES" ]]; then
        rm -rf "$REPORT_TABLES"
        mkdir -p "$REPORT_TABLES"
        print_success "Cleaned $REPORT_TABLES"
    fi
    
    # Copy new tables
    cp "$RESULTS_TABLES"/*.tex "$REPORT_TABLES/" 2>/dev/null || true
    
    # Verify copy
    COPIED_COUNT=$(find "$REPORT_TABLES" -maxdepth 1 -type f | wc -l | tr -d ' ')
    print_success "Copied $COPIED_COUNT table(s)"
    
    # List copied files with preview
    echo ""
    echo "Copied tables:"
    for table in "$REPORT_TABLES"/*.tex; do
        if [[ -f "$table" ]]; then
            filename=$(basename "$table")
            line_count=$(wc -l < "$table" | tr -d ' ')
            echo "  - $filename ($line_count lines)"
        fi
    done
else
    echo "Would remove: $REPORT_TABLES/*"
    echo "Would copy:"
    ls -1 "$RESULTS_TABLES"/*.tex 2>/dev/null | sed 's/^/  - /' || echo "  (no files)"
fi

echo ""

# ============================================================================
# SUMMARY
# ============================================================================

print_header "Summary"

if [[ "$DRY_RUN" == false ]]; then
    print_success "Copy complete!"
    echo ""
    echo "Next steps:"
    echo "  1. Review the copied files in REPORT_RL/"
    echo "  2. Update LaTeX to use the new tables (copy-paste from REPORT_RL/tables/*.tex)"
    echo "  3. Compile LaTeX: cd REPORT_RL && pdflatex RL_Report_schinne3.tex"
else
    print_warning "Dry run complete - no changes made"
    echo ""
    echo "Run without --dry-run to apply changes:"
    echo "  bash copy_report_assets.sh"
fi

echo ""
