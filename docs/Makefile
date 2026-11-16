# === CONFIGURATION ===
TEX_FILE    = main.tex
LOG_FILE	= $(TEX_FILE:.tex=.log)
PDF_OUT     = $(TEX_FILE:.tex=.pdf)
BUILD_DIR   = .build
PROJ_DIRS	= chapters
OUT_DIR     = $(abspath $(BUILD_DIR)/..)
FINAL_NAME  = Relazione RoboLiszt MAGGIO Francesco
TAG         = FDR

SHELL       := /bin/zsh

COMPILER    = latexmk
FLAGS  		= -pdf -synctex=1 -file-line-error -output-directory=$(BUILD_DIR) -cd
COMP_FLAGS  = $(FLAGS) -interaction=nonstopmode -f
DEBUG_FLAGS = $(FLAGS) -interaction=errorstopmode
WATCH_FLAGS = $(FLAGS) -pvc -view=none

.PHONY: all prepare compile post-process watch debug log clean view purge

all: prepare compile post-process

prepare:
	@mkdir -p $(BUILD_DIR)
	@mkdir -p $(addprefix $(BUILD_DIR)/, $(PROJ_DIRS))

# === COMPILATION ===
compile:
	@echo "⚙️  Start compiling..."
	-$(COMPILER) $(COMP_FLAGS) $(TEX_FILE)

# === POST-COMPILATION ===
post-process:
	@echo "--------------------------------------------------"; \
	if [ -f "$(BUILD_DIR)/$(LOG_FILE)" ]; then \
		grep -A 5 "^!" "$(BUILD_DIR)/$(LOG_FILE)" || true; \
	else \
		echo "No log file found."; \
	fi; \
	echo "--------------------------------------------------"; \
	if [ -f "$(BUILD_DIR)/$(PDF_OUT)" ]; then \
		echo "✅ PDF successfully generated!"; \
		cp "$(BUILD_DIR)/$(PDF_OUT)" "$(abspath $(BUILD_DIR)/..)/$(FINAL_NAME).pdf"; \
		if command -v tag >/dev/null 2>&1; then \
			tag -a "$(TAG)" "$(abspath $(BUILD_DIR)/..)/$(FINAL_NAME).pdf"; \
			echo "🔴 Tag '$(TAG)' added."; \
		fi; \
		echo "🚀 Opening PDF..."; \
		open "$(abspath $(BUILD_DIR)/..)/$(FINAL_NAME).pdf"; \
	else \
		echo "❌ No PDF found. Compilation failed fatally."; \
		exit 1; \
	fi

# === UTILS ===
log:
	@if [ -f "$(BUILD_DIR)/$(LOG_FILE)" ]; then \
		open "$(BUILD_DIR)/$(LOG_FILE)"; \
	else \
		echo "❌ No log file found. Please recompile first!"; \
	fi

view:
	@open "$(abspath $(BUILD_DIR)/..)/$(FINAL_NAME).pdf" || open "$(BUILD_DIR)/$(PDF_OUT)"
	
watch: prepare
	@echo "👀 Watching for changes..."
	@if [ -f "$(BUILD_DIR)/$(PDF_OUT)" ]; then \
		open "$(BUILD_DIR)/$(PDF_OUT)"; \
	fi
	-$(COMPILER) $(WATCH_FLAGS) $(TEX_FILE) 
	
debug: prepare
	@echo "🐞 Debugging mode..." 
	-$(COMPILER) $(DEBUG_FLAGS) $(TEX_FILE)

# === CLEAN ===
clean:
	@echo "🧹 Cleaning temporary files..."
	rm -rf $(BUILD_DIR)

# === DEEP CLEAN ===
purge:
	@echo "🧨 Purging directory... keeping only $(TEX_FILE) and Makefile"
	@find . -maxdepth 1 ! -name '$(TEX_FILE)' ! -name 'Makefile' ! -name '.' -exec rm -rf {} +
	@echo "✨ Clean up completely!"