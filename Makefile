NPM ?= npm
ORT_VERSION ?= 1.23.2
ORT_BASE ?= https://github.com/microsoft/onnxruntime/releases/download/v$(ORT_VERSION)
ORT_DIR ?= target/onnxruntime

UNAME_S := $(shell uname)
UNAME_M := $(shell uname -m)

ifeq ($(UNAME_S),Darwin)
	ifeq ($(UNAME_M),arm64)
		ORT_TARBALL ?= onnxruntime-osx-arm64-$(ORT_VERSION).tgz
	else
		ORT_TARBALL ?= onnxruntime-osx-x86_64-$(ORT_VERSION).tgz
	endif
	ORT_DYLIB_BASENAME := libonnxruntime.$(ORT_VERSION).dylib
else ifeq ($(OS),Windows_NT)
	ORT_TARBALL ?= onnxruntime-win-x64-$(ORT_VERSION).zip
	ORT_DYLIB_BASENAME := onnxruntime.dll
else
	ifeq ($(UNAME_M),aarch64)
		ORT_TARBALL ?= onnxruntime-linux-aarch64-$(ORT_VERSION).tgz
	else
		ORT_TARBALL ?= onnxruntime-linux-x64-$(ORT_VERSION).tgz
	endif
	ORT_DYLIB_BASENAME := libonnxruntime.so.$(ORT_VERSION)
endif

ORT_DIR_NAME_TMP := $(ORT_TARBALL:.tgz=)
ORT_DIR_NAME_TMP := $(ORT_DIR_NAME_TMP:.tar.gz=)
ORT_DIR_NAME ?= $(ORT_DIR_NAME_TMP:.zip=)
ORT_LIB_DIR ?= $(ORT_DIR)/$(ORT_DIR_NAME)/lib
ORT_DYLIB_FILE ?= $(ORT_LIB_DIR)/$(ORT_DYLIB_BASENAME)
ORT_DYLIB_FILE_ABS := $(shell pwd)/$(ORT_DYLIB_FILE)
ORT_LIB_DIR_ABS := $(shell pwd)/$(ORT_LIB_DIR)

ifeq ($(UNAME_S),Darwin)
	ORT_ENV_VARS = ORT_DYLIB_PATH=$(ORT_DYLIB_FILE_ABS)
else ifeq ($(OS),Windows_NT)
	ORT_ENV_VARS = ORT_DYLIB_PATH=$(ORT_DYLIB_FILE_ABS)
else
	ORT_ENV_VARS = LD_LIBRARY_PATH=$(ORT_LIB_DIR_ABS):$$LD_LIBRARY_PATH ORT_DYLIB_PATH=$(ORT_DYLIB_FILE_ABS)
endif

.PHONY: install build demo demo-only onnxruntime-download clean help

install:
	$(NPM) install

build:
	$(NPM) run build

onnxruntime-download:
	@if [ -d "$(ORT_LIB_DIR)" ]; then \
		echo "ONNX Runtime already downloaded at $(ORT_LIB_DIR)"; \
	else \
		echo "Downloading ONNX Runtime $(ORT_VERSION)..."; \
		mkdir -p $(ORT_DIR); \
		curl -L $(ORT_BASE)/$(ORT_TARBALL) -o $(ORT_DIR)/$(ORT_TARBALL); \
		if echo "$(ORT_TARBALL)" | grep -q '\\.zip$$'; then \
			unzip -q $(ORT_DIR)/$(ORT_TARBALL) -d $(ORT_DIR); \
		else \
			tar -xzf $(ORT_DIR)/$(ORT_TARBALL) -C $(ORT_DIR); \
		fi; \
		echo "[OK] ONNX Runtime downloaded and extracted"; \
	fi

demo: onnxruntime-download
	$(ORT_ENV_VARS) $(NPM) run demo

demo-only: onnxruntime-download
	$(ORT_ENV_VARS) $(NPM) --prefix demo run demo

clean:
	$(NPM) run clean
	rm -rf target

help:
	@echo "webnnjs - Available Targets"
	@echo "==========================="
	@echo "  make install              - Install npm dependencies"
	@echo "  make build                - Build native addon + TypeScript packages"
	@echo "  make onnxruntime-download - Download ONNX Runtime shared library"
	@echo "  make demo                 - Build and run demo with ORT env set"
	@echo "  make demo-only            - Run already-built demo with ORT env set"
	@echo "  make clean                - Clean build artifacts and downloaded ORT"
	@echo ""
	@echo "Variables:"
	@echo "  ORT_VERSION=$(ORT_VERSION)"
	@echo "  ORT_DIR=$(ORT_DIR)"
