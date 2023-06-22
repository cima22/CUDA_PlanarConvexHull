CC = nvcc
RM ?= @rm
MKDIR ?= @mkdir

CFLAGS := -O3

SRC_DIR = src
SRC_QH_DIR = $(SRC_DIR)/quickhull
OBJ_DIR = obj
BIN_DIR = bin
DATA_DIR = data
TEST_DIR = test

SRCS = $(wildcard $(SRC_QH_DIR)/*.cu)
SRCS += $(wildcard $(SRC_QH_DIR)/*.cpp)
OBJECTS = $(patsubst $(SRC_QH_DIR)/%.cu,$(OBJ_DIR)/%.o,$(SRCS))
OBJECTS := $(patsubst $(SRC_QH_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(OBJECTS))
BINARIES = $(patsubst $(OBJ_DIR)/%.o,$(BIN_DIR)/%.x,$(OBJECTS))

all: $(BIN_DIR) $(OBJ_DIR) $(BINARIES)

$(DATA_DIR):
	@echo "Creating data directory: $(DATA_DIR)"
	$(MKDIR) $(DATA_DIR)

$(BIN_DIR):
	@echo "Creating binary directory: $(BIN_DIR)"
	$(MKDIR) $(BIN_DIR)

$(BIN_DIR)/%.x: $(OBJ_DIR)/%.o $(OBJ_DIR)/points_generator.o
	@echo "Generating binary $@"
	$(CC) $(CFLAGS) -o $@ $^

$(OBJ_DIR):
	@echo "Creating build directory: $(OBJ_DIR)"
	$(MKDIR) $(OBJ_DIR)

$(OBJ_DIR)/points_generator.o: $(SRC_DIR)/points_generation/points_generator.cpp
	$(CC) $(CFLAGS) -c -o $@ $^

$(OBJ_DIR)/%.o: $(SRC_QH_DIR)/%.cu
	@echo "Compiling $<"
	$(CC) $(CFLAGS) -c -o $@ $^

$(OBJ_DIR)/%.o: $(SRC_QH_DIR)/%.cpp
	@echo "Compiling $<"
	$(CC) $(CFLAGS) -c -o $@ $^

clean:
	@echo "Cleaning build directories: $(OBJ_DIR), $(BIN_DIR)"
	$(RM) -Rf $(OBJ_DIR)
	$(RM) -Rf $(BIN_DIR)

.PHONY: clean
