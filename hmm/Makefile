# Makefile for HMM Stock Prediction

CC = gcc
CFLAGS = -Wall -O3 -std=c99
LDFLAGS = -lm
TARGET = hmm
SOURCE = hmm.c

# Default target
all: $(TARGET)

# Compile the program
$(TARGET): $(SOURCE)
	$(CC) $(CFLAGS) $(SOURCE) -o $(TARGET) $(LDFLAGS)

# Run the program
run: $(TARGET)
	./$(TARGET)

# Clean compiled files
clean:
	rm -f $(TARGET)

# Install dependencies (none needed for C)
install:
	@echo "No dependencies to install for C implementation"

# Help target
help:
	@echo "Available targets:"
	@echo "  all     - Compile the program (default)"
	@echo "  run     - Compile and run the program"
	@echo "  clean   - Remove compiled files"
	@echo "  install - Install dependencies (none needed)"
	@echo "  help    - Show this help message"

.PHONY: all run clean install help 