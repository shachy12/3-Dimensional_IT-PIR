CC = gcc
CFLAGS = -Iincludes/ -O3 -Wall -Wextra -march=native -mavx2 -mpopcnt
SRC_DIR = src
SOURCES = $(addprefix $(SRC_DIR)/, main.c pir_client.c pir_server.c)
BIN_DIR = bin
TARGET = server

all: $(TARGET)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(TARGET): $(BIN_DIR) $(SOURCES)
	$(CC) $(CFLAGS) -o $(BIN_DIR)/$@ $(SOURCES) -lm

clean:
	rm -rf $(BIN_DIR)