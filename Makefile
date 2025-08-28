CC = gcc
CFLAGS = -Isrc/ -O3 -Wall -Wextra -march=native -mavx2 -mpopcnt
SRC = src/server/main.c src/utils/vector_utils.c
OBJ = $(SRC:.c=.o)
TARGET = server

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(CFLAGS) -o $@ $^ -lm

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ) $(TARGET)