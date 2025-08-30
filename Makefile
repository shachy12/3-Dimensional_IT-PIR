CC = gcc
CFLAGS = -Iincludes/ -O3 -Wall -Wextra -march=native -mavx2 -mpopcnt
SRC = src/main.c src/pir_client.c src/pir_server.c
OBJ = $(SRC:.c=.o)
TARGET = server

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(CFLAGS) -o $@ $^ -lm

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ) $(TARGET)