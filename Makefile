CC ?= gcc
ARCH := $(shell uname -m)
ifeq ($(ARCH),aarch64)
ARCHFLAGS ?= -march=armv8.2-a+dotprod+fp16
else
ARCHFLAGS ?= -march=x86-64
endif
CFLAGS ?= -O3 $(ARCHFLAGS) -fPIC -Wall -Wextra -Wno-unused-parameter -std=c11 -fopenmp
LDFLAGS ?= -lm -fopenmp -luring -lroaring -lxxhash -lpthread -lcurl
CORE := src/anchor.o src/anchor_live.o

all: product
product: fissiondb-engine libfissiondb_anchor.so
fissiondb-engine: $(CORE) src/anchor_main.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
libfissiondb_anchor.so: $(CORE)
	$(CC) $(CFLAGS) -shared -Wl,--no-undefined -o $@ $^ $(LDFLAGS)
%.o: %.c $(wildcard src/*.h src/*.inc)
	$(CC) $(CFLAGS) -c $< -o $@
test: product
	PYTHONPATH=scripts OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m pytest tests -q
clean:
	rm -f $(CORE) src/anchor_main.o fissiondb-engine libfissiondb_anchor.so
.PHONY: all product test clean
