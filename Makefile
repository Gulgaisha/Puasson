CC = mpicxx
SRCS = $(wildcard *.c)
HDRS = $(wildcard *.h)
OBJS = $(SRCS:.c=.o)
DIRS = $(subst /, ,$(CURDIR))
PROJ = main

APP = $(PROJ)
CFLAGS= -c -O3
LDFLAGS=
LIBS= -O3 -fopenmp

all: $(APP)

$(APP): $(OBJS)
	$(CC) $(INCLUDE_DIR)  $(LDFLAGS) $(OBJS) -o $(APP) $(LIBS)

bluegene-openmp: 
	mpixlcxx_r $(CFLAGS) -qsmp=omp main.c -o main.o
	mpixlcxx_r $(INCLUDE_DIR)  $(LDFLAGS) $(OBJS) -o $(APP) -O3 -qsmp=omp

%.o: %.c $(HDRS) $(MF)
	$(CC) $(INCLUDE_DIR)  $(CFLAGS) $< -o $@

