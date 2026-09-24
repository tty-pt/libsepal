# Examples Makefile for libsepal

CC := cc
CFLAGS := -g -Wall -Wextra -I../include -I/home/quirinpa/site/external/libqmap/include
LDFLAGS := -L../lib -L/home/quirinpa/site/external/libqmap/lib -L/usr/lib
LDLIBS := -lsepal -lqmap -lqsys -lxxhash -lm

export LD_LIBRARY_PATH := $(abspath ../lib):/home/quirinpa/site/external/libqmap/lib:$(LD_LIBRARY_PATH)

EXAMPLES := basic twostage persistence kernel_form

all: $(EXAMPLES)

%: %.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS) $(LDLIBS)

run: all
	for e in $(EXAMPLES); do echo "--- $$e"; ./$$e $(ARGS) || exit 1; done

clean:
	rm -f $(EXAMPLES) *.o

.PHONY: all run clean
