all := libsepal

LDLIBS-libsepal := -lqsys -lqmap -lxxhash -lm
LDFLAGS-libsepal := -L/home/quirinpa/site/external/libqmap/lib

CFLAGS := -g
CFLAGS += -O3 -mpopcnt -mavx2 -mfma
CFLAGS += -I/home/quirinpa/site/external/libqmap/include

include ../mk/include.mk

test: all
	$(MAKE) -C tests test

bench: all
	$(MAKE) -C tests bench