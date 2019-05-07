CC		= g++
CFLAGS	= -g -pedantic -W -Wall
#CFLAGS	= -O3 -finline-functions -ffast-math -fomit-frame-pointer -funroll-loops

LDFLAGS	= 
#INCPATH	=       = -I. -I.

OBJECTS	= main.o ser_expInt.o utils.o

TARGET	= exponentialIntegral.out

all: $(OBJECTS)
	$(CC) $(CFLAGS) $^ -o $(TARGET) $(LDFLAGS)

.PHONY: clean

clean:
	$(RM) $(OBJECTS) $(TARGET)
