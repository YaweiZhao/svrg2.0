BIN_NAME = linsvrg
BIN_DIR = bin
TARGET = $(BIN_DIR)/$(BIN_NAME)
CFLAGS = -Isrc/com -O2

CPP_FILES= 	\
	src/com/AzDmat.cpp \
	src/com/AzParam.cpp \
	src/com/AzSmat.cpp \
	src/com/AzStrPool.cpp \
	src/com/AzSvDataS.cpp \
	src/com/AzTools.cpp \
	src/com/AzUtil.cpp \
	src/lin/AzsLinear.cpp \
	src/lin/AzsLmod.cpp \
	src/lin/AzsSvrg.cpp \
	src/lin/driv_lin.cpp

#$(TARGET): $(CPP_FILES)
all: 
	/bin/rm -f $(TARGET)
	g++ $(CPP_FILES) $(CFLAGS) -g -o $(TARGET)

clean: 
	/bin/rm -f $(TARGET)
