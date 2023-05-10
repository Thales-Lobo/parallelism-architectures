# ---------------------------- #
# -- SortieDeCache Makefile -- #
# ---------------------------- #

# -- File list ----------
FILE = main.cpp


# -- Paths ----------
SRC_PATH = src
OBJ_PATH = obj
EXE_PATH = exe

# -- Macros ----------
CC = g++-12

# -- Flags ----------
C_OPTIMISATION_FLAGS = -O3 -std=c++17 -g

CFLAGS = $(C_OPTIMISATION_FLAGS)
LDFLAGS = $(C_OPTIMISATION_FLAGS)

# -- Final product ----------
PRODUCT   = test.exe

# -- src and obj List ----------
SRC = $(addprefix ${SRC_PATH}/, $(FILE))
OBJ = $(addprefix ${OBJ_PATH}/, $(addsuffix .o, $(basename $(FILE))))

# -- Base rules ----------
$(OBJ_PATH)/%.o : $(SRC_PATH)/%.cpp
	$(CC) $(CFLAGS) -c $< -o $@

#-----Main rule ----------
$(EXE_PATH)/$(PRODUCT): $(OBJ)
	$(CC) -o $@ $^ $(LDFLAGS) $(INC) $(LIB_LIB_PATH) -lm

# -- Cleanup ----------
clean:
	rm -f $(OBJ)
	rm -f $(PRODUCT)
 