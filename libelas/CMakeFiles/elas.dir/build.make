# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.12

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = "/home/camilojd/Universidad/Primavera 2018/EL7008/proyecto/libelas"

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = "/home/camilojd/Universidad/Primavera 2018/EL7008/proyecto/libelas"

# Include any dependencies generated for this target.
include CMakeFiles/elas.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/elas.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/elas.dir/flags.make

CMakeFiles/elas.dir/src/descriptor.cpp.o: CMakeFiles/elas.dir/flags.make
CMakeFiles/elas.dir/src/descriptor.cpp.o: src/descriptor.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir="/home/camilojd/Universidad/Primavera 2018/EL7008/proyecto/libelas/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/elas.dir/src/descriptor.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/elas.dir/src/descriptor.cpp.o -c "/home/camilojd/Universidad/Primavera 2018/EL7008/proyecto/libelas/src/descriptor.cpp"

CMakeFiles/elas.dir/src/descriptor.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/elas.dir/src/descriptor.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/home/camilojd/Universidad/Primavera 2018/EL7008/proyecto/libelas/src/descriptor.cpp" > CMakeFiles/elas.dir/src/descriptor.cpp.i

CMakeFiles/elas.dir/src/descriptor.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/elas.dir/src/descriptor.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/home/camilojd/Universidad/Primavera 2018/EL7008/proyecto/libelas/src/descriptor.cpp" -o CMakeFiles/elas.dir/src/descriptor.cpp.s

CMakeFiles/elas.dir/src/elas.cpp.o: CMakeFiles/elas.dir/flags.make
CMakeFiles/elas.dir/src/elas.cpp.o: src/elas.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir="/home/camilojd/Universidad/Primavera 2018/EL7008/proyecto/libelas/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/elas.dir/src/elas.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/elas.dir/src/elas.cpp.o -c "/home/camilojd/Universidad/Primavera 2018/EL7008/proyecto/libelas/src/elas.cpp"

CMakeFiles/elas.dir/src/elas.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/elas.dir/src/elas.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/home/camilojd/Universidad/Primavera 2018/EL7008/proyecto/libelas/src/elas.cpp" > CMakeFiles/elas.dir/src/elas.cpp.i

CMakeFiles/elas.dir/src/elas.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/elas.dir/src/elas.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/home/camilojd/Universidad/Primavera 2018/EL7008/proyecto/libelas/src/elas.cpp" -o CMakeFiles/elas.dir/src/elas.cpp.s

CMakeFiles/elas.dir/src/filter.cpp.o: CMakeFiles/elas.dir/flags.make
CMakeFiles/elas.dir/src/filter.cpp.o: src/filter.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir="/home/camilojd/Universidad/Primavera 2018/EL7008/proyecto/libelas/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/elas.dir/src/filter.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/elas.dir/src/filter.cpp.o -c "/home/camilojd/Universidad/Primavera 2018/EL7008/proyecto/libelas/src/filter.cpp"

CMakeFiles/elas.dir/src/filter.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/elas.dir/src/filter.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/home/camilojd/Universidad/Primavera 2018/EL7008/proyecto/libelas/src/filter.cpp" > CMakeFiles/elas.dir/src/filter.cpp.i

CMakeFiles/elas.dir/src/filter.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/elas.dir/src/filter.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/home/camilojd/Universidad/Primavera 2018/EL7008/proyecto/libelas/src/filter.cpp" -o CMakeFiles/elas.dir/src/filter.cpp.s

CMakeFiles/elas.dir/src/main.cpp.o: CMakeFiles/elas.dir/flags.make
CMakeFiles/elas.dir/src/main.cpp.o: src/main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir="/home/camilojd/Universidad/Primavera 2018/EL7008/proyecto/libelas/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/elas.dir/src/main.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/elas.dir/src/main.cpp.o -c "/home/camilojd/Universidad/Primavera 2018/EL7008/proyecto/libelas/src/main.cpp"

CMakeFiles/elas.dir/src/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/elas.dir/src/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/home/camilojd/Universidad/Primavera 2018/EL7008/proyecto/libelas/src/main.cpp" > CMakeFiles/elas.dir/src/main.cpp.i

CMakeFiles/elas.dir/src/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/elas.dir/src/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/home/camilojd/Universidad/Primavera 2018/EL7008/proyecto/libelas/src/main.cpp" -o CMakeFiles/elas.dir/src/main.cpp.s

CMakeFiles/elas.dir/src/matrix.cpp.o: CMakeFiles/elas.dir/flags.make
CMakeFiles/elas.dir/src/matrix.cpp.o: src/matrix.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir="/home/camilojd/Universidad/Primavera 2018/EL7008/proyecto/libelas/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object CMakeFiles/elas.dir/src/matrix.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/elas.dir/src/matrix.cpp.o -c "/home/camilojd/Universidad/Primavera 2018/EL7008/proyecto/libelas/src/matrix.cpp"

CMakeFiles/elas.dir/src/matrix.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/elas.dir/src/matrix.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/home/camilojd/Universidad/Primavera 2018/EL7008/proyecto/libelas/src/matrix.cpp" > CMakeFiles/elas.dir/src/matrix.cpp.i

CMakeFiles/elas.dir/src/matrix.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/elas.dir/src/matrix.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/home/camilojd/Universidad/Primavera 2018/EL7008/proyecto/libelas/src/matrix.cpp" -o CMakeFiles/elas.dir/src/matrix.cpp.s

CMakeFiles/elas.dir/src/triangle.cpp.o: CMakeFiles/elas.dir/flags.make
CMakeFiles/elas.dir/src/triangle.cpp.o: src/triangle.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir="/home/camilojd/Universidad/Primavera 2018/EL7008/proyecto/libelas/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object CMakeFiles/elas.dir/src/triangle.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/elas.dir/src/triangle.cpp.o -c "/home/camilojd/Universidad/Primavera 2018/EL7008/proyecto/libelas/src/triangle.cpp"

CMakeFiles/elas.dir/src/triangle.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/elas.dir/src/triangle.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/home/camilojd/Universidad/Primavera 2018/EL7008/proyecto/libelas/src/triangle.cpp" > CMakeFiles/elas.dir/src/triangle.cpp.i

CMakeFiles/elas.dir/src/triangle.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/elas.dir/src/triangle.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/home/camilojd/Universidad/Primavera 2018/EL7008/proyecto/libelas/src/triangle.cpp" -o CMakeFiles/elas.dir/src/triangle.cpp.s

# Object files for target elas
elas_OBJECTS = \
"CMakeFiles/elas.dir/src/descriptor.cpp.o" \
"CMakeFiles/elas.dir/src/elas.cpp.o" \
"CMakeFiles/elas.dir/src/filter.cpp.o" \
"CMakeFiles/elas.dir/src/main.cpp.o" \
"CMakeFiles/elas.dir/src/matrix.cpp.o" \
"CMakeFiles/elas.dir/src/triangle.cpp.o"

# External object files for target elas
elas_EXTERNAL_OBJECTS =

elas: CMakeFiles/elas.dir/src/descriptor.cpp.o
elas: CMakeFiles/elas.dir/src/elas.cpp.o
elas: CMakeFiles/elas.dir/src/filter.cpp.o
elas: CMakeFiles/elas.dir/src/main.cpp.o
elas: CMakeFiles/elas.dir/src/matrix.cpp.o
elas: CMakeFiles/elas.dir/src/triangle.cpp.o
elas: CMakeFiles/elas.dir/build.make
elas: CMakeFiles/elas.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir="/home/camilojd/Universidad/Primavera 2018/EL7008/proyecto/libelas/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_7) "Linking CXX executable elas"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/elas.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/elas.dir/build: elas

.PHONY : CMakeFiles/elas.dir/build

CMakeFiles/elas.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/elas.dir/cmake_clean.cmake
.PHONY : CMakeFiles/elas.dir/clean

CMakeFiles/elas.dir/depend:
	cd "/home/camilojd/Universidad/Primavera 2018/EL7008/proyecto/libelas" && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" "/home/camilojd/Universidad/Primavera 2018/EL7008/proyecto/libelas" "/home/camilojd/Universidad/Primavera 2018/EL7008/proyecto/libelas" "/home/camilojd/Universidad/Primavera 2018/EL7008/proyecto/libelas" "/home/camilojd/Universidad/Primavera 2018/EL7008/proyecto/libelas" "/home/camilojd/Universidad/Primavera 2018/EL7008/proyecto/libelas/CMakeFiles/elas.dir/DependInfo.cmake" --color=$(COLOR)
.PHONY : CMakeFiles/elas.dir/depend

