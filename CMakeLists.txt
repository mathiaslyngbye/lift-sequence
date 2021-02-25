cmake_minimum_required(VERSION 2.8)
project(lift-sequence)

# OpenCV
find_package( OpenCV REQUIRED )

# Set dependencies
set(DEPENDENCIES ${OpenCV_LIBS})

add_executable(${PROJECT_NAME} "main.cpp")
target_link_libraries(${PROJECT_NAME} stdc++fs ${DEPENDENCIES})
