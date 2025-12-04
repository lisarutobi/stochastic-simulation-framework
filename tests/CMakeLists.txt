file(GLOB TEST_SOURCES *.cpp)

add_executable(unit_tests
    ${TEST_SOURCES}
)

target_link_libraries(unit_tests PRIVATE
    stochastic_framework
    Catch2::Catch2WithMain
)

include(Catch)
catch_discover_tests(unit_tests)