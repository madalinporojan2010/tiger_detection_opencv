#ifndef FILE_TESTER_H
#define FILE_TESTER_H
#include <string>
#include <vector>
#include <utility>

#define INPUT_TEST_DIR ".\\images\\specimens"
#define INPUT_TEST_DIR_LEN 18
#define OUTPUT_TEST_DIR ".\\images\\tests\\test_"
#define TEST_PROPS_FILENAME "test_info_README.txt"


namespace file_tester {
	std::pair<std::vector<std::string>, std::vector<std::string>> obtainFileNames();

	void makeTestDirs(int testNumber, std::vector<std::string> dirs, int iterations, int Kclusters, int patchSize, std::string heuristicFuncName);

	void createInfoFile(int testNumber, int iterations, int Kclusters, int patchSize, std::string heuristicFuncName);
};


#endif